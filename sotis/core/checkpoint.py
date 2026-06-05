"""
sotis.core.checkpoint
=====================
Lightweight, incremental workspace checkpointing using Git-style unified diffs.

Design rationale (per approved spec)
--------------------------------------
Full directory snapshots are expensive and unnecessary for an MVP. The goal
is reliable rollback and state recovery, not VM-level reproducibility.

Strategy:
    1.  Before a subtask begins (or after a context reset fires), Sotis records
        the current on-disk content of any files it considers "in scope" for
        the active session.
    2.  When a meltdown is detected, ``CheckpointManager.snapshot()`` computes
        a unified diff of every tracked file against its recorded baseline.
    3.  The resulting ``WorkspaceCheckpoint`` is a lightweight, JSON-serializable
        record containing:
            - The diff text for each modified file.
            - A flat list of added/removed files.
            - A copy of the ``ExecutionState`` trajectory summary.
    4.  On resumption, the checkpoint is passed to ``ContextResetter`` so it can
        narrate the workspace state inside the distilled prompt.

Diffing engine
--------------
Uses Python's built-in ``difflib.unified_diff`` — no external git dependency.
Diffs are produced in standard unified-diff format (3-line context).

Public API
----------
FileBaseline        : Snapshot of one file's content at tracking start.
FileDiff            : Computed diff for one file (added/removed/modified).
WorkspaceCheckpoint : Full serialisable checkpoint produced at meltdown time.
CheckpointManager   : Stateful manager used by the runtime.
"""

from __future__ import annotations

import difflib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from sotis.core.schemas import ExecutionState, MeltdownSignal


# ─────────────────────────────────────────────────────────────────────────────
# Invariant checking (Issue #2)
# ─────────────────────────────────────────────────────────────────────────────
#
# An InvariantChecker decides whether a workspace state is "good" — i.e. safe to
# resume from. It receives a snapshot of the current tracked-file contents as a
# ``{absolute_path: content}`` mapping and returns True if the invariant holds.
#
# This is what lets rollback target a *verified* state rather than just the most
# recent snapshot (which may itself be the silently-corrupted state that caused
# the meltdown). Invariants are deliberately cheap and run on the hot path; any
# exception they raise is caught and treated as "not verified" so a buggy or
# slow check can never crash the guard.
InvariantChecker = Callable[[Dict[str, str]], bool]


def always_valid(_files: Dict[str, str]) -> bool:
    """Default no-op invariant: every state is considered good."""
    return True


def python_imports_cleanly(_files: Dict[str, str]) -> bool:
    """
    Invariant: every tracked ``.py`` file parses without a SyntaxError.

    A cheap, dependency-free proxy for "the code still collects/imports" — a
    workspace where a tracked Python file no longer compiles is not a safe state
    to resume from. Non-Python files are ignored. Missing/empty content passes
    (nothing to break).
    """
    import ast
    for path, content in _files.items():
        if not path.endswith(".py"):
            continue
        if not content.strip():
            continue
        try:
            ast.parse(content)
        except SyntaxError:
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Low-level data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FileBaseline:
    """
    Content of a single file captured at the start of tracking.

    Attributes
    ----------
    path        : Absolute path to the file.
    content     : Raw text content at baseline time.
    captured_at : Unix epoch ms of capture.
    exists      : False if the file did not exist at baseline time.
    """
    path        : str
    content     : str
    captured_at : float = field(default_factory=lambda: time.time() * 1000)
    exists      : bool  = True


@dataclass
class FileDiff:
    """
    Computed diff for a single file against its baseline.

    Attributes
    ----------
    path          : Absolute path to the file.
    status        : One of 'modified', 'added', 'deleted', 'unchanged'.
    unified_diff  : Unified-diff text (empty string if unchanged or new/deleted).
    lines_added   : Number of + lines in the diff.
    lines_removed : Number of - lines in the diff.
    """
    path         : str
    status       : str   # 'modified' | 'added' | 'deleted' | 'unchanged'
    unified_diff : str   = ""
    lines_added  : int   = 0
    lines_removed: int   = 0

    @property
    def has_changes(self) -> bool:
        return self.status != "unchanged"


@dataclass
class WorkspaceCheckpoint:
    """
    Complete, JSON-serialisable checkpoint produced at meltdown time.

    Attributes
    ----------
    session_id         : Session identifier (from ExecutionState).
    subtask_id         : Active subtask at snapshot time.
    snapshot_at_step   : Global step index when the snapshot was taken.
    snapshot_at_ms     : Unix epoch ms of snapshot.
    completed_subtasks : List of subtask_ids already fully completed.
    active_subtask_desc: Human-readable description of the active subtask.
    trajectory_summary : Last N step events serialised as compact summaries.
    file_diffs         : Dict of {path: FileDiff} for all tracked files.
    reset_number       : Which reset attempt this checkpoint supports (1 or 2).
    meltdown_reason    : The MeltdownReason string from the triggering signal.
    """
    session_id          : str
    subtask_id          : Optional[str]
    snapshot_at_step    : int
    snapshot_at_ms      : float
    completed_subtasks  : List[str]
    active_subtask_desc : str
    trajectory_summary  : List[Dict]
    file_diffs          : Dict[str, FileDiff]
    reset_number        : int
    meltdown_reason     : str

    @property
    def modified_files(self) -> List[str]:
        """Paths of files with status 'modified'."""
        return [p for p, d in self.file_diffs.items() if d.status == "modified"]

    @property
    def added_files(self) -> List[str]:
        return [p for p, d in self.file_diffs.items() if d.status == "added"]

    @property
    def deleted_files(self) -> List[str]:
        return [p for p, d in self.file_diffs.items() if d.status == "deleted"]

    @property
    def total_lines_changed(self) -> int:
        return sum(d.lines_added + d.lines_removed for d in self.file_diffs.values())

    def to_json(self) -> str:
        """Serialise the checkpoint to a JSON string."""
        data = {
            "session_id"         : self.session_id,
            "subtask_id"         : self.subtask_id,
            "snapshot_at_step"   : self.snapshot_at_step,
            "snapshot_at_ms"     : self.snapshot_at_ms,
            "completed_subtasks" : self.completed_subtasks,
            "active_subtask_desc": self.active_subtask_desc,
            "trajectory_summary" : self.trajectory_summary,
            "file_diffs"         : {
                p: asdict(d) for p, d in self.file_diffs.items()
            },
            "reset_number"   : self.reset_number,
            "meltdown_reason": self.meltdown_reason,
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "WorkspaceCheckpoint":
        """Deserialise a checkpoint from a JSON string."""
        data = json.loads(raw)
        data["file_diffs"] = {
            p: FileDiff(**d) for p, d in data["file_diffs"].items()
        }
        return cls(**data)


# ─────────────────────────────────────────────────────────────────────────────
# Diffing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_diff(path: str, old_content: str, new_content: str) -> FileDiff:
    """
    Compute a unified diff between ``old_content`` and ``new_content``.

    Returns a ``FileDiff`` with status 'modified' or 'unchanged'.
    """
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff_lines = list(difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{os.path.basename(path)}",
        tofile=f"b/{os.path.basename(path)}",
        n=3,
    ))

    if not diff_lines:
        return FileDiff(path=path, status="unchanged")

    diff_text   = "".join(diff_lines)
    lines_added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
    lines_removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))

    return FileDiff(
        path=path,
        status="modified",
        unified_diff=diff_text,
        lines_added=lines_added,
        lines_removed=lines_removed,
    )


def _read_file_safe(path: str) -> Tuple[bool, str]:
    """
    Safely read a file. Returns (exists, content).
    Returns (False, '') if the file does not exist or cannot be read.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return True, fh.read()
    except (OSError, IOError):
        return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Manager
# ─────────────────────────────────────────────────────────────────────────────

_TRAJECTORY_SUMMARY_WINDOW = 20   # Number of recent steps to include in summary.


class CheckpointManager:
    """
    Stateful manager for incremental Git-style workspace checkpointing.

    Lifecycle
    ---------
    1.  ``track(paths)``  — Called by the runtime at session start or after a
                            context reset. Records the baseline content of all
                            listed file paths.
    2.  ``snapshot(state, signal)`` — Called immediately when a meltdown is
                                       detected. Reads each tracked file from
                                       disk, computes unified diffs, and returns
                                       a ``WorkspaceCheckpoint``.

    The manager stores all produced checkpoints internally so they can be
    retrieved for logging or dashboard display.

    Thread-safety
    -------------
    Not thread-safe by design — Sotis runs as single-threaded middleware.
    """

    def __init__(
        self,
        invariant: Optional[InvariantChecker] = None,
        max_verified: int = 5,
    ) -> None:
        self._baselines  : Dict[str, FileBaseline]       = {}
        self._checkpoints: List[WorkspaceCheckpoint]     = []
        # ── Verified-checkpoint chain (Issue #2) ──
        # Each entry is a {abs_path: FileBaseline} snapshot that PASSED the
        # invariant at commit time. rollback() prefers the most recent of these
        # over the single mutable baseline. Empty/None invariant => disabled,
        # and behavior is identical to the original single-baseline manager.
        self._invariant   : InvariantChecker = invariant or always_valid
        self._has_invariant: bool            = invariant is not None
        self._max_verified : int             = max(1, max_verified)
        self._verified_chain: List[Dict[str, FileBaseline]] = []

    # ── Invariant verification (Issue #2) ──────────────────────────────────────

    def _snapshot_contents(self) -> Dict[str, str]:
        """Read current on-disk content of all tracked files (best-effort)."""
        out: Dict[str, str] = {}
        for abs_path in self._baselines:
            exists, content = _read_file_safe(abs_path)
            out[abs_path] = content if exists else ""
        return out

    def check_invariant(self) -> bool:
        """
        Run the configured invariant against the current workspace state.

        Returns False on any exception (never propagates — hot-path safety).
        Returns True when no invariant is configured.
        """
        if not self._has_invariant:
            return True
        try:
            return bool(self._invariant(self._snapshot_contents()))
        except Exception:
            return False

    def commit_verified(self) -> bool:
        """
        If the current workspace state satisfies the invariant, push it onto the
        verified-checkpoint chain as a known-good rollback target.

        Call this at "quiet" moments — e.g. after a tool result the agent treats
        as success — so a later rollback can target a *proven-good* state rather
        than the most recent (possibly already-corrupted) snapshot.

        Returns True if a verified snapshot was committed, False otherwise.
        No-op (returns False) when no invariant is configured.
        """
        if not self._has_invariant:
            return False
        if not self.check_invariant():
            return False

        snap: Dict[str, FileBaseline] = {}
        for abs_path in self._baselines:
            exists, content = _read_file_safe(abs_path)
            snap[abs_path] = FileBaseline(path=abs_path, content=content, exists=exists)

        self._verified_chain.append(snap)
        if len(self._verified_chain) > self._max_verified:
            self._verified_chain.pop(0)
        return True

    @property
    def verified_count(self) -> int:
        """Number of verified-good snapshots currently retained."""
        return len(self._verified_chain)

    # ── Tracking ──────────────────────────────────────────────────────────────

    def track(self, paths: List[str]) -> None:
        """
        Record the baseline content of the given file paths.

        Paths that do not currently exist on disk are tracked with
        ``exists=False`` so they can be detected as 'added' later.

        Parameters
        ----------
        paths : Absolute or relative paths to the files to track.
        """
        for path in paths:
            abs_path      = str(Path(path).resolve())
            exists, content = _read_file_safe(abs_path)
            self._baselines[abs_path] = FileBaseline(
                path=abs_path,
                content=content,
                exists=exists,
            )

    def untrack(self, path: str) -> None:
        """Remove a file from tracking."""
        abs_path = str(Path(path).resolve())
        self._baselines.pop(abs_path, None)

    def reset_baselines(self) -> None:
        """
        Re-snapshot all currently tracked files as new baselines.

        Called after a context reset so the next checkpoint computes diffs
        from the post-reset workspace state, not the original start state.
        """
        paths = list(self._baselines.keys())
        self._baselines.clear()
        # Verified snapshots belong to the pre-reset execution; drop them so the
        # post-reset run rebuilds its own verified chain from a clean slate.
        self._verified_chain.clear()
        self.track(paths)

    def rollback(self) -> None:
        """
        Restore all tracked files to a known-good state.

        Target selection (Issue #2):
            - If a verified-checkpoint chain exists, restore to the MOST RECENT
              verified snapshot — a state the invariant proved good, not merely
              the last one before the meltdown (which may itself be corrupt).
            - Otherwise fall back to the original single baseline (legacy
              behavior, used when no invariant is configured).

        For each tracked file in the chosen target:
            - If it existed, overwrite the current file with the target content.
            - If it did not exist, delete the current file if present (reverting
              an 'added' file).
        """
        target: Dict[str, FileBaseline] = (
            self._verified_chain[-1] if self._verified_chain else self._baselines
        )
        for abs_path, baseline in target.items():
            if baseline.exists:
                try:
                    Path(abs_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(abs_path, "w", encoding="utf-8") as fh:
                        fh.write(baseline.content)
                except OSError:
                    pass  # Best-effort rollback; log in production
            else:
                # File didn't exist at target — remove it if it was created
                try:
                    if os.path.exists(abs_path):
                        os.remove(abs_path)
                except OSError:
                    pass

    # ── Snapshotting ──────────────────────────────────────────────────────────

    def snapshot(
        self,
        state: ExecutionState,
        signal: MeltdownSignal,
    ) -> WorkspaceCheckpoint:
        """
        Compute a complete workspace checkpoint at meltdown time.

        For every tracked file:
            - If it did not exist at baseline and now exists → 'added'
            - If it existed at baseline and now does not → 'deleted'
            - If it existed at baseline and still does → compute unified diff
            - If it existed at baseline, unchanged → 'unchanged'

        Parameters
        ----------
        state  : Current ExecutionState (provides trajectory, subtask info).
        signal : The MeltdownSignal that triggered the snapshot.

        Returns
        -------
        WorkspaceCheckpoint (also stored internally).
        """
        file_diffs: Dict[str, FileDiff] = {}

        for abs_path, baseline in self._baselines.items():
            exists_now, current_content = _read_file_safe(abs_path)

            if not baseline.exists and exists_now:
                # File was added after baseline.
                file_diffs[abs_path] = FileDiff(
                    path=abs_path,
                    status="added",
                    lines_added=len(current_content.splitlines()),
                )

            elif baseline.exists and not exists_now:
                # File was deleted after baseline.
                file_diffs[abs_path] = FileDiff(
                    path=abs_path,
                    status="deleted",
                    lines_removed=len(baseline.content.splitlines()),
                )

            elif baseline.exists and exists_now:
                file_diffs[abs_path] = _compute_diff(
                    path=abs_path,
                    old_content=baseline.content,
                    new_content=current_content,
                )

            else:
                # File didn't exist before and still doesn't — skip.
                pass

        # Build trajectory summary (last N steps, compact form).
        window      = state.get_window(_TRAJECTORY_SUMMARY_WINDOW)
        traj_summary = [
            {
                "step"     : e.step_index,
                "tool"     : e.tool_name,
                "args_hash": e.args_hash[:8],
                "result"   : e.result_summary or "",
            }
            for e in window
        ]

        # Completed subtask ids.
        completed = [s.subtask_id for s in state.subtasks if s.status == "DONE"]

        # Active subtask description.
        active_desc = ""
        if state.active_subtask:
            active_desc = state.active_subtask.description

        checkpoint = WorkspaceCheckpoint(
            session_id=state.session_id,
            subtask_id=signal.subtask_id,
            snapshot_at_step=signal.triggered_at_step,
            snapshot_at_ms=time.time() * 1000,
            completed_subtasks=completed,
            active_subtask_desc=active_desc,
            trajectory_summary=traj_summary,
            file_diffs=file_diffs,
            reset_number=signal.reset_attempt,
            meltdown_reason=signal.reason.value,
        )

        self._checkpoints.append(checkpoint)
        return checkpoint

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def checkpoint_count(self) -> int:
        return len(self._checkpoints)

    @property
    def latest_checkpoint(self) -> Optional[WorkspaceCheckpoint]:
        return self._checkpoints[-1] if self._checkpoints else None

    @property
    def tracked_paths(self) -> List[str]:
        return list(self._baselines.keys())

    def get_checkpoint(self, index: int) -> Optional[WorkspaceCheckpoint]:
        try:
            return self._checkpoints[index]
        except IndexError:
            return None
