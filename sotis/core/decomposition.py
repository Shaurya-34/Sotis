"""
sotis.core.decomposition
========================
Topological DAG cycle checking and domain-aware task graph parsing.
"""

from __future__ import annotations

from typing import List, Optional
from sotis.core.schemas import Domain, Subtask


def verify_dag(subtasks: List[Subtask]) -> None:
    """
    Asserts that the subtask dependencies form a valid Directed Acyclic Graph (DAG).
    Raises ValueError if duplicate IDs, non-existent dependencies, or circular
    dependencies are found.
    """
    # 1. Check for duplicates in subtask_id
    ids = [s.subtask_id for s in subtasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate subtask_id detected")

    # 2. Build graph and check that all dependencies exist
    id_map = {s.subtask_id: s for s in subtasks}
    adj = {s.subtask_id: [] for s in subtasks}
    for s in subtasks:
        for dep in s.dependencies:
            if dep not in id_map:
                raise ValueError(
                    f"Subtask '{s.subtask_id}' has non-existent dependency '{dep}'"
                )
            adj[dep].append(s.subtask_id)

    # 3. Detect cycle using topological DFS cycle checking
    # states: 0 = unvisited, 1 = visiting (in recursion stack), 2 = visited
    state = {s.subtask_id: 0 for s in subtasks}

    def dfs(u: str) -> None:
        state[u] = 1
        for v in adj[u]:
            if state[v] == 1:
                raise ValueError(
                    f"Circular dependency detected containing '{u}' -> '{v}'"
                )
            elif state[v] == 0:
                dfs(v)
        state[u] = 2

    for s in subtasks:
        if state[s.subtask_id] == 0:
            dfs(s.subtask_id)


class TaskDecomposer:
    """
    Decomposes high-level goals into modular subtasks with budgets and GDS weights.
    """

    def __init__(self) -> None:
        pass

    def get_template(self, domain: Domain) -> List[Subtask]:
        """
        Returns a fresh list of pre-configured Subtask templates for the given domain.
        """
        if domain == Domain.SOFTWARE_ENGINEERING:
            return [
                Subtask(
                    subtask_id="se-clone",
                    description="Clone and initialize repository workspace",
                    domain=domain,
                    step_budget=30,
                    gds_weight=0.10,
                    dependencies=[],
                ),
                Subtask(
                    subtask_id="se-analyze",
                    description="Locate relevant codebase components and parse files",
                    domain=domain,
                    step_budget=50,
                    gds_weight=0.20,
                    dependencies=["se-clone"],
                ),
                Subtask(
                    subtask_id="se-implement",
                    description="Modify source code or implement target solution",
                    domain=domain,
                    step_budget=60,
                    gds_weight=0.40,
                    dependencies=["se-analyze"],
                ),
                Subtask(
                    subtask_id="se-verify",
                    description="Run unit tests and verify correctness of execution",
                    domain=domain,
                    step_budget=40,
                    gds_weight=0.30,
                    dependencies=["se-implement"],
                ),
            ]
        elif domain == Domain.WEB_RESEARCH:
            return [
                Subtask(
                    subtask_id="wr-gather",
                    description="Query web pages and gather diverse primary sources",
                    domain=domain,
                    step_budget=40,
                    gds_weight=0.20,
                    dependencies=[],
                ),
                Subtask(
                    subtask_id="wr-extract",
                    description="Parse text content and extract relevant raw facts",
                    domain=domain,
                    step_budget=50,
                    gds_weight=0.40,
                    dependencies=["wr-gather"],
                ),
                Subtask(
                    subtask_id="wr-synthesize",
                    description="Summarize, cross-verify, and format final research report",
                    domain=domain,
                    step_budget=50,
                    gds_weight=0.40,
                    dependencies=["wr-extract"],
                ),
            ]
        elif domain == Domain.DOCUMENT_PROCESSING:
            return [
                Subtask(
                    subtask_id="dp-parse",
                    description="Ingest and parse PDF/Text document raw metadata",
                    domain=domain,
                    step_budget=30,
                    gds_weight=0.20,
                    dependencies=[],
                ),
                Subtask(
                    subtask_id="dp-extract",
                    description="Extract schema entities and run validation checks",
                    domain=domain,
                    step_budget=50,
                    gds_weight=0.40,
                    dependencies=["dp-parse"],
                ),
                Subtask(
                    subtask_id="dp-report",
                    description="Render output format and run schema validation",
                    domain=domain,
                    step_budget=50,
                    gds_weight=0.40,
                    dependencies=["dp-extract"],
                ),
            ]
        else:
            # Fallback for UNKNOWN domains
            return [
                Subtask(
                    subtask_id="task-main",
                    description="Execute high-level system goal",
                    domain=domain,
                    step_budget=100,
                    gds_weight=1.00,
                    dependencies=[],
                )
            ]

    def decompose(
        self,
        goal: str,
        domain: Domain = Domain.UNKNOWN,
        custom_subtasks: Optional[List[Subtask]] = None,
    ) -> List[Subtask]:
        """
        Decomposes a goal into a validated DAG of Subtask objects.
        """
        if custom_subtasks is not None:
            subtasks = custom_subtasks
        else:
            subtasks = self.get_template(domain)

        # 1. Validate that weights sum to exactly 1.0
        total_weight = sum(s.gds_weight for s in subtasks)
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Sum of subtask GDS weights must be exactly 1.0. Got: {total_weight}"
            )

        # 2. Verify directed acyclic graph structure (raises ValueError on cycles)
        verify_dag(subtasks)

        return subtasks
