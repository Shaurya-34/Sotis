"""
tests/test_decomposer
=====================
Unit tests for TaskDecomposer, DAG cycle detection, and GDS scoring engine.
"""

from __future__ import annotations

import pytest
from sotis.core.schemas import Domain, Subtask
from sotis.core.decomposition import TaskDecomposer, verify_dag
from sotis.core.gds import calculate_gds, calculate_max_possible_gds


# ─────────────────────────────────────────────────────────────────────────────
# DAG Validation Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_verify_dag_clean_success():
    """Asserts that a valid, acyclic dependency list passes verification."""
    subtasks = [
        Subtask(subtask_id="a", description="A", dependencies=[]),
        Subtask(subtask_id="b", description="B", dependencies=["a"]),
        Subtask(subtask_id="c", description="C", dependencies=["a", "b"]),
    ]
    # Should not raise any exception
    verify_dag(subtasks)


def test_verify_dag_duplicate_ids():
    """Asserts that duplicate subtask IDs raise a ValueError."""
    subtasks = [
        Subtask(subtask_id="a", description="A", dependencies=[]),
        Subtask(subtask_id="a", description="Another A", dependencies=[]),
    ]
    with pytest.raises(ValueError, match="Duplicate subtask_id detected"):
        verify_dag(subtasks)


def test_verify_dag_nonexistent_dependency():
    """Asserts that dependencies on non-existent subtasks raise a ValueError."""
    subtasks = [
        Subtask(subtask_id="a", description="A", dependencies=["ghost-task"]),
    ]
    with pytest.raises(ValueError, match="has non-existent dependency"):
        verify_dag(subtasks)


def test_verify_dag_direct_cycle():
    """Asserts that a direct cycle (A -> B -> A) raises a ValueError."""
    subtasks = [
        Subtask(subtask_id="a", description="A", dependencies=["b"]),
        Subtask(subtask_id="b", description="B", dependencies=["a"]),
    ]
    with pytest.raises(ValueError, match="Circular dependency detected"):
        verify_dag(subtasks)


def test_verify_dag_self_cycle():
    """Asserts that a self-loop (A -> A) raises a ValueError."""
    subtasks = [
        Subtask(subtask_id="a", description="A", dependencies=["a"]),
    ]
    with pytest.raises(ValueError, match="Circular dependency detected"):
        verify_dag(subtasks)


def test_verify_dag_transitive_cycle():
    """Asserts that a transitive cycle (A -> B -> C -> A) raises a ValueError."""
    subtasks = [
        Subtask(subtask_id="a", description="A", dependencies=["c"]),
        Subtask(subtask_id="b", description="B", dependencies=["a"]),
        Subtask(subtask_id="c", description="C", dependencies=["b"]),
    ]
    with pytest.raises(ValueError, match="Circular dependency detected"):
        verify_dag(subtasks)


# ─────────────────────────────────────────────────────────────────────────────
# TaskDecomposer Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_decomposer_se_template():
    """Asserts that the Software Engineering template maps cleanly and validates."""
    decomposer = TaskDecomposer()
    subtasks = decomposer.decompose(goal="Fix the linter", domain=Domain.SOFTWARE_ENGINEERING)
    
    assert len(subtasks) == 4
    assert subtasks[0].subtask_id == "se-clone"
    assert subtasks[1].subtask_id == "se-analyze"
    assert subtasks[2].subtask_id == "se-implement"
    assert subtasks[3].subtask_id == "se-verify"
    
    # Assert weights sum to exactly 1.0
    assert sum(s.gds_weight for s in subtasks) == pytest.approx(1.0)
    # Check steps
    assert subtasks[2].step_budget == 60


def test_decomposer_wr_template():
    """Asserts that the Web Research template maps cleanly and validates."""
    decomposer = TaskDecomposer()
    subtasks = decomposer.decompose(goal="Find tech trends", domain=Domain.WEB_RESEARCH)
    
    assert len(subtasks) == 3
    assert subtasks[0].subtask_id == "wr-gather"
    assert subtasks[0].gds_weight == 0.20
    assert subtasks[1].gds_weight == 0.40
    assert subtasks[2].gds_weight == 0.40
    assert sum(s.gds_weight for s in subtasks) == pytest.approx(1.0)


def test_decomposer_dp_template():
    """Asserts that the Document Processing template maps cleanly and validates."""
    decomposer = TaskDecomposer()
    subtasks = decomposer.decompose(goal="Process invoice", domain=Domain.DOCUMENT_PROCESSING)
    
    assert len(subtasks) == 3
    assert subtasks[0].subtask_id == "dp-parse"
    assert sum(s.gds_weight for s in subtasks) == pytest.approx(1.0)


def test_decomposer_fallback_unknown():
    """Asserts that an UNKNOWN domain decomposes to a single comprehensive subtask."""
    decomposer = TaskDecomposer()
    subtasks = decomposer.decompose(goal="Do some work", domain=Domain.UNKNOWN)
    
    assert len(subtasks) == 1
    assert subtasks[0].subtask_id == "task-main"
    assert subtasks[0].gds_weight == 1.00


def test_decomposer_custom_validation_error():
    """Asserts that custom subtasks are validated and raise on invalid weight sum."""
    decomposer = TaskDecomposer()
    custom = [
        Subtask(subtask_id="sub-1", description="Part 1", gds_weight=0.50),
        Subtask(subtask_id="sub-2", description="Part 2", gds_weight=0.40),  # Sums to 0.90
    ]
    with pytest.raises(ValueError, match="Sum of subtask GDS weights must be exactly 1.0"):
        decomposer.decompose(goal="Custom", custom_subtasks=custom)


def test_decomposer_custom_success():
    """Asserts that custom subtasks with exact weight sum are accepted and returned."""
    decomposer = TaskDecomposer()
    custom = [
        Subtask(subtask_id="sub-1", description="Part 1", gds_weight=0.50),
        Subtask(subtask_id="sub-2", description="Part 2", gds_weight=0.50, dependencies=["sub-1"]),
    ]
    res = decomposer.decompose(goal="Custom", custom_subtasks=custom)
    assert len(res) == 2
    assert res[0].subtask_id == "sub-1"


# ─────────────────────────────────────────────────────────────────────────────
# GDS Scorer Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_gds_empty_subtasks():
    """Asserts that an empty subtask list scores zero."""
    assert calculate_gds([]) == 0.0
    assert calculate_max_possible_gds([]) == 0.0


def test_gds_perfect_clean_run():
    """Asserts that GDS is exactly 1.0 when all subtasks complete cleanly."""
    subtasks = [
        Subtask(subtask_id="a", description="A", gds_weight=0.30, status="DONE", resets_used=0),
        Subtask(subtask_id="b", description="B", gds_weight=0.70, status="DONE", resets_used=0),
    ]
    assert calculate_gds(subtasks) == pytest.approx(1.0)
    assert calculate_max_possible_gds(subtasks) == pytest.approx(1.0)


def test_gds_with_resets_degradation():
    """Asserts that resets apply GDS penalty factors correctly."""
    subtasks = [
        Subtask(subtask_id="a", description="A", gds_weight=0.50, status="DONE", resets_used=1),
        Subtask(subtask_id="b", description="B", gds_weight=0.50, status="DONE", resets_used=2),
    ]
    # a: 0.5 * (1.0 - 0.2*1) = 0.5 * 0.8 = 0.40
    # b: 0.5 * (1.0 - 0.2*2) = 0.5 * 0.6 = 0.30
    # Total GDS = 0.70
    assert calculate_gds(subtasks) == pytest.approx(0.70)
    assert calculate_max_possible_gds(subtasks) == pytest.approx(0.70)


def test_gds_partial_failure():
    """Asserts that partial failures penalize GDS correctly while keeping completed value."""
    subtasks = [
        Subtask(subtask_id="a", description="A", gds_weight=0.40, status="DONE", resets_used=0),
        Subtask(subtask_id="b", description="B", gds_weight=0.60, status="FAILED", resets_used=0),
    ]
    # Completed 'a' cleanly (0.40), failed 'b' (0.00). Total GDS = 0.40. Max possible GDS = 0.40.
    assert calculate_gds(subtasks) == pytest.approx(0.40)
    assert calculate_max_possible_gds(subtasks) == pytest.approx(0.40)


def test_gds_max_possible_calculation():
    """Asserts that max possible GDS rewards pending/active items but ignores failed ones."""
    subtasks = [
        Subtask(subtask_id="a", description="A", gds_weight=0.30, status="DONE", resets_used=1),   # 0.30 * 0.8 = 0.24
        Subtask(subtask_id="b", description="B", gds_weight=0.30, status="ACTIVE", resets_used=0), # potential: 0.30
        Subtask(subtask_id="c", description="C", gds_weight=0.40, status="FAILED", resets_used=0), # potential: 0.00
    ]
    # Actual GDS: 'a' completed (0.24) + others not complete (0.0) = 0.24
    # Max Possible: 'a' completed (0.24) + 'b' active potential (0.30) + 'c' failed (0.0) = 0.54
    assert calculate_gds(subtasks) == pytest.approx(0.24)
    assert calculate_max_possible_gds(subtasks) == pytest.approx(0.54)


def test_gds_custom_penalty():
    """Asserts that GDS responds correctly to custom reset penalties and bounds below 0.0."""
    subtasks = [
        Subtask(subtask_id="a", description="A", gds_weight=1.00, status="DONE", resets_used=2),
    ]
    # With a heavy penalty of 0.6 per reset, 2 resets exceed 1.0 penalty, so GDS must bound to 0.0.
    assert calculate_gds(subtasks, reset_penalty=0.6) == 0.0
    assert calculate_max_possible_gds(subtasks, reset_penalty=0.6) == 0.0

    # With a small penalty of 0.05 per reset: 1.0 - 0.05*2 = 0.90
    assert calculate_gds(subtasks, reset_penalty=0.05) == pytest.approx(0.90)
