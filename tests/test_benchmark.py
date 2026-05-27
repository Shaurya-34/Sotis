"""
tests/test_benchmark
====================
Unit tests for Sotis structured logging and benchmark simulation run outcomes.
"""

from __future__ import annotations

import json
import os
import pytest
from sotis.core.schemas import Domain
from sotis.obs.logger import SessionLogger
from sotis.bench.tasks import get_bench_subtasks
from sotis.bench.runner import simulate_horizon_run, BenchmarkRunner


# ─────────────────────────────────────────────────────────────────────────────
# SessionLogger Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_session_logger_writes_jsonl(tmp_path):
    """Asserts that SessionLogger correctly logs events into a standard JSONL file."""
    log_dir = str(tmp_path)
    logger = SessionLogger(session_id="test-log-session", log_dir=log_dir)
    
    # 1. Log generic event
    logger.log_event("custom_test", {"value": 42})
    
    # 2. Check file exists
    assert os.path.exists(logger.log_path)
    
    # 3. Read and parse line
    with open(logger.log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    assert len(lines) == 1
    record = json.loads(lines[0].strip())
    assert record["event_type"] == "custom_test"
    assert record["data"]["value"] == 42
    assert record["session_id"] == "test-log-session"


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Tasks Generator Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("domain", [Domain.SOFTWARE_ENGINEERING, Domain.WEB_RESEARCH, Domain.DOCUMENT_PROCESSING])
@pytest.mark.parametrize("horizon", ["short", "medium", "long", "very_long"])
def test_get_bench_subtasks_correctness(domain, horizon):
    """Asserts that generated subtasks validate correctly and sum to 1.0 weight."""
    subtasks = get_bench_subtasks(domain, horizon)
    
    # 1. Assert weights sum to 1.0
    total_weight = sum(s.gds_weight for s in subtasks)
    assert total_weight == pytest.approx(1.0)
    
    # 2. Assert step budgets are positive
    for s in subtasks:
        assert s.step_budget > 0
        assert s.domain == domain


# ─────────────────────────────────────────────────────────────────────────────
# Simulation Runs Verification Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_simulate_horizon_run_baseline_fails(tmp_path):
    """Asserts that in baseline mode (without Sotis), the agent fails under induced meltdown."""
    # We use temporary path for logging
    orig_log_dir = "logs"
    try:
        # Override logs dir
        from unittest.mock import patch
        with patch("sotis.obs.logger.os.makedirs") as mock_make, \
             patch("builtins.open") as mock_open:
             
             success, gds, traj = simulate_horizon_run(
                 domain=Domain.SOFTWARE_ENGINEERING,
                 horizon="medium",
                 use_sotis=False,
                 session_id="test-baseline"
             )
             
             # Under baseline (no Sotis), the injected meltdown (infinite loop) causes budget exhaustion.
             # Subtask 2 fails, leading to goal failure (success = False) and decreased GDS.
             assert not success
             # Since subtask 1 (weight 0.4) succeeded and subtask 2 (weight 0.6) failed, GDS should be exactly 0.40.
             assert gds == pytest.approx(0.40)
             assert len(traj) > 0
    finally:
        pass


def test_simulate_horizon_run_sotis_recovers():
    """Asserts that Sotis successfully intercepts meltdown, executes reset, and completes task."""
    from unittest.mock import patch
    with patch("sotis.obs.logger.os.makedirs"), \
         patch("builtins.open"):
         
         success, gds, traj = simulate_horizon_run(
             domain=Domain.SOFTWARE_ENGINEERING,
             horizon="medium",
             use_sotis=True,
             session_id="test-sotis"
         )
         
         # Under Sotis, the meltdown is intercepted, reset triggered, and simulated agent completes goal cleanly.
         assert success
         # Subtask 1 completed cleanly (0.40).
         # Subtask 2 completed with 1 reset (GDS penalty = 0.2 -> multiplier = 0.8 -> GDS contribution = 0.60 * 0.8 = 0.48).
         # Total GDS = 0.40 + 0.48 = 0.88.
         assert gds == pytest.approx(0.88)


# ─────────────────────────────────────────────────────────────────────────────
# BenchmarkRunner Ledger Update Test
# ─────────────────────────────────────────────────────────────────────────────

def test_benchmark_runner_table_formatting():
    """Asserts that table strings are correctly constructed by the runner."""
    runner = BenchmarkRunner(repeats=1)
    
    # Mock results dictionary
    mock_results = {
        "SE": {
            "short": {"baseline": {"success_runs": 0, "gds_sum": 0.0}, "sotis": {"success_runs": 1, "gds_sum": 1.0}},
            "medium": {"baseline": {"success_runs": 0, "gds_sum": 0.4}, "sotis": {"success_runs": 1, "gds_sum": 0.88}},
            "long": {"baseline": {"success_runs": 0, "gds_sum": 0.2}, "sotis": {"success_runs": 1, "gds_sum": 0.90}},
            "very_long": {"baseline": {"success_runs": 0, "gds_sum": 0.1}, "sotis": {"success_runs": 1, "gds_sum": 0.92}}
        }
    }
    
    table_str = runner._build_domain_table(mock_results, "SE")
    assert "Short" in table_str
    assert "Sotis pass@1" in table_str
    assert "100.0%" in table_str
