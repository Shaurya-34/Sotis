"""
sotis.bench.tasks
=================
Programmatic benchmark task generators and simulated agent trajectory loops.
"""

from __future__ import annotations

from typing import List, Optional
from sotis.core.schemas import Domain, Subtask, StepEvent, ExecutionState, Domain, SessionStatus, MeltdownReason
from sotis.core.decomposition import TaskDecomposer


def get_bench_subtasks(domain: Domain, horizon: str) -> List[Subtask]:
    """
    Generates a validated list of subtasks for benchmarking depending on domain
    and horizon (short, medium, long, very_long).
    """
    decomposer = TaskDecomposer()
    
    if horizon == "short":
        # Single subtask, budget 20, weight 1.0
        return [
            Subtask(
                subtask_id=f"{domain.value.lower()}-short-1",
                description=f"Quick task for {domain.value} domain",
                domain=domain,
                step_budget=20,
                gds_weight=1.00,
                dependencies=[]
            )
        ]
    elif horizon == "medium":
        # 2 subtasks, budget 40
        return [
            Subtask(
                subtask_id=f"{domain.value.lower()}-med-1",
                description=f"Phase 1 for {domain.value}",
                domain=domain,
                step_budget=20,
                gds_weight=0.40,
                dependencies=[]
            ),
            Subtask(
                subtask_id=f"{domain.value.lower()}-med-2",
                description=f"Phase 2 for {domain.value}",
                domain=domain,
                step_budget=20,
                gds_weight=0.60,
                dependencies=[f"{domain.value.lower()}-med-1"]
            ),
        ]
    elif horizon == "long":
        # 3 subtasks, budget 50
        return [
            Subtask(
                subtask_id=f"{domain.value.lower()}-long-1",
                description=f"Analyze for {domain.value}",
                domain=domain,
                step_budget=15,
                gds_weight=0.20,
                dependencies=[]
            ),
            Subtask(
                subtask_id=f"{domain.value.lower()}-long-2",
                description=f"Process for {domain.value}",
                domain=domain,
                step_budget=20,
                gds_weight=0.40,
                dependencies=[f"{domain.value.lower()}-long-1"]
            ),
            Subtask(
                subtask_id=f"{domain.value.lower()}-long-3",
                description=f"Verify for {domain.value}",
                domain=domain,
                step_budget=15,
                gds_weight=0.40,
                dependencies=[f"{domain.value.lower()}-long-2"]
            ),
        ]
    else:
        # very_long
        # 4 subtasks, budget 60
        return [
            Subtask(
                subtask_id=f"{domain.value.lower()}-vl-1",
                description=f"Initialize for {domain.value}",
                domain=domain,
                step_budget=10,
                gds_weight=0.10,
                dependencies=[]
            ),
            Subtask(
                subtask_id=f"{domain.value.lower()}-vl-2",
                description=f"Execute for {domain.value}",
                domain=domain,
                step_budget=20,
                gds_weight=0.20,
                dependencies=[f"{domain.value.lower()}-vl-1"]
            ),
            Subtask(
                subtask_id=f"{domain.value.lower()}-vl-3",
                description=f"Test for {domain.value}",
                domain=domain,
                step_budget=15,
                gds_weight=0.40,
                dependencies=[f"{domain.value.lower()}-vl-2"]
            ),
            Subtask(
                subtask_id=f"{domain.value.lower()}-vl-4",
                description=f"Deploy for {domain.value}",
                domain=domain,
                step_budget=15,
                gds_weight=0.30,
                dependencies=[f"{domain.value.lower()}-vl-3"]
            ),
        ]
