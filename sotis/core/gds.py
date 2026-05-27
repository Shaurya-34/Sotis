"""
sotis.core.gds
==============
Empirical Graceful Degradation Score (GDS) calculation engine.
"""

from __future__ import annotations

from typing import List
from sotis.core.schemas import Subtask


def calculate_gds(subtasks: List[Subtask], reset_penalty: float = 0.2) -> float:
    """
    Calculates the Graceful Degradation Score (GDS) for the current session.
    
    Formula:
        GDS = Sum(subtask.gds_weight * SuccessMultiplier)
        
        Where SuccessMultiplier = max(0.0, 1.0 - (resets_used * reset_penalty))
        for 'DONE' subtasks, and 0.0 otherwise.
    """
    gds = 0.0
    for s in subtasks:
        if s.status == "DONE":
            mult = max(0.0, 1.0 - (s.resets_used * reset_penalty))
            gds += s.gds_weight * mult
    return min(1.0, max(0.0, gds))


def calculate_max_possible_gds(
    subtasks: List[Subtask], reset_penalty: float = 0.2
) -> float:
    """
    Calculates the maximum possible GDS score remaining in the session.
    
    Subtasks that are FAILED or HARD_FAILED cannot contribute further.
    Subtasks that are PENDING or ACTIVE are assumed to have the potential to
    complete cleanly with 0 resets.
    """
    gds = 0.0
    for s in subtasks:
        if s.status == "DONE":
            mult = max(0.0, 1.0 - (s.resets_used * reset_penalty))
            gds += s.gds_weight * mult
        elif s.status in ("PENDING", "ACTIVE"):
            # Potential to finish cleanly with 0 resets
            gds += s.gds_weight
    return min(1.0, max(0.0, gds))
