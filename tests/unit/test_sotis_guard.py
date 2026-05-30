"""
tests/test_sotis_guard.py
=========================
Unit tests for the primary developer-facing SotisGuard facade.
"""

from sotis import SotisGuard, StepEvent, EntropyConfig

def test_sotis_guard_basic_watch():
    guard = SotisGuard(entropy_config=EntropyConfig(hard_threshold=3.0))
    
    # Normal non-looping calls should return False
    assert not guard.watch("read_file", {"path": "a.py"})
    assert not guard.watch("write_file", {"path": "a.py"})
    assert not guard.watch("run_tests", {"cmd": "pytest"})


def test_sotis_guard_loop_detection():
    guard = SotisGuard()
    
    # Repeat the identical tool call 3 times (with LoopConfig defaults)
    assert not guard.watch("read_file", {"path": "a.py"})
    assert not guard.watch("read_file", {"path": "a.py"})
    # The 3rd repetition triggers loop meltdown detection
    assert guard.watch("read_file", {"path": "a.py"})

def test_sotis_guard_density_detection():
    guard = SotisGuard()
    
    # 3 consecutive edits to a file without test results changing triggers density guard
    assert not guard.watch("write_file", {"file_path": "app.py"})
    assert not guard.watch("write_file", {"file_path": "app.py"})
    assert guard.watch("write_file", {"file_path": "app.py"})

def test_sotis_guard_reset():
    guard = SotisGuard()
    
    assert not guard.watch("read_file", {"path": "a.py"})
    assert not guard.watch("read_file", {"path": "a.py"})
    
    guard.reset()
    
    # After reset, the counter starts fresh
    assert not guard.watch("read_file", {"path": "a.py"})
