"""
sotis.cli
=========
Command-line entry point for Sotis — watches your LLM agent and catches it before it spirals.

Commands
--------
sotis dashboard   — Launch the Streamlit observability dashboard
sotis benchmark   — Run the empirical benchmark suite
sotis demo        — Run the built-in meltdown/recovery demo
"""

from __future__ import annotations

import sys


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "help"

    if command == "dashboard":
        _launch_dashboard()
    elif command == "benchmark":
        _run_benchmark()
    elif command == "demo":
        _run_demo()
    else:
        _print_help()


def _launch_dashboard() -> None:
    try:
        import streamlit.web.cli as stcli
    except ImportError:
        print(
            "[sotis] Streamlit is required for the dashboard.\n"
            "Install it with:  pip install sotis[obs]"
        )
        sys.exit(1)

    import os
    from pathlib import Path

    app_path = str(Path(__file__).parent / "obs" / "app.py")
    sys.argv = ["streamlit", "run", app_path, "--server.headless", "false"]
    sys.exit(stcli.main())


def _run_benchmark() -> None:
    from sotis.bench.runner import BenchmarkRunner
    runner = BenchmarkRunner()
    runner.run_all()


def _run_demo() -> None:
    import importlib.util
    from pathlib import Path

    demo = Path(__file__).parent.parent / "scratch" / "demo_run.py"
    if not demo.exists():
        print("[sotis] Demo script not found. Clone the full repo to use this command.")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("demo", demo)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.run_demo()


def _print_help() -> None:
    print(
        "Sotis — watches your LLM agent and catches it before it spirals.\n"
        "\n"
        "Usage:\n"
        "  sotis dashboard    Launch the Streamlit observability dashboard\n"
        "  sotis benchmark    Run the empirical benchmark suite\n"
        "  sotis demo         Run the built-in meltdown/recovery demo\n"
        "\n"
        "Docs:  https://github.com/Shaurya-34/Sotis\n"
        "PyPI:  https://pypi.org/project/sotis/\n"
    )
