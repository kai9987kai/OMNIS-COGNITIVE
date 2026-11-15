"""OMNIS 1.0 Main Entrypoint supporting Observatory server, demonstration runner, and benchmark suite."""

from __future__ import annotations

import argparse
import json

import uvicorn

from benchmarks.suite import run_architecture_benchmarks
from core.orchestrator import OmnisCore
from observatory.api import create_observatory_app


def get_app():
    return create_observatory_app(OmnisCore(config_path="configs/default_config.json"))


app = get_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="OMNIS 1.0 Cognitive Ecosystem & Observatory")
    parser.add_argument("--mode", choices=["serve", "demo", "benchmark"], default="serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.mode == "demo":
        from scripts.run_demo import run_full_demonstration
        summary = run_full_demonstration()
        print(json.dumps(summary, indent=2))
    elif args.mode == "benchmark":
        report = run_architecture_benchmarks()
        print(json.dumps({
            "pareto_optimal": report["pareto_optimal_architectures"],
            "exported_files": report["exported_files"],
        }, indent=2))
    else:
        uvicorn.run("main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
