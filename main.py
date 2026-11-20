"""OMNIS 1.0 Main Entrypoint supporting Observatory server, demonstration runner, and benchmark suite."""

from __future__ import annotations

import argparse
import hashlib
import json

import uvicorn

from benchmarks.suite import run_architecture_benchmarks
from core.orchestrator import OmnisCore
from frontier import build_frontier_registry
from observatory.api import create_observatory_app


def get_app():
    return create_observatory_app(OmnisCore(config_path="configs/default_config.json"))


app = get_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="OMNIS 1.0 Cognitive Ecosystem & Observatory")
    parser.add_argument("--mode", choices=["serve", "demo", "benchmark", "frontier"], default="serve")
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
    elif args.mode == "frontier":
        registry = build_frontier_registry()
        seeds = [42, 101, 202, 303, 404]
        results = [entry.runner(seeds) for entry in registry.values()]
        improved = sum(1 for r in results if r["verdict"] == "improved")
        digest_payload = json.dumps(
            [(r["module_id"], r["verdict"], r["receipt_sha256"]) for r in results],
            sort_keys=True,
        ).encode("utf-8")
        print(json.dumps({
            "modules_executed": len(results),
            "improved_count": improved,
            "suite_sha256": hashlib.sha256(digest_payload).hexdigest(),
            "results": [
                {
                    "module_id": r["module_id"],
                    "verdict": r["verdict"],
                    "baseline_mean": r["baseline_mean"],
                    "candidate_mean": r["candidate_mean"],
                    "capacity_matched_control_mean": r["capacity_matched_control_mean"],
                    "receipt_sha256": r["receipt_sha256"],
                }
                for r in results
            ],
        }, indent=2))
    else:
        uvicorn.run("main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
