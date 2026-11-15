"""Standalone runnable demonstration of OMNIS CORE (Events, Persistence, Governance, Orchestration)."""

from __future__ import annotations

import json

from core.governance import generate_reproducibility_manifest, run_with_timeout, verify_api_key
from core.interfaces import EventBusProtocol, OrchestratorProtocol, PersistenceBackendProtocol
from core.orchestrator import OmnisCore
from core.schemas import SystemConfig


def main() -> None:
    cfg = SystemConfig(db_path=":memory:", initial_population=3, seed=42, require_api_auth=False)
    omnis = OmnisCore(config=cfg)

    assert isinstance(omnis.event_bus, EventBusProtocol)
    assert isinstance(omnis.db, PersistenceBackendProtocol)
    assert isinstance(omnis, OrchestratorProtocol)

    step_summary = run_with_timeout(lambda: omnis.step(steps=4), timeout_sec=10.0)
    manifest = generate_reproducibility_manifest(cfg, extra={"example": "core"})
    auth_ok = verify_api_key(cfg, None)
    status = omnis.get_command_centre_status()

    print("=== OMNIS CORE EXAMPLE ===")
    print(
        json.dumps(
            {
                "tick": omnis.tick,
                "step_latency_ms": status["step_latency_ms"],
                "auth_open_mode": auth_ok,
                "manifest_sha256": manifest["config_sha256"][:16],
                "command_centre_organisms": status["active_organisms"],
                "alive_after_step": step_summary["alive_count"],
            },
            indent=2,
        )
    )
    omnis.shutdown()


if __name__ == "__main__":
    main()
