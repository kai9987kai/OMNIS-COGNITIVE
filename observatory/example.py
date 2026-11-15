"""Standalone runnable demonstration booting Observatory FastAPI with TestClient (non-blocking)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from core.orchestrator import OmnisCore
from core.schemas import SystemConfig
from observatory.api import create_observatory_app
from observatory.interfaces import ObservatoryFactoryProtocol


def main() -> None:
    factory: ObservatoryFactoryProtocol = create_observatory_app
    assert isinstance(factory, ObservatoryFactoryProtocol)

    omnis = OmnisCore(config=SystemConfig(db_path=":memory:", initial_population=2, seed=42))
    app = factory(core=omnis)
    client = TestClient(app)

    health = client.get("/api/health")
    centre = client.get("/api/command-centre")
    q_opt = client.post("/api/hardware/quantum-optimize", json={"iterations": 8})

    print("=== OBSERVATORY FASTAPI EXAMPLE ===")
    print(
        json.dumps(
            {
                "health_status": health.status_code,
                "command_centre_status": centre.status_code,
                "quantum_opt_status": q_opt.status_code,
                "tick": centre.json().get("tick"),
                "registered_routes": len(app.routes),
            },
            indent=2,
        )
    )
    omnis.shutdown()


if __name__ == "__main__":
    main()
