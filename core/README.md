# core — OMNIS CORE Orchestration & Infrastructure

Central lifecycle, configuration (`SystemConfig`), event communication (`EventBus`), durable storage (`OmnisDatabase`), governance quotas, API key authentication, timeout enforcement, reproducibility manifests, and controlled self-improvement engine.

## Architecture & Key Modules
- `schemas.py`: Versioned Pydantic data models (`CognitiveStateSnapshot`, `OmnisEvent`, `ResourceQuota`, `SystemConfig`).
- `events.py`: Thread-safe pub/sub `EventBus` with wildcard listeners and telemetry counts.
- `persistence.py`: Dual SQLite + PostgreSQL (`psycopg` DSN detection & SQL translation) repository for organisms, memories, experiments, world snapshots, and audit trails.
- `governance.py`: `GovernanceManager`, `SelfImprovementEngine`, `verify_api_key`, `run_with_timeout`, and `generate_reproducibility_manifest`.
- `orchestrator.py`: `OmnisCore` unifying all subsystems.
- `interfaces.py`: Runtime-checkable `Protocol` contracts.

## Standalone Execution
```bash
python -m core.example
```

## Performance & Design Notes
`core/__init__.py` exposes `OmnisCore` via lazy `__getattr__` loading so downstream subsystems can import `core.schemas`, `core.events`, and `core.persistence` without circular dependencies.
