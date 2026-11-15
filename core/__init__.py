"""OMNIS Core package: schemas, event bus, mathematics, persistence, governance, and orchestration."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from core.schemas import (
    SCHEMA_VERSION,
    CognitiveStateSnapshot,
    OmnisEvent,
    ResourceQuota,
    SystemConfig,
)
from core.events import EventBus
from core.math_utils import (
    shannon_entropy,
    multi_objective_score,
    compute_pareto_front,
    statistical_summary,
    compare_distributions,
)
from core.persistence import OmnisDatabase, PostgresCompatibleCursor, translate_sqlite_to_postgres
from core.governance import (
    GovernanceManager,
    SelfImprovementEngine,
    generate_reproducibility_manifest,
    run_with_timeout,
    verify_api_key,
)
def __getattr__(name: str):
    """Lazily expose OmnisCore to avoid circular imports when subsystems import core submodules."""
    if name == "OmnisCore":
        from core.orchestrator import OmnisCore
        return OmnisCore
    raise AttributeError(f"module 'core' has no attribute {name!r}")

__all__ = [
    "SCHEMA_VERSION",
    "CognitiveStateSnapshot",
    "OmnisEvent",
    "ResourceQuota",
    "SystemConfig",
    "EventBus",
    "shannon_entropy",
    "multi_objective_score",
    "compute_pareto_front",
    "statistical_summary",
    "compare_distributions",
    "OmnisDatabase",
    "PostgresCompatibleCursor",
    "translate_sqlite_to_postgres",
    "GovernanceManager",
    "SelfImprovementEngine",
    "verify_api_key",
    "run_with_timeout",
    "generate_reproducibility_manifest",
    "OmnisCore",
]
