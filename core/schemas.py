"""Typed data structures and versioned message schemas for OMNIS 1.0."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


class OmnisEvent(BaseModel):
    """Versioned event structure passed across the OMNIS EventBus."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    source: str
    timestamp: str = Field(default_factory=utc_now_iso)
    tick: int = 0
    payload: Dict[str, Any] = Field(default_factory=dict)


class CognitiveStateSnapshot(BaseModel):
    """Unified internal cognitive state C_t = {P_t, A_t, M_t, G_t, W_t, H_t}."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    organism_id: str
    tick: int
    P_t: Dict[str, Any] = Field(default_factory=dict, description="Perceptual state")
    A_t: Dict[str, Any] = Field(default_factory=dict, description="Attention state")
    M_t: Dict[str, Any] = Field(default_factory=dict, description="Memory state")
    G_t: Dict[str, Any] = Field(default_factory=dict, description="Connectome state")
    W_t: Dict[str, Any] = Field(default_factory=dict, description="World model state")
    H_t: Dict[str, Any] = Field(default_factory=dict, description="Historical context")
    stage_metrics: Dict[str, Any] = Field(default_factory=dict, description="10-stage pipeline telemetry")


class ResourceQuota(BaseModel):
    """Hard resource limits enforced by GovernanceManager."""

    max_neurons_per_organism: int = Field(default=128, ge=4, le=4096)
    max_synapses_per_organism: int = Field(default=512, ge=4, le=65536)
    max_memories_per_organism: int = Field(default=500, ge=10, le=50000)
    max_population_size: int = Field(default=24, ge=1, le=500)
    max_experiment_timeout_sec: float = Field(default=30.0, gt=0.1, le=600.0)
    max_connectome_density: float = Field(default=0.35, gt=0.01, le=1.0)


class SystemConfig(BaseModel):
    """Versioned global configuration for the OMNIS platform."""

    config_version: str = Field(default=SCHEMA_VERSION)
    seed: int = Field(default=42)
    db_path: str = Field(default="datasets/omnis_state.db")
    world_width: int = Field(default=16, ge=6, le=64)
    world_height: int = Field(default=16, ge=6, le=64)
    continuous_world: bool = Field(default=False)
    initial_population: int = Field(default=6, ge=1, le=64)
    default_neuron_model: str = Field(default="rate_coded")
    learning_rate: float = Field(default=0.08, gt=0.0, le=1.0)
    plasticity_mode: str = Field(default="three_factor")
    regularization_lambda: float = Field(default=0.002, ge=0.0, le=0.5)
    discount_gamma: float = Field(default=0.92, ge=0.0, le=0.999)
    mutation_rate: float = Field(default=0.15, ge=0.0, le=1.0)
    crossover_rate: float = Field(default=0.65, ge=0.0, le=1.0)
    elite_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    hardware_backend: str = Field(default="cpu_vector")
    quotas: ResourceQuota = Field(default_factory=ResourceQuota)
    fitness_weights: Dict[str, float] = Field(
        default_factory=lambda: {"alpha": 1.0, "beta": 0.4, "gamma": 0.3, "lambda": 0.1}
    )
    allow_external_apis: bool = Field(default=False)
    require_api_auth: bool = Field(default=False)
    api_key: str = Field(default="omnis-research-key")
