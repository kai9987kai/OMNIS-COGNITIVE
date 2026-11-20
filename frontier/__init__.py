"""OMNIS Frontier Research Pack — 2025-2026 experimental neuroscience, AI, and neuromorphic modules.

Every module plugs into OMNIS through a neutral default-off extension hook and is evaluated
through a capacity-matched control experiment harness with 95% CIs, anytime-valid e-values,
and SHA-256 sealed receipts.
"""

from frontier.contract import (
    CapacityMatchedHarness,
    FrontierExtensionHook,
    FrontierModuleEntry,
    FrontierSpec,
    build_frontier_registry,
)

__all__ = [
    "CapacityMatchedHarness",
    "FrontierExtensionHook",
    "FrontierModuleEntry",
    "FrontierSpec",
    "build_frontier_registry",
]
