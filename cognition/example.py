"""Standalone runnable demonstration of the 10-stage Cognitive Pipeline."""

from __future__ import annotations

import json

from cognition.interfaces import CognitivePipelineProtocol
from cognition.pipeline import COGNITIVE_STAGES, CognitivePipeline
from diamond.environment import DiamondWorld
from diamond.rl import DynaWorldModel, QAgentLearner
from mnemorph.memory import MnemorphMemory
from sonapse.connectome import SonapseConnectome


def main() -> None:
    pipeline = CognitivePipeline()
    assert isinstance(pipeline, CognitivePipelineProtocol)

    world = DiamondWorld(width=10, height=10, seed=7)
    conn = SonapseConnectome(seed=7)
    mem = MnemorphMemory(agent_id="cog_demo", seed=7)
    learner = QAgentLearner(seed=7)
    wm = DynaWorldModel(seed=7)
    world.register_agent("cog_demo", energy=100.0)

    out = pipeline.step("cog_demo", 1, world, conn, mem, learner, wm)
    snap = out["snapshot"]
    print("=== COGNITION 10-STAGE PIPELINE EXAMPLE ===")
    print(
        json.dumps(
            {
                "stages_count": len(COGNITIVE_STAGES),
                "action": out["action"],
                "reward": round(out["reward"], 4),
                "td_error": round(out["td_error"], 4),
                "memory_records": snap.M_t["total_records"],
                "active_synapses": snap.G_t["active_synapses"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
