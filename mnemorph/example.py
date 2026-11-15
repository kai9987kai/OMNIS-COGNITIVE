"""Standalone runnable demonstration of MNEMORPH 7-tier memory dynamics."""

from __future__ import annotations

import json

from core.persistence import OmnisDatabase
from mnemorph.interfaces import MemoryHierarchyProtocol
from mnemorph.memory import MemoryClass, MnemorphMemory


def main() -> None:
    mem = MnemorphMemory(agent_id="mnem_demo", max_memories=60, seed=42)
    assert isinstance(mem, MemoryHierarchyProtocol)

    mem.encode(MemoryClass.SENSORY, {"pos": [1, 2]}, [0.8, 0.2, 0.9, 0.1], timestamp=1.0)
    mem.encode(MemoryClass.EPISODIC, {"event": "harvest"}, [0.9, 0.1, 0.85, 0.2], timestamp=2.0, reward_assoc=1.6, significance=0.9)
    mem.encode(MemoryClass.EPISODIC, {"event": "hazard"}, [0.88, 0.12, 0.82, 0.2], timestamp=3.0, reward_assoc=-1.2, significance=0.8)
    mem.encode(MemoryClass.PROCEDURAL, {"best_action": 4}, [0.9, 0.1, 0.85, 0.2], timestamp=4.0, reward_assoc=1.6, significance=0.95)

    promoted = mem.consolidate()
    replayed, overhead_ms = mem.replay(batch_size=4)
    conflicts = mem.detect_conflicts(similarity_threshold=0.85)

    db = OmnisDatabase(":memory:")
    saved = mem.persist_to_db(db)
    db.close()

    print("=== MNEMORPH MEMORY HIERARCHY EXAMPLE ===")
    print(
        json.dumps(
            {
                "class_counts": mem.get_class_counts(),
                "promoted_to_long_term": promoted,
                "replayed_records": len(replayed),
                "replay_overhead_ms": round(overhead_ms, 4),
                "conflicts_detected": len(conflicts),
                "db_persisted_records": saved,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
