"""MNEMORPH multi-level adaptive memory system with encoding, retrieval, consolidation, replay, decay, and SQLite persistence."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

import numpy as np

from core.persistence import OmnisDatabase


class MemoryClass(str, Enum):
    SENSORY = "sensory"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    LONG_TERM = "long_term"
    EXPERIMENTAL = "experimental"


def compute_importance(
    reward_assoc: float,
    novelty: float,
    significance: float,
    usefulness: float,
) -> float:
    """Compute memory importance I_i = f(R_i, N_i, S_i, U_i) clamped to [0.0, 1.0]."""
    raw = (
        0.35 * min(1.0, abs(float(reward_assoc)))
        + 0.25 * min(1.0, max(0.0, float(novelty)))
        + 0.20 * min(1.0, max(0.0, float(significance)))
        + 0.20 * min(1.0, max(0.0, float(usefulness)))
    )
    return float(max(0.0, min(1.0, raw)))


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    n = min(len(vec_a), len(vec_b))
    a = np.asarray(vec_a[:n], dtype=float)
    b = np.asarray(vec_b[:n], dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class MemoryRecord:
    """Structured memory record tracked across MNEMORPH levels."""

    memory_id: str
    agent_id: str
    memory_class: str
    timestamp: float
    content: Dict[str, Any]
    embedding: List[float]
    context: Dict[str, Any] = field(default_factory=dict)
    reward_assoc: float = 0.0
    novelty: float = 0.5
    significance: float = 0.5
    usefulness: float = 0.3
    importance: float = 0.4
    confidence: float = 0.8
    retrieval_count: int = 0
    related_memories: List[str] = field(default_factory=list)
    provenance: str = "experience"
    version: int = 1
    strength: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryRecord":
        return cls(**data)


class MnemorphMemory:
    """Multi-level memory system supporting all 7 cognitive memory classes and controlled capacity."""

    DECAY_RATES = {
        MemoryClass.SENSORY.value: 0.35,
        MemoryClass.WORKING.value: 0.18,
        MemoryClass.EPISODIC.value: 0.06,
        MemoryClass.PROCEDURAL.value: 0.03,
        MemoryClass.SEMANTIC.value: 0.02,
        MemoryClass.LONG_TERM.value: 0.01,
        MemoryClass.EXPERIMENTAL.value: 0.04,
    }

    def __init__(self, agent_id: str, max_memories: int = 500, seed: int = 42) -> None:
        self.agent_id = agent_id
        self.max_memories = max(10, int(max_memories))
        self.rng = random.Random(seed)
        self.records: Dict[str, MemoryRecord] = {}
        self.stats = {
            "encoded": 0,
            "retrieved": 0,
            "consolidated": 0,
            "forgotten": 0,
            "replayed": 0,
            "conflicts_detected": 0,
            "last_replay_overhead_ms": 0.0,
        }

    def _estimate_novelty(self, embedding: List[float]) -> float:
        if not self.records or not embedding:
            return 0.9
        recent = list(self.records.values())[-40:]
        max_sim = max((cosine_similarity(embedding, r.embedding) for r in recent), default=0.0)
        return float(max(0.05, min(1.0, 1.0 - max_sim)))

    def encode(
        self,
        memory_class: MemoryClass | str,
        content: Dict[str, Any],
        embedding: List[float],
        timestamp: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        reward_assoc: float = 0.0,
        novelty: Optional[float] = None,
        significance: float = 0.5,
        usefulness: float = 0.3,
        confidence: float = 0.85,
        provenance: str = "experience",
    ) -> MemoryRecord:
        """Encode a new memory record, compute importance, form associations, and enforce capacity."""
        m_class = memory_class.value if isinstance(memory_class, MemoryClass) else str(memory_class)
        ts = float(time.time() if timestamp is None else timestamp)
        nov = self._estimate_novelty(embedding) if novelty is None else float(novelty)
        imp = compute_importance(reward_assoc, nov, significance, usefulness)

        rec = MemoryRecord(
            memory_id=f"mem_{uuid.uuid4().hex[:10]}",
            agent_id=self.agent_id,
            memory_class=m_class,
            timestamp=ts,
            content=content,
            embedding=[float(x) for x in embedding],
            context=context or {},
            reward_assoc=float(reward_assoc),
            novelty=nov,
            significance=float(significance),
            usefulness=float(usefulness),
            importance=imp,
            confidence=float(max(0.0, min(1.0, confidence))),
            provenance=provenance,
        )

        # Link to top similar existing memories
        if self.records and embedding:
            scored: List[Tuple[float, str]] = []
            for existing_id, existing in list(self.records.items())[-60:]:
                sim = cosine_similarity(embedding, existing.embedding)
                if sim >= 0.7:
                    scored.append((sim, existing_id))
            scored.sort(reverse=True)
            for _, target_id in scored[:3]:
                rec.related_memories.append(target_id)
                if rec.memory_id not in self.records[target_id].related_memories:
                    self.records[target_id].related_memories.append(rec.memory_id)

        self.records[rec.memory_id] = rec
        self.stats["encoded"] += 1
        self._enforce_capacity()
        return rec

    def _enforce_capacity(self) -> None:
        while len(self.records) > self.max_memories:
            # Prefer evicting weakest sensory/working/episodic items first
            evict_candidates = sorted(
                self.records.values(),
                key=lambda r: (
                    1 if r.memory_class in (MemoryClass.LONG_TERM.value, MemoryClass.SEMANTIC.value) else 0,
                    r.strength * (0.5 + r.importance),
                    r.timestamp,
                ),
            )
            victim = evict_candidates[0]
            self.delete_memory(victim.memory_id)
            self.stats["forgotten"] += 1

    def retrieve(
        self,
        query_embedding: List[float],
        memory_class: Optional[MemoryClass | str] = None,
        top_k: int = 5,
        current_time: Optional[float] = None,
    ) -> List[MemoryRecord]:
        """Retrieve top_k relevant memories weighted by cosine similarity, importance, strength, and recency."""
        if not self.records:
            return []
        now = float(time.time() if current_time is None else current_time)
        cls_filter = memory_class.value if isinstance(memory_class, MemoryClass) else memory_class

        scored: List[Tuple[float, MemoryRecord]] = []
        for rec in self.records.values():
            if cls_filter is not None and rec.memory_class != cls_filter:
                continue
            sim = cosine_similarity(query_embedding, rec.embedding)
            age = max(0.0, now - rec.timestamp)
            recency = math.exp(-0.01 * age)
            score = 0.55 * max(0.0, sim) + 0.25 * rec.importance + 0.10 * rec.strength + 0.10 * recency
            scored.append((score, rec))

        scored.sort(key=lambda item: item[0], reverse=True)
        results = [item[1] for item in scored[:top_k]]
        for rec in results:
            rec.retrieval_count += 1
            rec.usefulness = min(1.0, rec.usefulness + 0.08)
            rec.strength = min(1.0, rec.strength + 0.05)
            rec.importance = compute_importance(rec.reward_assoc, rec.novelty, rec.significance, rec.usefulness)
        self.stats["retrieved"] += len(results)
        return results

    def consolidate(self, importance_threshold: float = 0.42) -> int:
        """Promote high-importance sensory/working/episodic memories into long-term & semantic memory."""
        promoted = 0
        episodic_pool: List[MemoryRecord] = []

        for rec in list(self.records.values()):
            if rec.memory_class in (MemoryClass.SENSORY.value, MemoryClass.WORKING.value):
                if rec.importance >= importance_threshold or rec.retrieval_count >= 2:
                    rec.memory_class = MemoryClass.EPISODIC.value
                    rec.version += 1
                    promoted += 1
            elif rec.memory_class == MemoryClass.EPISODIC.value:
                episodic_pool.append(rec)
                if rec.importance >= importance_threshold + 0.1 or rec.retrieval_count >= 3:
                    rec.memory_class = MemoryClass.LONG_TERM.value
                    rec.strength = min(1.0, rec.strength + 0.2)
                    rec.version += 1
                    promoted += 1

        # Build a semantic prototype from recent high-value episodic memories
        if len(episodic_pool) >= 4:
            high_val = sorted(episodic_pool, key=lambda r: r.importance, reverse=True)[:4]
            dim = min(len(r.embedding) for r in high_val)
            if dim > 0:
                avg_emb = np.mean([r.embedding[:dim] for r in high_val], axis=0).tolist()
                avg_rew = float(np.mean([r.reward_assoc for r in high_val]))
                sem = self.encode(
                    MemoryClass.SEMANTIC,
                    content={
                        "type": "semantic_prototype",
                        "source_ids": [r.memory_id for r in high_val],
                        "mean_reward": round(avg_rew, 4),
                    },
                    embedding=avg_emb,
                    reward_assoc=avg_rew,
                    novelty=0.4,
                    significance=0.8,
                    usefulness=0.75,
                    provenance="consolidation",
                )
                for src in high_val:
                    self.associate(sem.memory_id, src.memory_id)
                promoted += 1

        self.stats["consolidated"] += promoted
        return promoted

    def decay_and_forget(
        self,
        current_time: Optional[float] = None,
        decay_scale: float = 1.0,
        forget_threshold: float = 0.12,
    ) -> int:
        """Apply class-specific strength decay and explicitly delete records below forget_threshold."""
        forgotten_ids: List[str] = []
        for mid, rec in self.records.items():
            rate = self.DECAY_RATES.get(rec.memory_class, 0.05) * float(decay_scale)
            # Importance slows forgetting
            effective_rate = rate * (1.2 - 0.6 * rec.importance)
            rec.strength = max(0.0, rec.strength * math.exp(-max(0.0, effective_rate)))
            if rec.strength < forget_threshold:
                forgotten_ids.append(mid)

        for mid in forgotten_ids:
            self.delete_memory(mid)
        self.stats["forgotten"] += len(forgotten_ids)
        return len(forgotten_ids)

    def delete_memory(self, memory_id: str) -> bool:
        """Explicitly delete a memory record and clean up cross-references."""
        rec = self.records.pop(memory_id, None)
        if rec is None:
            return False
        for rel_id in rec.related_memories:
            if rel_id in self.records and memory_id in self.records[rel_id].related_memories:
                self.records[rel_id].related_memories.remove(memory_id)
        return True

    def replay(self, batch_size: int = 8) -> Tuple[List[MemoryRecord], float]:
        """Sample prioritized experiences for replay and measure computation overhead in milliseconds."""
        t0 = time.perf_counter()
        eligible = [
            r for r in self.records.values()
            if r.memory_class in (
                MemoryClass.WORKING.value,
                MemoryClass.EPISODIC.value,
                MemoryClass.LONG_TERM.value,
                MemoryClass.PROCEDURAL.value,
            )
        ]
        if not eligible:
            overhead_ms = (time.perf_counter() - t0) * 1000.0
            self.stats["last_replay_overhead_ms"] = overhead_ms
            return [], overhead_ms

        weights = [max(0.05, r.importance + 0.5 * abs(r.reward_assoc)) for r in eligible]
        k = min(batch_size, len(eligible))
        batch = self.rng.choices(eligible, weights=weights, k=k)
        self.stats["replayed"] += len(batch)
        overhead_ms = (time.perf_counter() - t0) * 1000.0
        self.stats["last_replay_overhead_ms"] = overhead_ms
        return batch, overhead_ms

    def associate(self, mem_id_a: str, mem_id_b: str) -> bool:
        """Create a bidirectional association between two existing memories."""
        if mem_id_a == mem_id_b or mem_id_a not in self.records or mem_id_b not in self.records:
            return False
        if mem_id_b not in self.records[mem_id_a].related_memories:
            self.records[mem_id_a].related_memories.append(mem_id_b)
        if mem_id_a not in self.records[mem_id_b].related_memories:
            self.records[mem_id_b].related_memories.append(mem_id_a)
        return True

    def detect_conflicts(self, similarity_threshold: float = 0.85) -> List[Dict[str, Any]]:
        """Identify pairs of highly similar memories with contradictory reward valence."""
        items = list(self.records.values())[-80:]
        conflicts: List[Dict[str, Any]] = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a.reward_assoc * b.reward_assoc >= -0.05:
                    continue
                sim = cosine_similarity(a.embedding, b.embedding)
                if sim >= similarity_threshold:
                    conflicts.append(
                        {
                            "memory_a": a.memory_id,
                            "memory_b": b.memory_id,
                            "similarity": round(sim, 4),
                            "reward_a": round(a.reward_assoc, 4),
                            "reward_b": round(b.reward_assoc, 4),
                        }
                    )
        self.stats["conflicts_detected"] = len(conflicts)
        return conflicts

    def persist_to_db(self, db: OmnisDatabase) -> int:
        """Persist all current memories for this agent to SQLite."""
        count = 0
        for rec in self.records.values():
            db.save_memory(rec.to_dict())
            count += 1
        return count

    def restore_from_db(self, db: OmnisDatabase) -> int:
        """Restore memories for this agent from SQLite."""
        loaded = db.load_memories(agent_id=self.agent_id)
        for row in loaded:
            rec = MemoryRecord.from_dict(row)
            self.records[rec.memory_id] = rec
        return len(loaded)

    def get_class_counts(self) -> Dict[str, int]:
        counts = {mc.value: 0 for mc in MemoryClass}
        for rec in self.records.values():
            counts[rec.memory_class] = counts.get(rec.memory_class, 0) + 1
        return counts

    def get_observatory_snapshot(self) -> Dict[str, Any]:
        """Return structured memory state for the Memory Observatory UI."""
        recent = list(self.records.values())[-40:]
        nodes = [
            {
                "id": r.memory_id,
                "class": r.memory_class,
                "importance": round(r.importance, 3),
                "strength": round(r.strength, 3),
                "reward": round(r.reward_assoc, 3),
                "retrievals": r.retrieval_count,
                "provenance": r.provenance,
            }
            for r in recent
        ]
        edges: List[Dict[str, str]] = []
        recent_ids = {n["id"] for n in nodes}
        seen = set()
        for r in recent:
            for rel in r.related_memories:
                if rel in recent_ids:
                    key = tuple(sorted((r.memory_id, rel)))
                    if key not in seen:
                        seen.add(key)
                        edges.append({"source": r.memory_id, "target": rel})

        return {
            "agent_id": self.agent_id,
            "total_memories": len(self.records),
            "max_memories": self.max_memories,
            "class_counts": self.get_class_counts(),
            "stats": dict(self.stats),
            "conflicts": self.detect_conflicts()[:10],
            "graph": {"nodes": nodes, "edges": edges},
        }

    @staticmethod
    def _tokenize_text(text: str) -> List[str]:
        cleaned = "".join(ch.lower() if (ch.isalnum() or ch == "_") else " " for ch in str(text))
        return [tok for tok in cleaned.split() if tok]

    def _record_tokens(self, rec: MemoryRecord) -> List[str]:
        parts: List[str] = [rec.memory_class, rec.provenance, rec.memory_id]
        for k, v in (rec.content or {}).items():
            parts.append(f"{k} {v}")
        for k, v in (rec.context or {}).items():
            parts.append(f"{k} {v}")
        if rec.reward_assoc > 0.05:
            parts.append("positive reward gain")
        elif rec.reward_assoc < -0.05:
            parts.append("negative hazard penalty")
        return self._tokenize_text(" ".join(parts))

    def hybrid_retrieve(
        self,
        query_embedding: Optional[List[float]] = None,
        query_text: str = "",
        memory_class: Optional[MemoryClass | str] = None,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """NexusSearch-inspired Hybrid Retrieval fusing BM25 lexical ranking and Cosine Vector ranking via Reciprocal Rank Fusion (RRF)."""
        if not self.records:
            return []
        cls_filter = memory_class.value if isinstance(memory_class, MemoryClass) else memory_class
        candidates = [
            r for r in self.records.values()
            if cls_filter is None or r.memory_class == cls_filter
        ]
        if not candidates:
            return []

        q_tokens = self._tokenize_text(query_text)
        doc_tokens: Dict[str, List[str]] = {r.memory_id: self._record_tokens(r) for r in candidates}
        avg_dl = max(1.0, sum(len(toks) for toks in doc_tokens.values()) / float(len(candidates)))
        n_docs = len(candidates)

        df: Dict[str, int] = {}
        for q_tok in set(q_tokens):
            df[q_tok] = sum(1 for toks in doc_tokens.values() if q_tok in toks)

        k1 = 1.5
        b = 0.75
        bm25_scores: Dict[str, float] = {}
        for r in candidates:
            toks = doc_tokens[r.memory_id]
            dl = len(toks)
            score = 0.0
            for q_tok in q_tokens:
                tf = toks.count(q_tok)
                if tf == 0:
                    continue
                doc_freq = df.get(q_tok, 0)
                idf = math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                denom = tf + k1 * (1.0 - b + b * (dl / avg_dl))
                score += idf * ((tf * (k1 + 1.0)) / max(1e-9, denom))
            bm25_scores[r.memory_id] = score

        vec_scores: Dict[str, float] = {}
        for r in candidates:
            sim = cosine_similarity(query_embedding or [], r.embedding) if query_embedding else 0.0
            vec_scores[r.memory_id] = 0.65 * max(0.0, sim) + 0.25 * r.importance + 0.10 * r.strength

        vec_ranked = sorted(candidates, key=lambda r: vec_scores[r.memory_id], reverse=True)
        bm25_ranked = sorted(candidates, key=lambda r: (bm25_scores[r.memory_id], r.importance), reverse=True)
        vec_rank_map = {r.memory_id: rank + 1 for rank, r in enumerate(vec_ranked)}
        bm25_rank_map = {r.memory_id: rank + 1 for rank, r in enumerate(bm25_ranked)}

        k_const = max(1, int(rrf_k))
        fused: List[Dict[str, Any]] = []
        for r in candidates:
            mid = r.memory_id
            rrf = (1.0 / (k_const + vec_rank_map[mid])) + (1.0 / (k_const + bm25_rank_map[mid]))
            if q_tokens and bm25_scores[mid] > 0:
                rrf += 0.005 * min(1.0, bm25_scores[mid])
            fused.append(
                {
                    "memory_id": mid,
                    "memory_class": r.memory_class,
                    "provenance": r.provenance,
                    "rrf_score": round(float(rrf), 6),
                    "bm25_score": round(float(bm25_scores[mid]), 4),
                    "vector_score": round(float(vec_scores[mid]), 4),
                    "importance": round(float(r.importance), 4),
                    "strength": round(float(r.strength), 4),
                    "reward_assoc": round(float(r.reward_assoc), 4),
                    "content": dict(r.content),
                }
            )

        fused.sort(key=lambda item: (item["rrf_score"], item["bm25_score"], item["vector_score"]), reverse=True)
        return fused[: max(1, int(top_k))]

    def run_carrier_archaeology_assay(self, connectome: Any) -> Dict[str, Any]:
        """MEMORY-CARRIER-OBSERVATORY / Mnemorph / MOLT 2^4 = 16 subset factorial carrier ablation assay."""
        active_syns = connectome.active_synapses() if connectome is not None else []
        w_capacity = min(
            1.0,
            sum(abs(s.weight) for s in active_syns) / max(1.0, len(active_syns) * 0.65),
        ) if active_syns else 0.25

        episodic_working = [
            r for r in self.records.values()
            if r.memory_class in (
                MemoryClass.WORKING.value,
                MemoryClass.EPISODIC.value,
                MemoryClass.LONG_TERM.value,
            )
        ]
        trace_capacity = min(
            1.0,
            0.25 + (sum(r.importance * r.strength for r in episodic_working) / max(1.0, len(self.records) * 0.45)),
        ) if self.records else 0.20

        ewc_fisher = sum(getattr(s, "fisher_importance", 0.0) + abs(getattr(s, "consolidated_weight", 0.0)) for s in active_syns)
        sem_proc = sum(
            1 for r in self.records.values()
            if r.memory_class in (MemoryClass.SEMANTIC.value, MemoryClass.PROCEDURAL.value)
        )
        anchor_capacity = min(1.0, 0.20 + 0.04 * ewc_fisher + 0.08 * sem_proc)

        graft_count = sum(1 for s in active_syns if getattr(s, "is_graft", False))
        p_trace = sum(getattr(s, "plasticity_trace", 0.0) for s in active_syns)
        nm_dict = getattr(connectome, "neuromodulators", {}) if connectome is not None else {}
        nm_energy = sum(abs(float(v)) for v in nm_dict.values()) / max(1, len(nm_dict) or 1)
        graft_capacity = min(1.0, 0.22 + 0.10 * graft_count + 0.08 * p_trace + 0.25 * nm_energy)

        carrier_capacities = {
            "C1_synaptic_weights": round(float(w_capacity), 4),
            "C2_episodic_working_traces": round(float(trace_capacity), 4),
            "C3_ewc_consolidated_anchors": round(float(anchor_capacity), 4),
            "C4_neuromodulator_grafts": round(float(graft_capacity), 4),
        }

        weights = (0.36, 0.26, 0.20, 0.18)
        caps = (w_capacity, trace_capacity, anchor_capacity, graft_capacity)
        full_raw = sum(w * c for w, c in zip(weights, caps)) + 0.12 * (w_capacity * trace_capacity) + 0.08 * (anchor_capacity * graft_capacity)
        norm_denom = max(1e-6, full_raw)

        factorial_matrix: List[Dict[str, Any]] = []
        for mask_int in range(16):
            b1 = (mask_int >> 3) & 1
            b2 = (mask_int >> 2) & 1
            b3 = (mask_int >> 1) & 1
            b4 = mask_int & 1
            bits = (b1, b2, b3, b4)
            mask_code = f"{b1}{b2}{b3}{b4}"
            linear = sum(b * w * c for b, w, c in zip(bits, weights, caps))
            synergy = 0.12 * (b1 * b2 * w_capacity * trace_capacity) + 0.08 * (b3 * b4 * anchor_capacity * graft_capacity)
            retention = min(1.0, max(0.0, (linear + synergy) / norm_denom))
            factorial_matrix.append(
                {
                    "mask": mask_code,
                    "active_carriers": sum(bits),
                    "C1_weights": bool(b1),
                    "C2_traces": bool(b2),
                    "C3_anchors": bool(b3),
                    "C4_grafts": bool(b4),
                    "retention_score": round(float(retention), 4),
                    "reconstruction_gap": round(float(1.0 - retention), 4),
                }
            )

        by_mask = {row["mask"]: row["retention_score"] for row in factorial_matrix}
        single_removal_drop = {
            "C1_synaptic_weights": round(1.0 - by_mask["0111"], 4),
            "C2_episodic_working_traces": round(1.0 - by_mask["1011"], 4),
            "C3_ewc_consolidated_anchors": round(1.0 - by_mask["1101"], 4),
            "C4_neuromodulator_grafts": round(1.0 - by_mask["1110"], 4),
        }
        dominant_carrier = max(single_removal_drop.items(), key=lambda kv: kv[1])[0]

        return {
            "agent_id": self.agent_id,
            "assay_protocol": "MEMORY-CARRIER-OBSERVATORY 2^4 Subset Factorial",
            "carrier_capacities": carrier_capacities,
            "single_removal_drop": single_removal_drop,
            "dominant_carrier": dominant_carrier,
            "synergy_index": round(float(1.0 - sum(by_mask[m] for m in ("1000", "0100", "0010", "0001"))), 4),
            "factorial_matrix": factorial_matrix,
        }
