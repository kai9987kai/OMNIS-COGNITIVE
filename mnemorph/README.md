# mnemorph — Multi-Level Adaptive Memory Architecture

Implements a 7-class hierarchical memory store with importance valuation (`I_i = f(R_i, N_i, S_i, U_i)`), similarity retrieval, consolidation, class-specific decay/forgetting, prioritized replay, and contradiction detection.

## Memory Classes
`sensory`, `working`, `episodic`, `semantic`, `procedural`, `long_term`, and `experimental`.

## Key Features
- Controlled `max_memories` eviction prevents unbounded RAM growth.
- Cosine similarity + importance + recency scoring during retrieval.
- Contradiction/valence conflict detection between similar embeddings with opposing rewards.
- Durable SQLite/PostgreSQL persistence via `persist_to_db` and `restore_from_db`.

## Standalone Execution
```bash
python -m mnemorph.example
```
