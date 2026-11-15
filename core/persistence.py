"""Dual SQLite and PostgreSQL persistence layer for OMNIS memories, organisms, experiments, world snapshots, and audit logs."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple


def translate_sqlite_to_postgres(sql: str) -> str:
    """Translate SQLite parameter placeholders and DDL idioms into PostgreSQL syntax."""
    pg_sql = sql.replace("?", "%s")
    pg_sql = pg_sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
    pg_sql = pg_sql.replace("ORDER BY rowid DESC", "ORDER BY created_at DESC")
    return pg_sql


class PostgresCompatibleCursor:
    """Adapter wrapping a DB-API 2.0 PostgreSQL cursor to accept SQLite-style '?' queries."""

    def __init__(self, raw_conn: Any) -> None:
        self._conn = raw_conn

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Any:
        pg_sql = translate_sqlite_to_postgres(sql)
        cur = self._conn.cursor()
        cur.execute(pg_sql, tuple(params))
        return cur

    def executescript(self, script: str) -> None:
        pg_script = translate_sqlite_to_postgres(script)
        cur = self._conn.cursor()
        for stmt in pg_script.split(";"):
            cleaned = stmt.strip()
            if cleaned:
                cur.execute(cleaned)


class OmnisDatabase:
    """Thread-safe dual SQLite / PostgreSQL database wrapper for OMNIS state and provenance persistence."""

    def __init__(
        self,
        db_path: str = "datasets/omnis_state.db",
        pg_connection: Optional[Any] = None,
    ) -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self.backend = "sqlite"
        self._pg_adapter: Optional[PostgresCompatibleCursor] = None

        if pg_connection is not None or db_path.startswith(("postgresql://", "postgres://")):
            self.backend = "postgresql"
            if pg_connection is not None:
                self._conn = pg_connection
            else:
                try:
                    import psycopg
                except ImportError as exc:
                    raise RuntimeError(
                        "PostgreSQL DSN provided, but 'psycopg' is not installed. "
                        "Install optional dependency via: pip install 'psycopg[binary]>=3.2.0'"
                    ) from exc
                self._conn = psycopg.connect(db_path)
            self._pg_adapter = PostgresCompatibleCursor(self._conn)
        else:
            if db_path != ":memory:":
                parent = os.path.dirname(os.path.abspath(db_path))
                os.makedirs(parent, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row

        self._init_schema()

    def _execute(self, sql: str, params: Tuple[Any, ...] = ()) -> Any:
        if self.backend == "postgresql" and self._pg_adapter is not None:
            return self._pg_adapter.execute(sql, params)
        return self._conn.execute(sql, params)

    def _row_value(self, row: Any, key: str, idx: int) -> Any:
        if isinstance(row, sqlite3.Row) or isinstance(row, dict):
            return row[key]
        return row[idx]

    def _init_schema(self) -> None:
        schema_sql = """
        CREATE TABLE IF NOT EXISTS memories (
            memory_id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            memory_class TEXT NOT NULL,
            timestamp REAL NOT NULL,
            importance REAL NOT NULL,
            confidence REAL NOT NULL,
            retrieval_count INTEGER NOT NULL,
            version INTEGER NOT NULL,
            data_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_id);

        CREATE TABLE IF NOT EXISTS organisms (
            organism_id TEXT PRIMARY KEY,
            generation INTEGER NOT NULL,
            stage TEXT NOT NULL,
            fitness REAL NOT NULL,
            alive INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            checkpoint_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            result_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audit_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            details_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS world_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            tick INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            state_json TEXT NOT NULL
        );
        """
        with self._lock:
            if self.backend == "postgresql" and self._pg_adapter is not None:
                self._pg_adapter.executescript(schema_sql)
            else:
                self._conn.cursor().executescript(schema_sql)
            self._conn.commit()

    def save_memory(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self._execute(
                """
                INSERT INTO memories (
                    memory_id, agent_id, memory_class, timestamp, importance,
                    confidence, retrieval_count, version, data_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    agent_id=excluded.agent_id,
                    memory_class=excluded.memory_class,
                    timestamp=excluded.timestamp,
                    importance=excluded.importance,
                    confidence=excluded.confidence,
                    retrieval_count=excluded.retrieval_count,
                    version=excluded.version,
                    data_json=excluded.data_json
                """,
                (
                    record["memory_id"],
                    record["agent_id"],
                    record["memory_class"],
                    float(record["timestamp"]),
                    float(record["importance"]),
                    float(record["confidence"]),
                    int(record.get("retrieval_count", 0)),
                    int(record.get("version", 1)),
                    json.dumps(record),
                ),
            )
            self._conn.commit()

    def load_memories(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            if agent_id is not None:
                rows = self._execute(
                    "SELECT data_json FROM memories WHERE agent_id = ? ORDER BY timestamp ASC",
                    (agent_id,),
                ).fetchall()
            else:
                rows = self._execute(
                    "SELECT data_json FROM memories ORDER BY timestamp ASC"
                ).fetchall()
            return [json.loads(self._row_value(r, "data_json", 0)) for r in rows]

    def delete_memory(self, memory_id: str) -> bool:
        with self._lock:
            cur = self._execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
            self._conn.commit()
            return getattr(cur, "rowcount", 0) > 0

    def save_organism(self, organism_data: Dict[str, Any], updated_at: str) -> None:
        with self._lock:
            self._execute(
                """
                INSERT INTO organisms (
                    organism_id, generation, stage, fitness, alive, updated_at, checkpoint_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(organism_id) DO UPDATE SET
                    generation=excluded.generation,
                    stage=excluded.stage,
                    fitness=excluded.fitness,
                    alive=excluded.alive,
                    updated_at=excluded.updated_at,
                    checkpoint_json=excluded.checkpoint_json
                """,
                (
                    organism_data["organism_id"],
                    int(organism_data.get("generation", 0)),
                    str(organism_data.get("stage", "genesis")),
                    float(organism_data.get("fitness", 0.0)),
                    1 if organism_data.get("alive", True) else 0,
                    updated_at,
                    json.dumps(organism_data),
                ),
            )
            self._conn.commit()

    def load_organism(self, organism_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._execute(
                "SELECT checkpoint_json FROM organisms WHERE organism_id = ?",
                (organism_id,),
            ).fetchone()
            return json.loads(self._row_value(row, "checkpoint_json", 0)) if row else None

    def list_organisms(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._execute(
                "SELECT checkpoint_json FROM organisms ORDER BY generation ASC, fitness DESC"
            ).fetchall()
            return [json.loads(self._row_value(r, "checkpoint_json", 0)) for r in rows]

    def save_experiment(self, experiment_result: Dict[str, Any]) -> None:
        with self._lock:
            self._execute(
                """
                INSERT INTO experiments (experiment_id, name, status, created_at, result_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(experiment_id) DO UPDATE SET
                    name=excluded.name,
                    status=excluded.status,
                    created_at=excluded.created_at,
                    result_json=excluded.result_json
                """,
                (
                    experiment_result["experiment_id"],
                    experiment_result.get("name", "unnamed"),
                    experiment_result.get("status", "completed"),
                    experiment_result.get("created_at", ""),
                    json.dumps(experiment_result),
                ),
            )
            self._conn.commit()

    def load_experiments(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._execute(
                "SELECT result_json FROM experiments ORDER BY rowid DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            return [json.loads(self._row_value(r, "result_json", 0)) for r in rows]

    def save_world_snapshot(self, snapshot_id: str, tick: int, created_at: str, state: Dict[str, Any]) -> None:
        with self._lock:
            self._execute(
                """
                INSERT INTO world_snapshots (snapshot_id, tick, created_at, state_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    tick=excluded.tick,
                    created_at=excluded.created_at,
                    state_json=excluded.state_json
                """,
                (snapshot_id, int(tick), created_at, json.dumps(state)),
            )
            self._conn.commit()

    def load_world_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._execute(
                "SELECT state_json FROM world_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
            return json.loads(self._row_value(row, "state_json", 0)) if row else None

    def append_audit_log(self, timestamp: str, actor: str, action: str, details: Dict[str, Any]) -> int:
        with self._lock:
            cur = self._execute(
                "INSERT INTO audit_logs (timestamp, actor, action, details_json) VALUES (?, ?, ?, ?)",
                (timestamp, actor, action, json.dumps(details)),
            )
            self._conn.commit()
            return int(getattr(cur, "lastrowid", 0) or 0)

    def get_audit_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._execute(
                "SELECT log_id, timestamp, actor, action, details_json FROM audit_logs ORDER BY log_id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            return [
                {
                    "log_id": self._row_value(r, "log_id", 0),
                    "timestamp": self._row_value(r, "timestamp", 1),
                    "actor": self._row_value(r, "actor", 2),
                    "action": self._row_value(r, "action", 3),
                    "details": json.loads(self._row_value(r, "details_json", 4)),
                }
                for r in rows
            ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
