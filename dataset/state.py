"""断点状态：SQLite 记录已处理 segment_id。"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Iterable


class SegmentState:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS segments (
                segment_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                error TEXT,
                updated_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_segments_status ON segments(status)"
        )
        self._conn.commit()
        import threading

        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def is_done(self, segment_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "SELECT status FROM segments WHERE segment_id=?", (segment_id,)
            )
            row = cur.fetchone()
            return bool(row and row[0] == "ok")

    def mark(
        self,
        segment_id: str,
        status: str,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO segments(segment_id, status, error, updated_at)
                VALUES(?,?,?,?)
                ON CONFLICT(segment_id) DO UPDATE SET
                    status=excluded.status,
                    error=excluded.error,
                    updated_at=excluded.updated_at
                """,
                (segment_id, status, error, time.time()),
            )
            self._conn.commit()

    def load_done_ids(self) -> set[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT segment_id FROM segments WHERE status='ok'"
            )
            return {r[0] for r in cur.fetchall()}

    def counts(self) -> dict[str, int]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT status, COUNT(*) FROM segments GROUP BY status"
            )
            return {r[0]: r[1] for r in cur.fetchall()}

    def error_ids(self) -> list[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT segment_id FROM segments WHERE status='error'"
            )
            return [r[0] for r in cur.fetchall()]
