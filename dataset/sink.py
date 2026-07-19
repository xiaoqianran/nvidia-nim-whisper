"""增量 JSONL 写出。"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class JsonlSink:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self.written = 0

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._fp.write(line + "\n")
            self._fp.flush()
            self.written += 1

    def close(self) -> None:
        with self._lock:
            self._fp.close()
