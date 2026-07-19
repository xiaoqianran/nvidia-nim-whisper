"""
通用 NVIDIA / OpenAI 兼容 Chat Completions + 多 Key 负载均衡。

翻译、总结等共用此客户端，避免重复造限速轮子。
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_RATE_LIMIT = 40
DEFAULT_RATE_WINDOW_SEC = 60.0
DEFAULT_TIMEOUT = 180
DEFAULT_MAX_RETRIES = 4


def mask_key(key: str) -> str:
    k = (key or "").strip()
    if len(k) <= 16:
        return k[:4] + "…"
    return f"{k[:12]}…{k[-4:]}"


def parse_keys_blob(text: str) -> list[str]:
    if not text:
        return []
    parts: list[str] = []
    for line in text.replace(";", "\n").replace(",", "\n").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts.append(line)
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def load_keys_from_env(
    *,
    file_env: tuple[str, ...] = (),
    multi_env: tuple[str, ...] = (),
    single_env: tuple[str, ...] = (),
    default_files: tuple[str, ...] = (),
    cli_key: str | None = None,
    cli_keys: str | None = None,
    keys_file: Path | None = None,
) -> list[str]:
    keys: list[str] = []
    if cli_key:
        keys.extend(parse_keys_blob(cli_key))
    if cli_keys:
        keys.extend(parse_keys_blob(cli_keys))
    if keys_file and Path(keys_file).is_file():
        keys.extend(parse_keys_blob(Path(keys_file).read_text(encoding="utf-8")))
    for env in file_env:
        p = os.environ.get(env)
        if p and Path(p).expanduser().is_file():
            keys.extend(parse_keys_blob(Path(p).expanduser().read_text(encoding="utf-8")))
    for env in multi_env:
        keys.extend(parse_keys_blob(os.environ.get(env) or ""))
    for env in single_env:
        keys.extend(parse_keys_blob(os.environ.get(env) or ""))
    if not keys:
        for name in default_files:
            for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
                p = base / name
                if p.is_file():
                    keys.extend(parse_keys_blob(p.read_text(encoding="utf-8")))
                    break
            if keys:
                break
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


class SlidingWindowRateLimiter:
    def __init__(self, max_calls: int, window_sec: float = 60.0) -> None:
        self.max_calls = max_calls
        self.window_sec = window_sec
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self._times and now - self._times[0] >= self.window_sec:
            self._times.popleft()

    def remaining(self) -> int:
        if self.max_calls <= 0:
            return 10**9
        with self._lock:
            self._prune(time.monotonic())
            return max(0, self.max_calls - len(self._times))

    def wait_seconds(self) -> float:
        if self.max_calls <= 0:
            return 0.0
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            if len(self._times) < self.max_calls:
                return 0.0
            return max(0.0, self.window_sec - (now - self._times[0]) + 0.02)

    def try_acquire(self) -> bool:
        if self.max_calls <= 0:
            return True
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            if len(self._times) < self.max_calls:
                self._times.append(now)
                return True
            return False


class KeyPool:
    def __init__(
        self,
        keys: list[str],
        rate_limit: int = DEFAULT_RATE_LIMIT,
        rate_window_sec: float = DEFAULT_RATE_WINDOW_SEC,
        label: str = "LLM",
    ) -> None:
        if not keys:
            raise ValueError(f"{label} API Key 池为空")
        self.keys = list(keys)
        self.rate_limit = rate_limit
        self.rate_window_sec = rate_window_sec
        self.label = label
        self._limiters = {
            k: SlidingWindowRateLimiter(rate_limit, rate_window_sec) for k in self.keys
        }
        self._rr = 0
        self._rr_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats: dict[str, int] = {k: 0 for k in self.keys}

    @property
    def size(self) -> int:
        return len(self.keys)

    def effective_rpm(self) -> int:
        if self.rate_limit <= 0:
            return 0
        return self.rate_limit * len(self.keys)

    def acquire(self, quiet: bool = True) -> str:
        if self.rate_limit <= 0:
            with self._rr_lock:
                idx = self._rr % len(self.keys)
                self._rr += 1
            key = self.keys[idx]
            with self._stats_lock:
                self._stats[key] = self._stats.get(key, 0) + 1
            return key
        while True:
            with self._rr_lock:
                start = self._rr
                self._rr += 1
            n = len(self.keys)
            for off in range(n):
                key = self.keys[(start + off) % n]
                if self._limiters[key].try_acquire():
                    with self._stats_lock:
                        self._stats[key] = self._stats.get(key, 0) + 1
                    return key
            sleep_for = max(0.05, min(self._limiters[k].wait_seconds() for k in self.keys))
            if not quiet:
                print(
                    f"  {self.label} Key 池限速等待 {sleep_for:.1f}s"
                    f"（{self.size}×{self.rate_limit}/{self.rate_window_sec:g}s"
                    f" ≈ {self.effective_rpm()}/min）…",
                    flush=True,
                )
            time.sleep(sleep_for)

    def stats_summary(self) -> list[dict[str, Any]]:
        with self._stats_lock:
            return [
                {
                    "key": mask_key(k),
                    "requests": self._stats.get(k, 0),
                    "remaining_slots": self._limiters[k].remaining(),
                }
                for k in self.keys
            ]


@dataclass
class ChatConfig:
    api_keys: list[str] = field(default_factory=list)
    base_url: str = DEFAULT_BASE_URL
    model: str = "stepfun-ai/step-3.5-flash"
    temperature: float = 0.3
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    rate_limit: int = DEFAULT_RATE_LIMIT
    rate_window_sec: float = DEFAULT_RATE_WINDOW_SEC
    label: str = "LLM"


class ChatClient:
    """OpenAI 兼容 chat.completions + Key 池。"""

    def __init__(self, config: ChatConfig) -> None:
        if not config.api_keys:
            raise ValueError(f"{config.label} 需要 API Key")
        self.config = config
        base = config.base_url.rstrip("/")
        self.endpoint = (
            base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        )
        self.pool = KeyPool(
            config.api_keys,
            rate_limit=config.rate_limit,
            rate_window_sec=config.rate_window_sec,
            label=config.label,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        quiet: bool = True,
        max_tokens: int | None = None,
    ) -> str:
        body: dict[str, Any] = {
            "model": model or self.config.model,
            "temperature": self.config.temperature if temperature is None else temperature,
            "messages": messages,
            "stream": False,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        data = json.dumps(body).encode("utf-8")
        last_err: BaseException | None = None
        for attempt in range(1, self.config.max_retries + 1):
            key = self.pool.acquire(quiet=quiet)
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {key}",
                "User-Agent": "nvidia-nim-whisper/1.0 (+llm-chat-pool)",
            }
            req = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                    raw = resp.read().decode("utf-8")
                payload = json.loads(raw)
                content = (
                    payload.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
                if not isinstance(content, str):
                    content = str(content or "")
                return content.strip()
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
                last_err = RuntimeError(
                    f"HTTP {e.code} key={mask_key(key)}: {err_body[:500]}"
                )
                if e.code in (429, 500, 502, 503, 504) and attempt < self.config.max_retries:
                    time.sleep(min(20.0, 1.5 * attempt + 0.5))
                    continue
                raise last_err from e
            except Exception as e:
                last_err = e
                if attempt < self.config.max_retries:
                    time.sleep(min(10.0, 0.8 * attempt))
                    continue
                raise
        raise last_err or RuntimeError("chat 失败")
