#!/usr/bin/env python3
"""
OpenAI 兼容 Chat Completions 翻译模块（支持 NVIDIA 多 Key 负载均衡）。

默认对接 https://integrate.api.nvidia.com/v1 ，每把 nvapi Key 独立 40/min 滑动窗口，
请求在 Key 池内轮询调度（与 Whisper ASR 池同思路）。

环境变量:
  NVIDIA_TRANSLATE_API_KEYS / NVIDIA_TRANSLATE_API_KEYS_FILE  — 翻译专用多 Key
  OPENAI_API_KEY / OPENAI_API_KEYS / OPENAI_API_KEYS_FILE     — 兼容旧配置
  OPENAI_BASE_URL   默认 https://integrate.api.nvidia.com/v1
  OPENAI_MODEL      默认 mistralai/mistral-small-4-119b-2603
  TRANSLATE_TO      默认 zh-CN
  TRANSLATE_RATE_LIMIT         每 Key 限速，默认 40
  TRANSLATE_RATE_WINDOW_SEC    默认 60

用法:
  python translate_openai.py --text "Hello world"
  from translate_openai import OpenAICompatTranslator
  t = OpenAICompatTranslator.from_env()
  print(t.translate("Hello"))
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# NVIDIA OpenAI 兼容网关
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "mistralai/mistral-small-4-119b-2603"
DEFAULT_TARGET = "zh-CN"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_RETRIES = 4
DEFAULT_TIMEOUT = 120
# 每把 NVIDIA Key 独立限速（与 ASR Trial 一致）
DEFAULT_RATE_LIMIT = 40
DEFAULT_RATE_WINDOW_SEC = 60.0


def _die(msg: str, code: int = 1) -> None:
    print(f"错误: {msg}", file=sys.stderr)
    raise SystemExit(code)


def mask_key(key: str) -> str:
    k = key.strip()
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


def load_translate_api_keys(
    *,
    cli_key: str | None = None,
    cli_keys: str | None = None,
    keys_file: Path | None = None,
) -> list[str]:
    """
    收集翻译用 API Key（优先翻译专用，其次通用 OPENAI/NVIDIA）。
    """
    keys: list[str] = []
    if cli_key:
        keys.extend(parse_keys_blob(cli_key))
    if cli_keys:
        keys.extend(parse_keys_blob(cli_keys))
    if keys_file and keys_file.is_file():
        keys.extend(parse_keys_blob(keys_file.read_text(encoding="utf-8")))

    for env_file in (
        os.environ.get("NVIDIA_TRANSLATE_API_KEYS_FILE"),
        os.environ.get("OPENAI_API_KEYS_FILE"),
        os.environ.get("TRANSLATE_API_KEYS_FILE"),
    ):
        if env_file:
            p = Path(env_file).expanduser()
            if p.is_file():
                keys.extend(parse_keys_blob(p.read_text(encoding="utf-8")))

    for env_multi in (
        "NVIDIA_TRANSLATE_API_KEYS",
        "OPENAI_API_KEYS",
        "TRANSLATE_API_KEYS",
        "NVAPI_TRANSLATE_KEYS",
    ):
        keys.extend(parse_keys_blob(os.environ.get(env_multi) or ""))

    for env_single in (
        "NVIDIA_TRANSLATE_API_KEY",
        "OPENAI_API_KEY",
        "LLM_API_KEY",
        "TRANSLATE_API_KEY",
    ):
        v = os.environ.get(env_single) or ""
        if v.strip():
            keys.extend(parse_keys_blob(v))

    # 仅默认加载「翻译专用」文件，不混入 Whisper ASR 的 nvidia_api_keys.txt
    if not keys:
        for base in (Path.cwd(), Path(__file__).resolve().parent):
            p = base / "nvidia_translate_api_keys.txt"
            if p.is_file():
                keys.extend(parse_keys_blob(p.read_text(encoding="utf-8")))
                break

    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


class SlidingWindowRateLimiter:
    """滑动窗口限速：window_sec 内最多 max_calls 次。"""

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

    def acquire(self, quiet: bool = True) -> float:
        waited = 0.0
        if self.max_calls <= 0:
            return 0.0
        while True:
            if self.try_acquire():
                return waited
            sleep_for = max(0.05, self.wait_seconds())
            if not quiet:
                print(
                    f"  翻译限速等待 {sleep_for:.1f}s"
                    f"（每 Key {self.max_calls}/{self.window_sec:g}s）…",
                    flush=True,
                )
            time.sleep(sleep_for)
            waited += sleep_for


class TranslateKeyPool:
    """多 Key 负载均衡：每 Key 独立滑动窗口限速。"""

    def __init__(
        self,
        keys: list[str],
        rate_limit: int = DEFAULT_RATE_LIMIT,
        rate_window_sec: float = DEFAULT_RATE_WINDOW_SEC,
    ) -> None:
        if not keys:
            raise ValueError("翻译 API Key 池为空")
        self.keys = list(keys)
        self.rate_limit = rate_limit
        self.rate_window_sec = rate_window_sec
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
            sleep_for = min(self._limiters[k].wait_seconds() for k in self.keys)
            sleep_for = max(0.05, sleep_for)
            if not quiet:
                print(
                    f"  翻译 Key 池限速等待 {sleep_for:.1f}s"
                    f"（{self.size} keys × {self.rate_limit}/{self.rate_window_sec:g}s"
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
class TranslateConfig:
    api_keys: list[str] = field(default_factory=list)
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    target: str = DEFAULT_TARGET
    temperature: float = DEFAULT_TEMPERATURE
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    rate_limit: int = DEFAULT_RATE_LIMIT
    rate_window_sec: float = DEFAULT_RATE_WINDOW_SEC

    @property
    def api_key(self) -> str:
        """兼容旧代码：返回池中第一把 Key。"""
        return self.api_keys[0] if self.api_keys else ""


class OpenAICompatTranslator:
    """OpenAI 兼容 /v1/chat/completions 翻译客户端（NVIDIA 多 Key 池）。"""

    def __init__(self, config: TranslateConfig) -> None:
        if not config.api_keys:
            raise ValueError(
                "翻译需要 API Key。请设置 NVIDIA_TRANSLATE_API_KEYS_FILE "
                "或 OPENAI_API_KEY / nvidia_translate_api_keys.txt"
            )
        self.config = config
        base = config.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            self.endpoint = base
        else:
            self.endpoint = f"{base}/chat/completions"
        self._pool = TranslateKeyPool(
            config.api_keys,
            rate_limit=config.rate_limit,
            rate_window_sec=config.rate_window_sec,
        )
        self._quiet_rate = True

    @property
    def pool(self) -> TranslateKeyPool:
        return self._pool

    @classmethod
    def from_env(
        cls,
        api_key: str | None = None,
        api_keys: str | None = None,
        keys_file: Path | None = None,
        base_url: str | None = None,
        model: str | None = None,
        target: str | None = None,
        temperature: float | None = None,
        rate_limit: int | None = None,
        rate_window_sec: float | None = None,
    ) -> "OpenAICompatTranslator":
        keys = load_translate_api_keys(
            cli_key=api_key,
            cli_keys=api_keys,
            keys_file=keys_file,
        )
        rl = rate_limit
        if rl is None:
            # 每 Key 默认 40；若显式设了旧的 120 全局逻辑，仍按「每 Key」解释
            rl = int(os.environ.get("TRANSLATE_RATE_LIMIT", DEFAULT_RATE_LIMIT))
        rw = rate_window_sec
        if rw is None:
            rw = float(os.environ.get("TRANSLATE_RATE_WINDOW_SEC", DEFAULT_RATE_WINDOW_SEC))

        # 若 key 全是 nvapi- 且未指定 base，默认走 NVIDIA
        default_base = DEFAULT_BASE_URL
        if keys and all(k.startswith("nvapi-") for k in keys):
            default_base = "https://integrate.api.nvidia.com/v1"
        default_model = DEFAULT_MODEL

        return cls(
            TranslateConfig(
                api_keys=keys,
                base_url=(
                    base_url
                    or os.environ.get("OPENAI_BASE_URL")
                    or os.environ.get("LLM_BASE_URL")
                    or os.environ.get("NVIDIA_TRANSLATE_BASE_URL")
                    or default_base
                ),
                model=(
                    model
                    or os.environ.get("OPENAI_MODEL")
                    or os.environ.get("LLM_MODEL")
                    or os.environ.get("NVIDIA_TRANSLATE_MODEL")
                    or default_model
                ),
                target=target or os.environ.get("TRANSLATE_TO") or DEFAULT_TARGET,
                temperature=(
                    temperature
                    if temperature is not None
                    else float(os.environ.get("TRANSLATE_TEMPERATURE", DEFAULT_TEMPERATURE))
                ),
                rate_limit=rl,
                rate_window_sec=rw,
            )
        )

    def _system_prompt(self) -> str:
        target = self.config.target
        if target.lower() in ("zh", "zh-cn", "zh_cn", "cn", "chinese", "简体中文", "中文"):
            lang_name = "简体中文"
        elif target.lower() in ("zh-tw", "zh_tw", "繁体中文"):
            lang_name = "繁体中文"
        else:
            lang_name = target
        return (
            f"你是专业字幕翻译。将用户给出的文本翻译成{lang_name}。\n"
            "要求：\n"
            "1. 只输出译文，不要解释、不要引号包裹、不要加前缀\n"
            "2. 保留专有名词、产品名、代码标识符（可保留原文或常见译法）\n"
            "3. 语气自然，适合字幕阅读；勿合并或删减原意\n"
            "4. 若输入已是目标语言，可轻微润色后原样返回"
        )

    def translate(self, text: str, *, quiet: bool | None = None) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        body = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": text},
            ],
            # NVIDIA 部分模型需要
            "stream": False,
        }
        data = json.dumps(body).encode("utf-8")
        q = self._quiet_rate if quiet is None else quiet

        last_err: BaseException | None = None
        for attempt in range(1, self.config.max_retries + 1):
            key = self._pool.acquire(quiet=q)
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {key}",
                "User-Agent": "nvidia-nim-whisper/1.0 (+nvidia-translate-pool)",
            }
            req = urllib.request.Request(
                self.endpoint,
                data=data,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                    raw = resp.read().decode("utf-8")
                payload = json.loads(raw)
                content = (
                    payload.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
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
        raise last_err or RuntimeError("翻译失败")

    def translate_segments(
        self,
        segments: list[dict[str, Any]],
        *,
        text_key: str = "text",
        out_key: str = "text_zh",
        workers: int = 4,
        quiet: bool = False,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> list[dict[str, Any]]:
        total = len(segments)
        if total == 0:
            return segments

        self._quiet_rate = quiet
        results: dict[int, str] = {}

        def _one(i: int, seg: dict[str, Any]) -> tuple[int, str]:
            src = (seg.get(text_key) or "").strip()
            if not src:
                return i, ""
            return i, self.translate(src, quiet=quiet)

        if workers <= 1:
            for i, seg in enumerate(segments):
                _, zh = _one(i, seg)
                results[i] = zh
                if on_progress:
                    on_progress(i + 1, total, zh)
                elif not quiet:
                    preview = (zh[:60] + "…") if len(zh) > 60 else zh
                    print(f"  翻译 [{i + 1}/{total}] {preview}", flush=True)
        else:
            if not quiet:
                print(
                    f"  翻译并行 workers={workers}，共 {total} 段，"
                    f"{self._pool.size} keys × {self.config.rate_limit}/"
                    f"{self.config.rate_window_sec:g}s ≈ {self._pool.effective_rpm()}/min …",
                    flush=True,
                )
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_one, i, seg): i for i, seg in enumerate(segments)}
                done = 0
                for fut in as_completed(futs):
                    i, zh = fut.result()
                    results[i] = zh
                    done += 1
                    if on_progress:
                        on_progress(done, total, zh)
                    elif not quiet:
                        preview = (zh[:60] + "…") if len(zh) > 60 else zh
                        print(f"  翻译 [{done}/{total}] #{i} {preview}", flush=True)

        out: list[dict[str, Any]] = []
        for i, seg in enumerate(segments):
            s = dict(seg)
            s[out_key] = results.get(i, "")
            out.append(s)
        return out

    def translate_segments_as_text_list(
        self,
        segments: list[dict[str, Any]],
        *,
        text_key: str = "text",
        out_key: str = "text_zh",
        workers: int = 4,
        quiet: bool = False,
    ) -> tuple[str, list[dict[str, Any]]]:
        segs = self.translate_segments(
            segments,
            text_key=text_key,
            out_key=out_key,
            workers=workers,
            quiet=quiet,
        )
        parts = [(s.get(out_key) or "").strip() for s in segs]
        if self.config.target.lower() in (
            "zh",
            "zh-cn",
            "zh_cn",
            "cn",
            "chinese",
            "简体中文",
            "中文",
        ):
            full = "".join(p for p in parts if p)
        else:
            full = " ".join(p for p in parts if p)
        return full, segs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NVIDIA/OpenAI 兼容接口文本翻译（多 Key 池）")
    p.add_argument("--text", default=None, help="直接翻译的字符串")
    p.add_argument("-i", "--input", type=Path, default=None, help="输入文本文件")
    p.add_argument("-o", "--output", type=Path, default=None, help="输出文件（默认 stdout）")
    p.add_argument("--openai-api-key", default=None, help="单个 Key")
    p.add_argument("--openai-api-keys", default=None, help="多个 Key，逗号分隔")
    p.add_argument("--api-keys-file", type=Path, default=None, help="Key 列表文件")
    p.add_argument("--openai-base-url", default=None, help="覆盖 base URL")
    p.add_argument("--openai-model", default=None, help="覆盖 model")
    p.add_argument("--to", dest="target", default=None, help="目标语言，默认 zh-CN")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument(
        "--rate-limit",
        type=int,
        default=None,
        help=f"每 Key 滑动窗口最大请求数，默认 {DEFAULT_RATE_LIMIT}",
    )
    p.add_argument(
        "--rate-window-sec",
        type=float,
        default=None,
        help=f"限速窗口秒数，默认 {DEFAULT_RATE_WINDOW_SEC:g}",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.text and not args.input:
        _die("请提供 --text 或 -i 输入文件")

    # 加载 .env
    try:
        from transcribe_whisper_nvidia import load_dotenv

        root = Path(__file__).resolve().parent
        load_dotenv([Path.cwd() / ".env", root / ".env"])
    except Exception:
        pass

    try:
        translator = OpenAICompatTranslator.from_env(
            api_key=args.openai_api_key,
            api_keys=args.openai_api_keys,
            keys_file=args.api_keys_file,
            base_url=args.openai_base_url,
            model=args.openai_model,
            target=args.target,
            temperature=args.temperature,
            rate_limit=args.rate_limit,
            rate_window_sec=args.rate_window_sec,
        )
    except ValueError as e:
        _die(str(e))

    if args.text:
        src = args.text
    else:
        src = args.input.read_text(encoding="utf-8")

    print(
        f"翻译 model={translator.config.model} base={translator.config.base_url} "
        f"to={translator.config.target} | keys={translator.pool.size} "
        f"× {translator.config.rate_limit}/{translator.config.rate_window_sec:g}s "
        f"≈ {translator.pool.effective_rpm()}/min",
        file=sys.stderr,
    )
    out = translator.translate(src)
    if args.output:
        args.output.write_text(out + "\n", encoding="utf-8")
        print(f"已写入 {args.output}", file=sys.stderr)
    else:
        print(out)
    if not args.output:
        pass
    print("Key 统计:", translator.pool.stats_summary(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
