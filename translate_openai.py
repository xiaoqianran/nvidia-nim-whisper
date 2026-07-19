#!/usr/bin/env python3
"""
OpenAI 兼容 Chat Completions 翻译模块。

用于把 Whisper 转写结果译成目标语言（默认简体中文）。
Whisper 自带 task=translate 只能到英文，故用独立 LLM 做任意目标语。

环境变量（或 CLI / 调用方传入）:
  OPENAI_API_KEY      API Key
  OPENAI_BASE_URL     如 https://api.openai.com/v1 或 https://integrate.api.nvidia.com/v1
  OPENAI_MODEL        如 gpt-4o-mini / meta/llama-3.1-8b-instruct
  TRANSLATE_TO        目标语言，默认 zh-CN

用法（独立）:
  python translate_openai.py --text "Hello world"
  python translate_openai.py -i transcript.txt -o transcript.zh.txt

用法（库）:
  from translate_openai import OpenAICompatTranslator
  t = OpenAICompatTranslator.from_env()
  print(t.translate("Hello"))
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TARGET = "zh-CN"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_RETRIES = 4
DEFAULT_TIMEOUT = 120


def _die(msg: str, code: int = 1) -> None:
    print(f"错误: {msg}", file=sys.stderr)
    raise SystemExit(code)


@dataclass
class TranslateConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    target: str = DEFAULT_TARGET
    temperature: float = DEFAULT_TEMPERATURE
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES


class OpenAICompatTranslator:
    """OpenAI 兼容 /v1/chat/completions 翻译客户端。"""

    def __init__(self, config: TranslateConfig) -> None:
        if not config.api_key:
            raise ValueError("翻译需要 API Key（OPENAI_API_KEY 或 --openai-api-key）")
        self.config = config
        base = config.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            self.endpoint = base
        else:
            self.endpoint = f"{base}/chat/completions"

    @classmethod
    def from_env(
        cls,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        target: str | None = None,
        temperature: float | None = None,
    ) -> "OpenAICompatTranslator":
        key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("TRANSLATE_API_KEY")
            or ""
        )
        return cls(
            TranslateConfig(
                api_key=key,
                base_url=(
                    base_url
                    or os.environ.get("OPENAI_BASE_URL")
                    or os.environ.get("LLM_BASE_URL")
                    or DEFAULT_BASE_URL
                ),
                model=model or os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL") or DEFAULT_MODEL,
                target=target or os.environ.get("TRANSLATE_TO") or DEFAULT_TARGET,
                temperature=(
                    temperature
                    if temperature is not None
                    else float(os.environ.get("TRANSLATE_TEMPERATURE", DEFAULT_TEMPERATURE))
                ),
            )
        )

    def _system_prompt(self) -> str:
        target = self.config.target
        # 常见别名
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

    def translate(self, text: str) -> str:
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
        }
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        last_err: BaseException | None = None
        for attempt in range(1, self.config.max_retries + 1):
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
                last_err = RuntimeError(f"HTTP {e.code}: {err_body[:500]}")
                # 429/5xx 重试
                if e.code in (429, 500, 502, 503, 504) and attempt < self.config.max_retries:
                    time.sleep(min(20.0, 1.2 * attempt))
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
        """
        按片段翻译，写入 out_key，保留时间轴字段。
        workers=1 串行；>1 并行（按 index 回填）。
        """
        total = len(segments)
        if total == 0:
            return segments

        results: dict[int, str] = {}

        def _one(i: int, seg: dict[str, Any]) -> tuple[int, str]:
            src = (seg.get(text_key) or "").strip()
            if not src:
                return i, ""
            return i, self.translate(src)

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
                print(f"  翻译并行 workers={workers}，共 {total} 段 …", flush=True)
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
        """翻译片段并拼出全文。"""
        segs = self.translate_segments(
            segments, text_key=text_key, out_key=out_key, workers=workers, quiet=quiet
        )
        parts = [(s.get(out_key) or "").strip() for s in segs]
        full = " ".join(p for p in parts if p)
        # 中文用空串拼接更自然，但保留空格对中英混排也行；对纯中文去多余空格
        if self.config.target.lower() in ("zh", "zh-cn", "zh_cn", "cn", "chinese", "简体中文", "中文"):
            full = "".join(p for p in parts if p)
        return full, segs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI 兼容接口文本翻译")
    p.add_argument("--text", default=None, help="直接翻译的字符串")
    p.add_argument("-i", "--input", type=Path, default=None, help="输入文本文件")
    p.add_argument("-o", "--output", type=Path, default=None, help="输出文件（默认 stdout）")
    p.add_argument("--openai-api-key", default=None, help="覆盖 OPENAI_API_KEY")
    p.add_argument("--openai-base-url", default=None, help="覆盖 OPENAI_BASE_URL")
    p.add_argument("--openai-model", default=None, help="覆盖 OPENAI_MODEL")
    p.add_argument("--to", dest="target", default=None, help="目标语言，默认 zh-CN")
    p.add_argument("--temperature", type=float, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.text and not args.input:
        _die("请提供 --text 或 -i 输入文件")

    try:
        translator = OpenAICompatTranslator.from_env(
            api_key=args.openai_api_key,
            base_url=args.openai_base_url,
            model=args.openai_model,
            target=args.target,
            temperature=args.temperature,
        )
    except ValueError as e:
        _die(str(e))

    if args.text:
        src = args.text
    else:
        src = args.input.read_text(encoding="utf-8")

    print(
        f"翻译 model={translator.config.model} base={translator.config.base_url} "
        f"to={translator.config.target}",
        file=sys.stderr,
    )
    out = translator.translate(src)
    if args.output:
        args.output.write_text(out + "\n", encoding="utf-8")
        print(f"已写入 {args.output}", file=sys.stderr)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
