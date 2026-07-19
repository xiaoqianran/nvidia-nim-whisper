"""
文本总结（公共）：多 Key 限速 + 超长自动分块 map-reduce。

默认模型 stepfun-ai/step-3.5-flash（NVIDIA integrate）。
上下文：为稳妥按「输入约 24k 字符/块」切分（≈ 保守 6–8k tokens 级），
避免顶满 NIM 上下文或网关 body 限制；最后再综合成一篇总结。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from common.llm_chat import (
    DEFAULT_BASE_URL,
    DEFAULT_RATE_LIMIT,
    DEFAULT_RATE_WINDOW_SEC,
    ChatClient,
    ChatConfig,
    load_keys_from_env,
)

# stepfun-ai/step-3.5-flash：官方为大上下文 MoE，但托管/免费网关常更紧
# 用字符预算做拆分（中文≈1.5–2 字/token，英文≈4 字符/token）取保守值
DEFAULT_MODEL = "stepfun-ai/step-3.5-flash"
DEFAULT_MAX_CHUNK_CHARS = 24000  # 单次送入模型的原文上限
DEFAULT_MAX_PARTIAL_CHARS = 6000  # 分块摘要汇总时的单块摘要长度控制


def load_summarize_client(
    *,
    keys_file: Path | None = None,
    model: str | None = None,
    rate_limit: int | None = None,
    rate_window_sec: float | None = None,
    base_url: str | None = None,
) -> ChatClient:
    keys = load_keys_from_env(
        file_env=("NVIDIA_SUMMARIZE_API_KEYS_FILE", "SUMMARIZE_API_KEYS_FILE"),
        multi_env=("NVIDIA_SUMMARIZE_API_KEYS", "SUMMARIZE_API_KEYS"),
        single_env=("NVIDIA_SUMMARIZE_API_KEY", "SUMMARIZE_API_KEY"),
        default_files=("nvidia_summarize_api_keys.txt",),
        keys_file=keys_file,
    )
    rl = rate_limit
    if rl is None:
        rl = int(os.environ.get("SUMMARIZE_RATE_LIMIT", DEFAULT_RATE_LIMIT))
    rw = rate_window_sec
    if rw is None:
        rw = float(os.environ.get("SUMMARIZE_RATE_WINDOW_SEC", DEFAULT_RATE_WINDOW_SEC))
    return ChatClient(
        ChatConfig(
            api_keys=keys,
            base_url=base_url
            or os.environ.get("SUMMARIZE_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_BASE_URL,
            model=model
            or os.environ.get("SUMMARIZE_MODEL")
            or DEFAULT_MODEL,
            temperature=float(os.environ.get("SUMMARIZE_TEMPERATURE", "0.3")),
            rate_limit=rl,
            rate_window_sec=rw,
            label="总结",
        )
    )


def _split_text(text: str, max_chars: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    # 优先按空行/段落切
    paras = text.replace("\r\n", "\n").split("\n")
    buf: list[str] = []
    n = 0
    for p in paras:
        add = len(p) + 1
        if n + add > max_chars and buf:
            chunks.append("\n".join(buf).strip())
            buf = [p]
            n = len(p)
        else:
            buf.append(p)
            n += add
    if buf:
        chunks.append("\n".join(buf).strip())
    # 仍过长则硬切
    final: list[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
            continue
        for i in range(0, len(c), max_chars):
            final.append(c[i : i + max_chars])
    return [c for c in final if c.strip()]


def _summarize_one(
    client: ChatClient,
    text: str,
    *,
    title: str = "",
    quiet: bool = False,
    partial: bool = False,
) -> str:
    if partial:
        system = (
            "你是中文内容编辑。请用简体中文总结下面这一段字幕/文稿的要点。"
            "保留关键事实、数字、产品名；不要编造；控制在 400–800 字。"
        )
    else:
        system = (
            "你是中文内容主编。请根据提供的材料写一份**完整、结构清晰**的简体中文总结，"
            "适合没看视频的人快速了解内容。要求：\n"
            "1. 用 Markdown：标题 + 若干小节 + 要点列表\n"
            "2. 覆盖主题背景、核心论点、关键步骤/结论\n"
            "3. 保留重要专有名词（模型名、工具名）\n"
            "4. 不要编造材料中没有的信息\n"
            "5. 全文约 800–1500 字（材料很短可更短）"
        )
    user = text
    if title:
        user = f"视频/文稿标题：{title}\n\n正文：\n{text}"
    return client.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        quiet=quiet,
        max_tokens=4096,
    )


def summarize_text(
    text: str,
    client: ChatClient | None = None,
    *,
    title: str = "",
    max_chunk_chars: int | None = None,
    quiet: bool = False,
    log: Callable[[str], None] | None = None,
) -> str:
    """
    对长文本做总结。超长则：分块摘要 → 再综合。
    """

    def _log(msg: str) -> None:
        if log:
            log(msg)
        elif not quiet:
            print(msg, flush=True)

    if client is None:
        client = load_summarize_client()
    max_chars = max_chunk_chars or int(
        os.environ.get("SUMMARIZE_MAX_CHUNK_CHARS", DEFAULT_MAX_CHUNK_CHARS)
    )
    text = (text or "").strip()
    if not text:
        return ""

    chunks = _split_text(text, max_chars)
    _log(f"总结：原文 {len(text)} 字 → {len(chunks)} 块（每块≤{max_chars}）")

    if len(chunks) == 1:
        return _summarize_one(client, chunks[0], title=title, quiet=quiet, partial=False)

    partials: list[str] = []
    for i, ch in enumerate(chunks, 1):
        _log(f"  分块摘要 [{i}/{len(chunks)}] {len(ch)} 字…")
        partials.append(
            _summarize_one(
                client,
                ch,
                title=f"{title}（第{i}/{len(chunks)}部分）" if title else f"第{i}部分",
                quiet=quiet,
                partial=True,
            )
        )

    merged = "\n\n".join(f"### 分段摘要 {i}\n{p}" for i, p in enumerate(partials, 1))
    # 若合并仍过长，再压一轮
    if len(merged) > max_chars:
        _log("分段摘要过长，二次压缩…")
        mid = _split_text(merged, max_chars)
        partials2 = [
            _summarize_one(client, m, title=title, quiet=quiet, partial=True) for m in mid
        ]
        merged = "\n\n".join(partials2)

    _log("综合最终总结…")
    return _summarize_one(client, merged, title=title, quiet=quiet, partial=False)


def summarize_file(
    path: Path | str,
    client: ChatClient | None = None,
    *,
    title: str = "",
    quiet: bool = False,
) -> str:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    return summarize_text(text, client, title=title or p.stem, quiet=quiet)
