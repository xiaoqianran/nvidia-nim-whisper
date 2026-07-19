"""字幕 cue 列表 → 简体中文（复用 translate_openai）。"""

from __future__ import annotations

from typing import Any, Protocol


class CueLike(Protocol):
    start: float
    end: float
    text: str


def translate_cues(
    cues: list[Any],
    translator: Any,
    *,
    workers: int = 4,
    quiet: bool = False,
    cue_factory: Any = None,
) -> list[Any]:
    """
    将 cue 文本译成中文，保留时间轴。
    cue_factory(start, end, text) 用于构造返回对象；默认返回简单命名空间。
    """
    if not cues:
        return []
    if cue_factory is None:
        from types import SimpleNamespace

        def cue_factory(start, end, text):  # type: ignore
            return SimpleNamespace(start=start, end=end, text=text)

    segs = [{"text": c.text, "i": i} for i, c in enumerate(cues)]
    translated = translator.translate_segments(
        segs,
        text_key="text",
        out_key="text_zh",
        workers=max(1, workers),
        quiet=quiet,
    )
    out: list[Any] = []
    for c, t in zip(cues, translated):
        zh = (t.get("text_zh") or "").strip() or c.text
        out.append(cue_factory(c.start, c.end, zh))
    return out


def translate_plain_text(text: str, translator: Any) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return translator.translate(text)
