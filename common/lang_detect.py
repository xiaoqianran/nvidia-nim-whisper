"""轻量语种启发式：中文口播 / 是否需翻译。"""

from __future__ import annotations

import re

_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    # 只统计字母+汉字，忽略数字标点
    cjk = len(_CJK.findall(text))
    lat = len(_LATIN.findall(text))
    total = cjk + lat
    if total == 0:
        return 0.0
    return cjk / total


def looks_chinese(text: str, *, threshold: float = 0.3) -> bool:
    """文本是否 predominantly 中文（标题/简介/口播判断）。"""
    t = (text or "").strip()
    if not t:
        return False
    # 至少若干汉字
    if len(_CJK.findall(t)) < 2:
        return False
    return cjk_ratio(t) >= threshold


def needs_translate_to_zh(text: str) -> bool:
    """已是中文则不必再译。"""
    return not looks_chinese(text, threshold=0.35)


def whisper_language_code(title: str = "", description: str = "", hint: str | None = None) -> str:
    """
    为 Whisper 选 language_code。
    中文口播 → zh-CN；否则 en-US。
    hint 可强制 'zh'/'en'。
    """
    if hint:
        h = hint.lower()
        if h.startswith("zh"):
            return "zh-CN"
        if h.startswith("en"):
            return "en-US"
    sample = f"{title}\n{(description or '')[:500]}"
    if looks_chinese(sample, threshold=0.25):
        return "zh-CN"
    return "en-US"
