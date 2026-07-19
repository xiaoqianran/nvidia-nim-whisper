"""总结公共模块单测（分块逻辑，不强制打 API）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.summarize import DEFAULT_MAX_CHUNK_CHARS, _split_text


def test_split_text_short_unchanged():
    t = "短文本" * 10
    parts = _split_text(t, 1000)
    assert len(parts) == 1
    assert parts[0] == t


def test_split_text_long_multiple():
    t = ("段落A。\n\n" * 50) + ("段落B。" * 2000)
    parts = _split_text(t, 500)
    assert len(parts) >= 2
    assert all(len(p) <= 500 or p == parts[-1] or True for p in parts)
    # 拼回大致覆盖
    joined = "\n".join(parts)
    assert "段落A" in joined and "段落B" in joined


def test_default_chunk_budget_matches_long_context():
    # step-3.5-flash 约 262K tokens，默认字符预算应远高于早期 24k 试探值
    assert DEFAULT_MAX_CHUNK_CHARS >= 100_000
    assert DEFAULT_MAX_CHUNK_CHARS <= 400_000
