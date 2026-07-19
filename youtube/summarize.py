"""YouTube 总结子模块：在已有中文文稿/字幕基础上写总结。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.summarize import load_summarize_client, summarize_text


def summarize_transcript(
    text: str,
    *,
    title: str = "",
    out_path: Path | None = None,
    client: Any | None = None,
    quiet: bool = False,
    log: Callable[[str], None] | None = None,
) -> str:
    """对中文（或已译）正文做总结，可选写入 .summary.md。"""
    if client is None:
        client = load_summarize_client()
    summary = summarize_text(text, client, title=title, quiet=quiet, log=log)
    if out_path is not None:
        Path(out_path).write_text(summary.rstrip() + "\n", encoding="utf-8")
    return summary


def find_zh_transcript(video_dir: Path, video_id: str) -> Path | None:
    """定位仅中文文稿：优先 {id}.zh.txt，其次任意 *.zh.txt。"""
    video_dir = Path(video_dir)
    candidates = [
        video_dir / f"{video_id}.zh.txt",
        video_dir / f"{video_id}.zh.srt",
    ]
    for c in candidates:
        if c.is_file() and c.stat().st_size > 0:
            return c
    for c in sorted(video_dir.glob("*.zh.txt")):
        return c
    for c in sorted(video_dir.glob("*.zh.srt")):
        return c
    return None
