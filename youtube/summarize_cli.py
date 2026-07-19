#!/usr/bin/env python3
"""
总结 CLI：对中文 txt/srt 或目录内 zh 文稿生成总结。

用法:
  python -m youtube.summarize_cli path/to/video.zh.txt -o path/to/summary.md
  python -m youtube.summarize_cli path/to/video_id_dir --video-id abc123
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.summarize import load_summarize_client
from transcribe_whisper_nvidia import load_dotenv
from youtube.captions import strip_to_plain_text
from youtube.summarize import find_zh_transcript, summarize_transcript


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="中文文稿/字幕 → 总结（step-3.5-flash 多 Key）")
    p.add_argument("input", type=Path, help="zh.txt / zh.srt 或含文稿的目录")
    p.add_argument("-o", "--output", type=Path, default=None, help="输出 .md，默认旁路 .summary.md")
    p.add_argument("--title", default="", help="标题（写入总结上下文）")
    p.add_argument("--video-id", default="", help="目录模式下指定 video_id")
    p.add_argument("--env-file", type=Path, default=None)
    p.add_argument("-q", "--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = _ROOT
    if args.env_file:
        load_dotenv([args.env_file.expanduser().resolve()])
    else:
        load_dotenv([Path.cwd() / ".env", root / ".env"])

    inp = args.input.expanduser().resolve()
    if inp.is_dir():
        vid = args.video_id or inp.name
        f = find_zh_transcript(inp, vid)
        if not f:
            print(f"错误: 目录中未找到中文 txt/srt: {inp}", file=sys.stderr)
            return 1
        text_path = f
    elif inp.is_file():
        text_path = inp
    else:
        print(f"错误: 输入不存在: {inp}", file=sys.stderr)
        return 1

    raw = text_path.read_text(encoding="utf-8", errors="replace")
    # srt 则去时间轴
    if text_path.suffix.lower() == ".srt" or "-->" in raw:
        text = strip_to_plain_text(raw)
    else:
        text = raw

    out = args.output
    if out is None:
        out = text_path.with_suffix("").with_suffix(".summary.md")
        if str(out).endswith(".zh.summary.md"):
            pass
        elif text_path.name.endswith(".zh.txt"):
            out = text_path.with_name(text_path.name.replace(".zh.txt", ".summary.md"))
        elif text_path.name.endswith(".zh.srt"):
            out = text_path.with_name(text_path.name.replace(".zh.srt", ".summary.md"))
        else:
            out = text_path.with_suffix(".summary.md")

    try:
        client = load_summarize_client()
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(
            f"总结 model={client.config.model} | "
            f"{client.pool.size} keys × {client.config.rate_limit}/min "
            f"≈ {client.pool.effective_rpm()}/min | 输入 {len(text)} 字",
            flush=True,
        )

    summary = summarize_transcript(
        text,
        title=args.title or text_path.stem,
        out_path=out,
        client=client,
        quiet=args.quiet,
    )
    if not args.quiet:
        print(f"已写: {out}")
        print("--- 预览 ---")
        print(summary[:800] + ("…" if len(summary) > 800 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
