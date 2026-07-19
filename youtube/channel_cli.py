#!/usr/bin/env python3
"""
YouTube 频道最近 N 视频：元数据 + 字幕/Whisper → 简体。

用法:
  python -m youtube.channel_cli "https://www.youtube.com/@AIsuperdomain" \\
    --limit 5 --cookies /root/Desktop/www.youtube.com_cookies.txt \\
    -o ./out/channel_AIsuperdomain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from transcribe_whisper_nvidia import (
    DEFAULT_FUNCTION_ID,
    DEFAULT_RATE_LIMIT,
    DEFAULT_RATE_WINDOW_SEC,
    DEFAULT_SERVER,
    DEFAULT_WORKERS,
    NvidiaApiKeyPool,
    import_riva,
    load_dotenv,
    load_nvidia_api_keys,
)
from youtube.channel import process_channel


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YouTube 频道：最近 N 视频字幕/Whisper→简体")
    p.add_argument("channel", help="频道 URL，如 https://www.youtube.com/@AIsuperdomain")
    p.add_argument("--limit", type=int, default=5, help="最近视频数量")
    p.add_argument("-o", "--out-dir", type=Path, default=Path("./out/channel"))
    p.add_argument("--cookies", type=Path, default=None)
    p.add_argument("--no-audio-fallback", action="store_true", help="无字幕时不 Whisper")
    p.add_argument("--no-translate", action="store_true")
    p.add_argument("--translate-workers", type=int, default=4)
    p.add_argument("--whisper-workers", type=int, default=None)
    p.add_argument("--resume", action="store_true", default=True, help="跳过已完成 video_id（默认开）")
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument(
        "--zh-only",
        action="store_true",
        help="仅保留中文文稿/字幕（给总结用，不写英文 en.txt/srt）",
    )
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

    cookies = args.cookies.expanduser() if args.cookies else None
    if cookies and not cookies.is_file():
        print(f"错误: cookies 不存在: {cookies}", file=sys.stderr)
        return 2

    translator = None
    if not args.no_translate:
        try:
            from translate_openai import OpenAICompatTranslator

            translator = OpenAICompatTranslator.from_env()
            if not args.quiet:
                print(
                    f"翻译: {translator.config.model} | "
                    f"{translator.pool.size} keys ≈ {translator.pool.effective_rpm()}/min",
                    flush=True,
                )
        except Exception as e:
            print(f"警告: 翻译器未就绪: {e}", file=sys.stderr)

    riva_client = None
    asr_pool = None
    if not args.no_audio_fallback:
        keys = load_nvidia_api_keys(cli_key=None, cli_keys=None, keys_file=None)
        if keys:
            riva_client = import_riva()
            asr_pool = NvidiaApiKeyPool(
                keys,
                riva_client=riva_client,
                server=DEFAULT_SERVER,
                function_id=DEFAULT_FUNCTION_ID,
                max_mb=128,
                rate_limit=int(
                    __import__("os").environ.get("WHISPER_RATE_LIMIT") or DEFAULT_RATE_LIMIT
                ),
                rate_window_sec=float(
                    __import__("os").environ.get("WHISPER_RATE_WINDOW_SEC")
                    or DEFAULT_RATE_WINDOW_SEC
                ),
            )
            if not args.quiet:
                print(
                    f"Whisper: {asr_pool.size} keys ≈ {asr_pool.effective_rpm()}/min",
                    flush=True,
                )
        else:
            print("警告: 无 NVIDIA ASR Key，无字幕时无法 Whisper 回退", file=sys.stderr)

    ww = args.whisper_workers
    if ww is None:
        ww = min(32, max(DEFAULT_WORKERS, (asr_pool.size if asr_pool else 1) * 4))

    try:
        results = process_channel(
            args.channel,
            args.out_dir.expanduser().resolve(),
            limit=args.limit,
            cookies=cookies,
            translator=translator,
            riva_client=riva_client,
            asr_pool=asr_pool,
            translate_workers=args.translate_workers,
            whisper_workers=ww,
            fallback_audio=not args.no_audio_fallback,
            resume=args.resume,
            quiet=args.quiet,
            output_profile="zh_only" if args.zh_only else "full",
        )
    except Exception as e:
        print(f"错误: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    ok = sum(1 for r in results if r.outputs.get("zh_txt") or r.outputs.get("en_txt"))
    print(f"完成 {ok}/{len(results)} 有转写产物 → {args.out_dir}")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
