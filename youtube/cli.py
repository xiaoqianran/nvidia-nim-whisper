#!/usr/bin/env python3
"""
YouTube 字幕 CLI：优先简体；否则英文 + 译成简体。

用法:
  python -m youtube.cli "https://www.youtube.com/watch?v=VIDEO_ID" \\
    --cookies /root/Desktop/www.youtube.com_cookies.txt \\
    -o ./out/youtube
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from transcribe_whisper_nvidia import load_dotenv
from youtube.pipeline import process_youtube_url


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YouTube 字幕：有简体则下载；否则英文转简体（复用 translate 库）",
    )
    p.add_argument("url", help="YouTube 视频链接")
    p.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path("./out/youtube"),
        help="输出目录",
    )
    p.add_argument(
        "--cookies",
        type=Path,
        default=None,
        help="Netscape cookies 文件（如 /root/Desktop/www.youtube.com_cookies.txt）",
    )
    p.add_argument(
        "--no-translate",
        action="store_true",
        help="仅下载/导出，不调用翻译（英文路径则只出英文）",
    )
    p.add_argument("--translate-workers", type=int, default=4)
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

    translator = None
    if not args.no_translate:
        try:
            from translate_openai import OpenAICompatTranslator

            translator = OpenAICompatTranslator.from_env()
            if not args.quiet:
                print(
                    f"翻译: {translator.config.model} | "
                    f"{translator.pool.size} keys × "
                    f"{translator.config.rate_limit}/min "
                    f"≈ {translator.pool.effective_rpm()}/min",
                    flush=True,
                )
        except Exception as e:
            print(f"警告: 翻译器未就绪（{e}），将仅保存已下载字幕/英文", file=sys.stderr)

    # 无字幕时 Whisper 回退
    riva_client = None
    asr_pool = None
    try:
        from transcribe_whisper_nvidia import (
            DEFAULT_FUNCTION_ID,
            DEFAULT_RATE_LIMIT,
            DEFAULT_RATE_WINDOW_SEC,
            DEFAULT_SERVER,
            NvidiaApiKeyPool,
            import_riva,
            load_nvidia_api_keys,
        )
        import os

        keys = load_nvidia_api_keys(cli_key=None, cli_keys=None, keys_file=None)
        if keys:
            riva_client = import_riva()
            asr_pool = NvidiaApiKeyPool(
                keys,
                riva_client=riva_client,
                server=DEFAULT_SERVER,
                function_id=DEFAULT_FUNCTION_ID,
                max_mb=128,
                rate_limit=int(os.environ.get("WHISPER_RATE_LIMIT") or DEFAULT_RATE_LIMIT),
                rate_window_sec=float(
                    os.environ.get("WHISPER_RATE_WINDOW_SEC") or DEFAULT_RATE_WINDOW_SEC
                ),
            )
    except Exception as e:
        if not args.quiet:
            print(f"提示: Whisper 回退未启用（{e}）", file=sys.stderr)

    cookies = args.cookies.expanduser() if args.cookies else None
    if cookies and not cookies.is_file():
        print(f"错误: cookies 文件不存在: {cookies}", file=sys.stderr)
        return 2

    try:
        result = process_youtube_url(
            args.url,
            args.out_dir.expanduser().resolve(),
            cookies=cookies,
            translator=translator,
            riva_client=riva_client,
            asr_pool=asr_pool,
            translate_workers=args.translate_workers,
            fallback_audio=True,
            quiet=args.quiet,
        )
    except Exception as e:
        print(f"错误: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if result.error and result.mode == "none":
        print(f"错误: {result.error}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"mode={result.mode} lang={result.chosen_lang}")
        for k, p in result.outputs.items():
            print(f"  {k}: {p}")
        if result.error:
            print(f"警告: {result.error}", file=sys.stderr)

    # 成功条件：至少写出 zh_txt，或 en_txt（允许翻译失败时仍有英文）
    if "zh_txt" in result.outputs or "en_txt" in result.outputs:
        return 0 if not result.error or "zh_txt" in result.outputs or result.mode == "en" else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
