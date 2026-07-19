#!/usr/bin/env python3
"""
持续拉取 GigaSpeech 并 Whisper+翻译的 Python 入口（与 run_gigaspeech_continuous.sh 等价）。

默认开启 --translate，音频不落盘长期保存，HF 缓存定期清理。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset.gigaspeech_pipeline import main as pipeline_main


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # 若用户未显式关翻译，则默认开
    if "--translate" not in argv and "--no-translate" not in argv:
        argv = ["--translate", *argv]
    if "--cleanup-every" not in argv:
        argv = ["--cleanup-every", "20", *argv]
    if "--max-cache-gb" not in argv:
        argv = ["--max-cache-gb", "1.5", *argv]
    if "--min-free-gb" not in argv:
        argv = ["--min-free-gb", "2", *argv]
    return pipeline_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
