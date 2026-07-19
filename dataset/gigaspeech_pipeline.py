#!/usr/bin/env python3
"""
GigaSpeech 增量流水线：HuggingFace streaming → Whisper →（可选）中文翻译。

磁盘策略：不整库下载；仅保留 JSONL 结果 + SQLite 进度；音频在内存处理完即丢弃。

用法:
  export HF_TOKEN=hf_...
  export NVIDIA_API_KEYS_FILE=nvidia_api_keys.txt
  export OPENAI_API_KEY=... OPENAI_BASE_URL=... OPENAI_MODEL=...

  python -m dataset.gigaspeech_pipeline --subset xs --max-samples 50 --translate
  python -m dataset.gigaspeech_pipeline --subset xs --resume --out-dir ./out/gs_xs
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# 仓库根
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset.cleanup import disk_free_gb, purge_hf_cache
from dataset.sink import JsonlSink
from dataset.source_hf import iter_gigaspeech
from dataset.state import SegmentState
from dataset.worker import process_sample
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
    mask_api_key,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GigaSpeech streaming → Whisper → 中文（增量、低磁盘）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--subset", default=os.environ.get("GIGASPEECH_SUBSET", "xs"), help="xs/s/m/l/xl/dev/test")
    p.add_argument("--split", default="train", help="train 等")
    p.add_argument("--max-samples", type=int, default=0, help="最多处理条数，0=不限制")
    p.add_argument("--out-dir", type=Path, default=Path("./out/gigaspeech"), help="输出目录")
    p.add_argument("--resume", action="store_true", default=True, help="跳过已成功样本")
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--retry-errors", action="store_true", help="重试 state 中 error 的样本")
    p.add_argument("--translate", action="store_true", help="译成中文")
    p.add_argument(
        "--skip-whisper-use-ref",
        action="store_true",
        help="跳过 ASR，直接用官方 ref_text（可只做翻译）",
    )
    p.add_argument("--max-in-flight", type=int, default=12, help="同时在处理的样本上限")
    p.add_argument("--workers", type=int, default=None, help="线程数，默认随 Key 数")
    p.add_argument("--language", default="en-US")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--api-key", default=None)
    p.add_argument("--api-keys", default=None)
    p.add_argument("--api-keys-file", type=Path, default=None)
    p.add_argument("--function-id", default=DEFAULT_FUNCTION_ID)
    p.add_argument("--server", default=DEFAULT_SERVER)
    p.add_argument("--rate-limit", type=int, default=None)
    p.add_argument("--rate-window-sec", type=float, default=None)
    p.add_argument("--openai-api-key", default=None)
    p.add_argument("--openai-base-url", default=None)
    p.add_argument("--openai-model", default=None)
    p.add_argument("--to", dest="translate_to", default=None)
    p.add_argument("--translate-rate-limit", type=int, default=None)
    p.add_argument("--hf-token", default=None, help="覆盖 HF_TOKEN")
    p.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="限制 HF 缓存目录（省磁盘）；处理完会定期清理",
    )
    p.add_argument(
        "--cleanup-every",
        type=int,
        default=25,
        help="每成功 N 条清理一次 HF 缓存（0=仅结束时清理）",
    )
    p.add_argument(
        "--min-free-gb",
        type=float,
        default=2.0,
        help="磁盘剩余低于此值时强制清理缓存",
    )
    p.add_argument(
        "--max-cache-gb",
        type=float,
        default=2.0,
        help="HF 缓存超过此大小时清理",
    )
    p.add_argument("--env-file", type=Path, default=None)
    p.add_argument("-q", "--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    script_dir = _ROOT
    cwd = Path.cwd()
    if args.env_file:
        load_dotenv([args.env_file.expanduser().resolve()])
    else:
        load_dotenv([cwd / ".env", script_dir / ".env"])

    # 默认把 HF 缓存放到 out-dir 下，便于清理且不占系统盘
    if args.hf_cache_dir:
        cache = args.hf_cache_dir.expanduser().resolve()
    else:
        cache = (args.out_dir.expanduser().resolve() / ".hf_cache")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_DATASETS_CACHE"] = str(cache / "datasets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache / "hub")
    # 尽量少落盘
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    if args.rate_limit is None:
        args.rate_limit = int(
            os.environ.get("WHISPER_RATE_LIMIT")
            or os.environ.get("NVIDIA_RATE_LIMIT")
            or DEFAULT_RATE_LIMIT
        )
    if args.rate_window_sec is None:
        args.rate_window_sec = float(
            os.environ.get("WHISPER_RATE_WINDOW_SEC") or DEFAULT_RATE_WINDOW_SEC
        )

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"gigaspeech_{args.subset}_{args.split}.jsonl"
    state_path = out_dir / f"gigaspeech_{args.subset}_{args.split}.state.sqlite"

    need_asr = not args.skip_whisper_use_ref
    api_keys: list[str] = []
    pool = None
    riva_client = None

    if need_asr:
        api_keys = load_nvidia_api_keys(
            cli_key=args.api_key,
            cli_keys=args.api_keys,
            keys_file=args.api_keys_file.expanduser() if args.api_keys_file else None,
        )
        if not api_keys:
            print(
                "错误: 需要 NVIDIA API Key（或使用 --skip-whisper-use-ref）",
                file=sys.stderr,
            )
            return 1
        riva_client = import_riva()
        pool = NvidiaApiKeyPool(
            api_keys,
            riva_client=riva_client,
            server=args.server,
            function_id=args.function_id,
            max_mb=64,
            rate_limit=args.rate_limit,
            rate_window_sec=args.rate_window_sec,
        )
        if args.workers is None:
            args.workers = min(32, max(DEFAULT_WORKERS, len(api_keys) * 4))
    else:
        if args.workers is None:
            args.workers = 4

    translator = None
    if args.translate:
        from translate_openai import OpenAICompatTranslator

        try:
            translator = OpenAICompatTranslator.from_env(
                api_key=args.openai_api_key,
                base_url=args.openai_base_url,
                model=args.openai_model,
                target=args.translate_to,
                rate_limit=args.translate_rate_limit,
            )
        except ValueError as e:
            print(f"错误: {e}", file=sys.stderr)
            return 1

    workers = max(1, min(args.workers, args.max_in_flight))
    state = SegmentState(state_path)
    sink = JsonlSink(jsonl_path)

    if not args.quiet:
        print(f"子集={args.subset} split={args.split} max_samples={args.max_samples or '∞'}")
        print(f"输出: {jsonl_path}")
        print(f"状态: {state_path}")
        print(f"磁盘剩余: {disk_free_gb(out_dir):.2f} GB @ {out_dir}")
        print(f"HF 缓存: {cache}（每 {args.cleanup_every or '∞'} 条清理）")
        if pool:
            print(
                f"ASR Key 池: {pool.size} × {args.rate_limit}/{args.rate_window_sec:g}s"
                f" ≈ {pool.effective_rpm()}/min | workers={workers}"
            )
            for i, k in enumerate(api_keys, 1):
                print(f"  [{i}] {mask_api_key(k)}")
        else:
            print("ASR: 跳过（--skip-whisper-use-ref）")
        if translator:
            print(
                f"翻译: {translator.config.model} → {translator.config.target} "
                f"| 限速 {translator.config.rate_limit}/{translator.config.rate_window_sec:g}s"
            )
        else:
            print("翻译: 关闭")
        print("模式: HuggingFace streaming（增量，不整库落盘）")

    done_skip = 0
    ok = 0
    err = 0
    submitted = 0
    t0 = time.time()

    def _maybe_cleanup(force: bool = False) -> None:
        if not force and args.cleanup_every <= 0:
            return
        if not force and args.cleanup_every > 0 and ok > 0 and ok % args.cleanup_every != 0:
            return
        info = purge_hf_cache(
            cache,
            min_free_gb=args.min_free_gb,
            max_cache_gb=args.max_cache_gb,
            force=force or (args.cleanup_every > 0 and ok > 0 and ok % args.cleanup_every == 0),
            older_than_sec=60,
        )
        if not args.quiet and info.get("purged"):
            mb = info["removed_bytes"] / (1024 * 1024)
            print(
                f"  [cleanup] 清理缓存 {mb:.1f} MB | "
                f"free {info['free_gb_before']:.2f}→{info['free_gb_after']:.2f} GB",
                flush=True,
            )

    def _handle_result(rec: dict[str, Any]) -> None:
        nonlocal ok, err
        sid = rec["segment_id"]
        if rec.get("error"):
            state.mark(sid, "error", rec["error"])
            err += 1
            if not args.quiet:
                print(f"  FAIL {sid}: {rec['error']}", flush=True)
        else:
            state.mark(sid, "ok")
            sink.write(rec)
            ok += 1
            if not args.quiet:
                prev = (rec.get("whisper_text") or "")[:60]
                zh = (rec.get("text_zh") or "")[:40]
                extra = f" | zh={zh}…" if zh else ""
                print(
                    f"  OK [{ok}] {sid} {rec.get('duration_sec')}s | {prev}…{extra}",
                    flush=True,
                )
            _maybe_cleanup(force=False)

    exit_code = 0
    try:
        pending: set[Any] = set()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for sample in iter_gigaspeech(
                subset=args.subset,
                split=args.split,
                max_samples=args.max_samples,
                token=args.hf_token,
            ):
                if args.resume and state.is_done(sample.segment_id):
                    done_skip += 1
                    continue
                if not args.retry_errors:
                    # error 也默认跳过，除非 --retry-errors
                    pass

                while len(pending) >= args.max_in_flight:
                    finished = next(as_completed(pending))
                    pending.remove(finished)
                    _handle_result(finished.result())

                fut = ex.submit(
                    process_sample,
                    sample,
                    riva_client=riva_client,
                    pool=pool,
                    translator=translator,
                    language_code=args.language,
                    sample_rate=args.sample_rate,
                    skip_whisper_use_ref=args.skip_whisper_use_ref,
                )
                pending.add(fut)
                submitted += 1

            for fut in as_completed(pending):
                _handle_result(fut.result())

    except KeyboardInterrupt:
        print("\n中断：已写出部分结果，可用 --resume 继续", file=sys.stderr)
        exit_code = 130
    except Exception as e:
        print(f"错误: {type(e).__name__}: {e}", file=sys.stderr)
        exit_code = 1
    finally:
        elapsed = time.time() - t0
        if not args.quiet:
            print("--- 汇总 ---")
            print(f"跳过(已完成): {done_skip}")
            print(f"成功: {ok}  失败: {err}  提交: {submitted}")
            print(f"耗时: {elapsed:.1f}s")
            print(f"JSONL 行(本次打开后写入): {sink.written}")
            try:
                print(f"state: {state.counts()}")
            except Exception:
                pass
            if pool:
                print("Key 统计:")
                for row in pool.stats_summary():
                    print(f"  {row['key']}: {row['requests']} 次")
            # 结束强制清缓存
            try:
                info = purge_hf_cache(
                    cache,
                    min_free_gb=0,
                    max_cache_gb=0,
                    force=True,
                    older_than_sec=0,
                )
                print(
                    f"结束清理缓存: {info['removed_bytes']/(1024*1024):.1f} MB | "
                    f"磁盘剩余 {disk_free_gb(out_dir):.2f} GB"
                )
            except Exception as ce:
                print(f"结束清理失败: {ce}")
            print(f"磁盘剩余: {disk_free_gb(out_dir):.2f} GB")
        else:
            try:
                purge_hf_cache(cache, force=True, older_than_sec=0, min_free_gb=0, max_cache_gb=0)
            except Exception:
                pass
        try:
            state.close()
        except Exception:
            pass
        try:
            sink.close()
        except Exception:
            pass

    if exit_code != 0:
        return exit_code
    return 0 if err == 0 or ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
