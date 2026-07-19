"""磁盘清理：释放内存中的音频，并定期清 HF 缓存。"""

from __future__ import annotations

import gc
import os
import shutil
import time
from pathlib import Path


def disk_free_gb(path: Path | str) -> float:
    p = Path(path)
    try:
        u = shutil.disk_usage(p if p.exists() else p.parent)
        return u.free / (1024**3)
    except OSError:
        return -1.0


def dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def free_audio_memory(sample: object) -> None:
    """丢掉样本上的大数组引用，便于 GC。"""
    for attr in ("audio_array", "raw"):
        if hasattr(sample, attr):
            try:
                setattr(sample, attr, None)
            except Exception:
                pass


def purge_path(path: Path, *, older_than_sec: float | None = 300) -> int:
    """
    删除目录下较旧的文件（默认 5 分钟未修改）。
    返回删除的字节数。不删目录本身。
    """
    if not path.exists():
        return 0
    now = time.time()
    removed = 0
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            fp = Path(root) / name
            try:
                st = fp.stat()
                if older_than_sec is not None and now - st.st_mtime < older_than_sec:
                    continue
                removed += st.st_size
                fp.unlink(missing_ok=True)
            except OSError:
                pass
        for name in dirs:
            dp = Path(root) / name
            try:
                dp.rmdir()  # 仅空目录
            except OSError:
                pass
    return removed


def purge_hf_cache(
    cache_dir: Path,
    *,
    min_free_gb: float = 2.0,
    max_cache_gb: float = 3.0,
    force: bool = False,
    older_than_sec: float = 120,
) -> dict:
    """
    在磁盘紧张或缓存过大时清理 HF 缓存。

    - force=True：按 older_than_sec 清缓存文件
    - 否则：free < min_free_gb 或 cache > max_cache_gb 时清理
    """
    cache_dir = Path(cache_dir)
    free = disk_free_gb(cache_dir if cache_dir.exists() else cache_dir.parent)
    cache_gb = dir_size_bytes(cache_dir) / (1024**3)
    need = force or free < min_free_gb or cache_gb > max_cache_gb
    removed = 0
    if need and cache_dir.exists():
        removed = purge_path(cache_dir, older_than_sec=older_than_sec if not force else 0)
        # force 且仍大：再扫一遍全部文件
        if force or cache_gb > max_cache_gb:
            removed += purge_path(cache_dir, older_than_sec=0)
    gc.collect()
    return {
        "purged": need,
        "removed_bytes": removed,
        "free_gb_before": free,
        "cache_gb_before": cache_gb,
        "free_gb_after": disk_free_gb(cache_dir if cache_dir.exists() else cache_dir.parent),
    }
