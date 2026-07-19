"""磁盘清理：释放内存中的音频，并异步清 HF 缓存（不阻塞主流程）。"""

from __future__ import annotations

import gc
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Callable


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
    单次 walk 内完成 stat+删除，避免重复扫描。
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
                if older_than_sec is not None and older_than_sec > 0:
                    if now - st.st_mtime < older_than_sec:
                        continue
                removed += st.st_size
                fp.unlink(missing_ok=True)
            except OSError:
                pass
        for name in dirs:
            dp = Path(root) / name
            try:
                dp.rmdir()
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
    同步清理 HF 缓存（可能较慢，请优先用 AsyncCacheCleaner）。

    - force=True：清理全部可删文件
    - 否则：free < min_free_gb 或 cache > max_cache_gb 时，先清 older_than_sec 旧文件
    """
    cache_dir = Path(cache_dir)
    free = disk_free_gb(cache_dir if cache_dir.exists() else cache_dir.parent)
    # 仅在需要判断 max_cache 时才全量算 size（force 可跳过）
    cache_gb = 0.0
    if not force and max_cache_gb > 0:
        cache_gb = dir_size_bytes(cache_dir) / (1024**3)
    need = force or free < min_free_gb or (max_cache_gb > 0 and cache_gb > max_cache_gb)
    removed = 0
    if need and cache_dir.exists():
        # 一次 walk 搞定：force 则 older=0 全删可删文件
        age = 0.0 if force else older_than_sec
        removed = purge_path(cache_dir, older_than_sec=age)
        # 非 force 但磁盘仍紧：再狠清一遍
        if not force and disk_free_gb(cache_dir) < min_free_gb:
            removed += purge_path(cache_dir, older_than_sec=0)
    gc.collect()
    return {
        "purged": need,
        "removed_bytes": removed,
        "free_gb_before": free,
        "cache_gb_before": cache_gb,
        "free_gb_after": disk_free_gb(cache_dir if cache_dir.exists() else cache_dir.parent),
    }


class AsyncCacheCleaner:
    """
    后台线程清理 HF 缓存，不阻塞 ASR/翻译主循环。

    - 同一时间最多一个清理任务
    - 若清理进行中再次请求，合并为「待再清一次」
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        min_free_gb: float = 2.0,
        max_cache_gb: float = 2.0,
        older_than_sec: float = 60,
        on_done: Callable[[dict], None] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.min_free_gb = min_free_gb
        self.max_cache_gb = max_cache_gb
        self.older_than_sec = older_than_sec
        self.on_done = on_done
        self._lock = threading.Lock()
        self._running = False
        self._pending = False
        self._pending_force = False
        self._last_info: dict | None = None

    def request(self, *, force: bool = False) -> bool:
        """
        请求一次清理。立即返回。
        返回 True 表示已启动新线程；False 表示已在跑/已排队。
        """
        with self._lock:
            if self._running:
                self._pending = True
                self._pending_force = self._pending_force or force
                return False
            self._running = True
            self._pending_force = force
        t = threading.Thread(
            target=self._run,
            name="hf-cache-cleaner",
            daemon=True,
            kwargs={"force": force},
        )
        t.start()
        return True

    def _run(self, force: bool = False) -> None:
        try:
            while True:
                info = purge_hf_cache(
                    self.cache_dir,
                    min_free_gb=self.min_free_gb,
                    max_cache_gb=self.max_cache_gb,
                    force=force,
                    older_than_sec=self.older_than_sec,
                )
                self._last_info = info
                if self.on_done and info.get("purged"):
                    try:
                        self.on_done(info)
                    except Exception:
                        pass
                with self._lock:
                    if self._pending:
                        force = self._pending_force
                        self._pending = False
                        self._pending_force = False
                        continue
                    self._running = False
                    break
        except Exception:
            with self._lock:
                self._running = False

    def wait(self, timeout: float | None = 120.0) -> None:
        """结束前可选等待当前清理完成（shutdown 用）。"""
        deadline = None if timeout is None else time.time() + timeout
        while True:
            with self._lock:
                if not self._running:
                    return
            if deadline is not None and time.time() >= deadline:
                return
            time.sleep(0.1)
