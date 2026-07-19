"""内存音频 → 16k mono PCM16（不落盘）。"""

from __future__ import annotations

from typing import Any


def to_pcm16_mono_16k(array: Any, sampling_rate: int, target_sr: int = 16000) -> tuple[bytes, float]:
    """
    返回 (pcm_bytes, duration_sec)。
    使用 numpy 线性重采样，避免强制依赖 librosa。
    """
    import numpy as np

    x = np.asarray(array, dtype=np.float64)
    if x.ndim > 1:
        # (channels, samples) or (samples, channels)
        if x.shape[0] <= 8 and x.shape[0] < x.shape[-1]:
            x = x.mean(axis=0)
        else:
            x = x.mean(axis=-1)
    x = x.reshape(-1)

    sr = int(sampling_rate) or target_sr
    if sr != target_sr and len(x) > 1:
        n_out = max(1, int(round(len(x) * target_sr / sr)))
        xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
        xq = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        x = np.interp(xq, xp, x)
        sr = target_sr

    # 若是 int 输入已放大，归一化
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 1.5:
        x = x / 32768.0
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16).tobytes()
    duration = len(x) / float(sr) if sr else 0.0
    return pcm, duration
