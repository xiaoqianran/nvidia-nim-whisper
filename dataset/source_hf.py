"""HuggingFace GigaSpeech streaming 数据源。"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass
class GigaSample:
    segment_id: str
    subset: str
    split: str
    ref_text: str
    audio_array: Any  # numpy-like
    sampling_rate: int
    audio_id: str | None = None
    begin_time: float | None = None
    end_time: float | None = None
    raw: dict[str, Any] | None = None


def _decode_audio_dict(audio: dict[str, Any]) -> tuple[Any, int]:
    """
    解码 HF Audio 字段。
    新版 datasets 默认 decode 依赖 torchcodec；我们用 decode=False + soundfile。
    """
    if audio.get("array") is not None:
        return audio["array"], int(audio.get("sampling_rate") or 16000)

    import numpy as np
    import soundfile as sf

    raw = audio.get("bytes")
    if raw:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32")
        return np.asarray(data), int(sr)

    path = audio.get("path")
    if path and os.path.isfile(path):
        data, sr = sf.read(path, dtype="float32")
        return np.asarray(data), int(sr)

    raise ValueError("audio 字段无法解码（无 array/bytes/path）")


def iter_gigaspeech(
    subset: str = "xs",
    split: str = "train",
    *,
    max_samples: int = 0,
    token: str | None = None,
) -> Iterator[GigaSample]:
    """
    流式迭代 GigaSpeech，不整库下载。

    需要: pip install datasets soundfile numpy
    授权: HF_TOKEN，并接受数据集条款。
    """
    try:
        from datasets import Audio, load_dataset
    except ImportError as e:
        raise ImportError(
            "需要 datasets 库: pip install datasets huggingface_hub soundfile"
        ) from e

    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    kwargs: dict[str, Any] = {
        "path": "speechcolab/gigaspeech",
        "name": subset,
        "split": split,
        "streaming": True,
    }
    if tok:
        kwargs["token"] = tok

    ds = load_dataset(**kwargs)
    # 关键：不在 datasets 内 decode（避免强制 torchcodec）
    try:
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception:
        pass

    n = 0
    for row in ds:
        sid = str(row.get("segment_id") or row.get("id") or f"idx_{n}")
        audio = row.get("audio") or {}
        try:
            if isinstance(audio, dict):
                arr, sr = _decode_audio_dict(audio)
            else:
                n += 1
                continue
        except Exception:
            n += 1
            continue

        ref = row.get("text") or row.get("transcript") or ""
        if isinstance(ref, str):
            ref = ref.strip()
        else:
            ref = str(ref)

        yield GigaSample(
            segment_id=sid,
            subset=subset,
            split=split,
            ref_text=ref,
            audio_array=arr,
            sampling_rate=sr,
            audio_id=row.get("audio_id"),
            begin_time=_f(row.get("begin_time")),
            end_time=_f(row.get("end_time")),
            raw=None,
        )
        n += 1
        if max_samples > 0 and n >= max_samples:
            break


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
