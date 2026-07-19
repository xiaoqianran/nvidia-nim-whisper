"""单样本：PCM → Whisper →（可选）译中。"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataset.audio_util import to_pcm16_mono_16k
from dataset.source_hf import GigaSample

# 保证可 import 仓库根模块
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def process_sample(
    sample: GigaSample,
    *,
    riva_client: Any,
    pool: Any,
    translator: Any | None,
    language_code: str = "en-US",
    sample_rate: int = 16000,
    skip_whisper_use_ref: bool = False,
    word_offsets: bool = False,
) -> dict[str, Any]:
    """处理一条 GigaSpeech 样本，返回可 JSONL 写出的记录。"""
    from transcribe_whisper_nvidia import (
        mask_api_key,
        offline_transcribe,
        parse_response,
    )

    t0 = time.time()
    rec: dict[str, Any] = {
        "segment_id": sample.segment_id,
        "subset": sample.subset,
        "split": sample.split,
        "ref_text": sample.ref_text,
        "whisper_text": None,
        "text_zh": None,
        "duration_sec": None,
        "audio_id": sample.audio_id,
        "begin_time": sample.begin_time,
        "end_time": sample.end_time,
        "asr_key": None,
        "error": None,
        "ts": datetime.now(timezone.utc).isoformat(),
    }

    try:
        pcm, duration = to_pcm16_mono_16k(sample.audio_array, sample.sampling_rate, sample_rate)
        rec["duration_sec"] = round(duration, 3)

        if skip_whisper_use_ref:
            whisper_text = sample.ref_text or ""
            rec["whisper_text"] = whisper_text
            rec["asr_key"] = "ref_text"
        else:
            if not pcm or duration <= 0.05:
                raise ValueError("音频过短或为空")
            key, asr = pool.acquire(quiet=True)
            rec["asr_key"] = mask_api_key(key)
            resp = offline_transcribe(
                riva_client,
                asr,
                pcm,
                language_code=language_code,
                sample_rate=sample_rate,
                word_offsets=word_offsets,
            )
            text, _segs = parse_response(resp)
            whisper_text = text or ""
            rec["whisper_text"] = whisper_text

        if translator is not None:
            src = (rec.get("whisper_text") or sample.ref_text or "").strip()
            if src:
                rec["text_zh"] = translator.translate(src, quiet=True)

        rec["elapsed_sec"] = round(time.time() - t0, 3)
        return rec
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["elapsed_sec"] = round(time.time() - t0, 3)
        return rec
