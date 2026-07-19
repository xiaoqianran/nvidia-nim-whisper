"""无字幕时：yt-dlp 下音频 → Whisper → 译中。"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.translate_cues import translate_plain_text
from youtube.captions import Cue, cues_to_plain_text, cues_to_srt, strip_to_plain_text


def _require_yt_dlp():
    try:
        import yt_dlp
    except ImportError as e:
        raise ImportError("需要 yt-dlp: pip install yt-dlp") from e
    return yt_dlp


def download_audio(
    url: str,
    out_dir: Path,
    *,
    cookies: str | Path | None = None,
    video_id: str = "audio",
) -> Path:
    """下载 bestaudio，尽量转成 wav（需 ffmpeg）。返回音频路径。"""
    from youtube.ytdlp_fetch import base_ydl_opts

    yt_dlp = _require_yt_dlp()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(out_dir / f"{video_id}.%(ext)s")
    opts: dict[str, Any] = base_ydl_opts(cookies=cookies)
    # 保留压缩音轨（m4a/webm），后续 ensure_wav 转 16k mono；避免整段解成巨量 wav
    opts.update(
        {
            "format": "bestaudio[ext=m4a]/bestaudio[abr<=128]/bestaudio/best",
            "outtmpl": outtmpl,
            "ignore_no_formats_error": False,
        }
    )

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        vid = info.get("id") or video_id

    # 优先 wav
    for ext in ("wav", "m4a", "webm", "opus", "mp3"):
        p = out_dir / f"{vid}.{ext}"
        if p.is_file():
            return p
        p2 = out_dir / f"{video_id}.{ext}"
        if p2.is_file():
            return p2
    # glob
    matches = list(out_dir.glob(f"{vid}.*")) + list(out_dir.glob(f"{video_id}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"音频下载后未找到文件: {out_dir}")


def whisper_transcribe_file(
    audio_path: Path,
    *,
    riva_client: Any,
    pool: Any,
    language_code: str = "en-US",
    sample_rate: int = 16000,
    chunk_seconds: float = 30.0,
    workers: int = 8,
    quiet: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    """
    用现有 Whisper Key 池转写本地音频。
    返回 (full_text, segments)。
    """
    from transcribe_whisper_nvidia import (
        ensure_wav,
        offline_transcribe,
        parse_response,
        run_chunks,
        split_wav_chunks,
        wav_duration_and_rate,
    )

    with tempfile.TemporaryDirectory(prefix="yt_whisper_") as tmp:
        tmp_dir = Path(tmp)
        wav_path = ensure_wav(Path(audio_path), tmp_dir, sample_rate)
        duration, rate, _ = wav_duration_and_rate(wav_path)
        chunks = split_wav_chunks(
            wav_path,
            tmp_dir / "chunks",
            chunk_seconds=chunk_seconds,
            overlap_seconds=0.0,
            sample_rate=sample_rate,
        )
        # 短音频可直接一次
        if duration and duration <= chunk_seconds and len(chunks) == 1:
            pcm = wav_path.read_bytes()
            # use raw frames
            import wave

            with wave.open(str(wav_path), "rb") as w:
                pcm = w.readframes(w.getnframes())
            key, asr = pool.acquire(quiet=quiet)
            resp = offline_transcribe(
                riva_client,
                asr,
                pcm,
                language_code=language_code,
                sample_rate=sample_rate,
                word_offsets=False,
            )
            text, segs = parse_response(resp)
            return text, segs

        full_text, segments, _meta = run_chunks(
            riva_client,
            pool,
            chunks,
            language_code=language_code,
            sample_rate=sample_rate,
            word_offsets=False,
            workers=workers,
            quiet=quiet,
        )
        return full_text, segments


def segments_to_cues(segments: list[dict[str, Any]], duration: float | None = None) -> list[Cue]:
    """Whisper segments → Cue 列表（无词级时间则按字符比例估）。"""
    cues: list[Cue] = []
    usable = [s for s in segments if (s.get("text") or "").strip()]
    if not usable:
        return cues

    has_time = any(s.get("start") is not None and s.get("end") is not None for s in usable)
    if has_time:
        for s in usable:
            try:
                start = float(s.get("start") or 0)
                end = float(s.get("end") or start)
            except (TypeError, ValueError):
                continue
            text = (s.get("text") or "").strip()
            if text:
                cues.append(Cue(start=start, end=end, text=text))
        return cues

    # 估算
    total_chars = sum(len(s.get("text") or "") for s in usable) or 1
    span = float(duration or 0) or float(len(usable) * 3.0)
    t = 0.0
    for s in usable:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        dur = span * (len(text) / total_chars)
        cues.append(Cue(start=t, end=min(span, t + dur), text=text))
        t = cues[-1].end
    return cues


def process_audio_fallback(
    url: str,
    out_dir: Path,
    *,
    stem: str,
    video_id: str,
    cookies: str | Path | None,
    riva_client: Any,
    pool: Any,
    translator: Any | None,
    language_code: str = "en-US",
    translate_workers: int = 4,
    whisper_workers: int = 8,
    quiet: bool = False,
    log: Callable[[str], None] | None = None,
    keep_audio: bool = False,
    output_profile: str = "full",
) -> dict[str, Path]:
    """
    下载音频 → Whisper → 英文 txt/srt → 译中 zh txt/srt。
    返回 outputs 字典。
    """

    def _log(msg: str) -> None:
        if log:
            log(msg)
        elif not quiet:
            print(msg, flush=True)

    out_dir = Path(out_dir)
    audio_dir = out_dir / "_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    _log("无字幕，下载音频并用 Whisper 转写…")
    audio_path = download_audio(url, audio_dir, cookies=cookies, video_id=video_id)
    _log(f"音频: {audio_path}")

    text, segments = whisper_transcribe_file(
        audio_path,
        riva_client=riva_client,
        pool=pool,
        language_code=language_code,
        workers=whisper_workers,
        quiet=quiet,
    )
    # duration
    try:
        from transcribe_whisper_nvidia import probe_duration

        duration = probe_duration(audio_path)
    except Exception:
        duration = None

    cues = segments_to_cues(segments, duration)
    if not cues and text:
        cues = [Cue(0.0, float(duration or 1.0), text)]

    plain = strip_to_plain_text(cues_to_plain_text(cues) if cues else (text or ""))
    outputs: dict[str, Path] = {}
    zh_only = output_profile == "zh_only"

    if not zh_only:
        en_txt = out_dir / f"{stem}.en.txt"
        en_txt.write_text(plain if plain.endswith("\n") else plain + "\n", encoding="utf-8")
        outputs["en_txt"] = en_txt
        en_srt = out_dir / f"{stem}.en.srt"
        en_srt.write_text(cues_to_srt(cues), encoding="utf-8")
        outputs["en_srt"] = en_srt
        _log(f"Whisper 原文: {en_txt}")
    outputs["source"] = Path("whisper_audio")

    if translator is not None and plain.strip():
        from common.lang_detect import looks_chinese, needs_translate_to_zh
        from common.translate_cues import translate_cues
        from youtube.captions import Cue as YCue

        # Whisper 已是中文则不再翻译，直接作为简体产物
        if looks_chinese(plain, threshold=0.3) or not needs_translate_to_zh(plain[:500]):
            _log("转写已是中文，跳过翻译")
            zh_cues = cues
            zh_plain = plain
        else:
            _log("Whisper 结果译成简体…")
            zh_cues = translate_cues(
                cues,
                translator,
                workers=translate_workers,
                quiet=quiet,
                cue_factory=lambda s, e, t: YCue(s, e, t),
            )
            zh_plain = strip_to_plain_text(cues_to_plain_text(zh_cues))
        zh_srt = out_dir / f"{stem}.zh.srt"
        zh_srt.write_text(cues_to_srt(zh_cues), encoding="utf-8")
        outputs["zh_srt"] = zh_srt
        zh_txt = out_dir / f"{stem}.zh.txt"
        zh_txt.write_text(zh_plain if zh_plain.endswith("\n") else zh_plain + "\n", encoding="utf-8")
        outputs["zh_txt"] = zh_txt
        _log(f"简体: {zh_txt}")

    if not keep_audio:
        try:
            audio_path.unlink(missing_ok=True)
            # 清理同 stem 残留
            for p in audio_dir.glob(f"{video_id}.*"):
                p.unlink(missing_ok=True)
        except OSError:
            pass

    return outputs
