"""YouTube 单视频：字幕优先，否则音频 Whisper → 简体。"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from youtube.captions import (
    Cue,
    cues_to_plain_text,
    cues_to_srt,
    parse_subtitle_file,
    pick_caption_track,
    safe_stem,
    strip_to_plain_text,
)
from youtube.ytdlp_fetch import download_captions, dump_lang_list_json, list_caption_langs

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass
class YoutubeProcessResult:
    mode: str  # zh-Hans | en | whisper | none
    video_id: str
    title: str
    chosen_lang: str | None
    outputs: dict[str, Path] = field(default_factory=dict)
    error: str | None = None
    description: str = ""
    description_zh: str = ""
    title_zh: str = ""


def process_youtube_url(
    url: str,
    out_dir: Path,
    *,
    cookies: str | Path | None = None,
    translator: Any | None = None,
    riva_client: Any | None = None,
    asr_pool: Any | None = None,
    translate_workers: int = 4,
    whisper_workers: int = 8,
    language_code: str = "en-US",
    fallback_audio: bool = True,
    keep_audio: bool = False,
    quiet: bool = False,
    log: Callable[[str], None] | None = None,
    prefetched_meta: dict[str, Any] | None = None,
) -> YoutubeProcessResult:
    """
    主流程：
    1. 有简体字幕 → 下简体 + 简体 txt/srt
    2. 有英文字幕 → 英文 txt/srt + 译简体
    3. 都没有且 fallback_audio → 下音频 Whisper + 译简体
    """

    def _log(msg: str) -> None:
        if log:
            log(msg)
        elif not quiet:
            print(msg, flush=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_subs_raw"
    work.mkdir(parents=True, exist_ok=True)

    if prefetched_meta:
        meta = prefetched_meta
    else:
        meta = list_caption_langs(url, cookies=cookies)

    video_id = meta.get("id") or "unknown"
    title = meta.get("title") or video_id
    stem = safe_stem(title, video_id)
    dump_lang_list_json(meta, out_dir / f"{stem}.langs.json")

    mode, track = pick_caption_track(meta.get("manual") or [], meta.get("auto") or [])

    # —— 路径 A/B：字幕 ——
    if mode != "none" and track is not None:
        _log(f"视频: {title} ({video_id})")
        _log(f"选择字幕: {track.lang} ({track.kind}) → 路径 {mode}")

        files = download_captions(
            url,
            work,
            [track.lang],
            cookies=cookies,
            prefer_ext="vtt",
        )
        if track.lang not in files:
            _log(f"字幕下载失败: {track.lang}，尝试音频回退…")
            mode = "none"
            track = None
        else:
            src_path = files[track.lang]
            cues = parse_subtitle_file(src_path)
            plain = strip_to_plain_text(cues_to_plain_text(cues))
            outputs: dict[str, Path] = {}

            raw_copy = out_dir / f"{stem}.{track.lang}{src_path.suffix}"
            raw_copy.write_bytes(src_path.read_bytes())
            outputs["source_sub"] = raw_copy

            if mode == "zh-Hans":
                zh_txt = out_dir / f"{stem}.zh.txt"
                zh_txt.write_text(
                    plain if plain.endswith("\n") else plain + "\n", encoding="utf-8"
                )
                outputs["zh_txt"] = zh_txt
                zh_srt = out_dir / f"{stem}.zh.srt"
                zh_srt.write_text(cues_to_srt(cues), encoding="utf-8")
                outputs["zh_srt"] = zh_srt
                _log(f"已写简体文本: {zh_txt}")
                return YoutubeProcessResult(
                    mode=mode,
                    video_id=video_id,
                    title=title,
                    chosen_lang=track.lang,
                    outputs=outputs,
                )

            # EN captions
            en_txt = out_dir / f"{stem}.en.txt"
            en_txt.write_text(
                plain if plain.endswith("\n") else plain + "\n", encoding="utf-8"
            )
            outputs["en_txt"] = en_txt
            en_srt = out_dir / f"{stem}.en.srt"
            en_srt.write_text(cues_to_srt(cues), encoding="utf-8")
            outputs["en_srt"] = en_srt
            _log(f"已写英文文本: {en_txt}")

            if translator is None:
                return YoutubeProcessResult(
                    mode=mode,
                    video_id=video_id,
                    title=title,
                    chosen_lang=track.lang,
                    outputs=outputs,
                    error="需要翻译器才能生成简体中文（未提供 translator）",
                )

            _log("正在英→简体翻译字幕…")
            from common.translate_cues import translate_cues

            zh_cues = translate_cues(
                cues,
                translator,
                workers=translate_workers,
                quiet=quiet,
                cue_factory=lambda s, e, t: Cue(s, e, t),
            )
            zh_srt = out_dir / f"{stem}.zh.srt"
            zh_srt.write_text(cues_to_srt(zh_cues), encoding="utf-8")
            outputs["zh_srt"] = zh_srt
            zh_plain = strip_to_plain_text(cues_to_plain_text(zh_cues))
            zh_txt = out_dir / f"{stem}.zh.txt"
            zh_txt.write_text(
                zh_plain if zh_plain.endswith("\n") else zh_plain + "\n", encoding="utf-8"
            )
            outputs["zh_txt"] = zh_txt
            _log(f"已写简体字幕: {zh_srt}")
            _log(f"已写简体文本: {zh_txt}")
            return YoutubeProcessResult(
                mode=mode,
                video_id=video_id,
                title=title,
                chosen_lang=track.lang,
                outputs=outputs,
            )

    # —— 路径 C：无字幕 → 音频 Whisper ——
    if not fallback_audio:
        return YoutubeProcessResult(
            mode="none",
            video_id=video_id,
            title=title,
            chosen_lang=None,
            error="未找到简体/英文字幕，且未启用音频回退",
        )

    if riva_client is None or asr_pool is None:
        return YoutubeProcessResult(
            mode="none",
            video_id=video_id,
            title=title,
            chosen_lang=None,
            error="未找到字幕，且未配置 Whisper（riva_client/asr_pool）",
        )

    from youtube.audio_whisper import process_audio_fallback

    _log(f"视频: {title} ({video_id})")
    try:
        outputs = process_audio_fallback(
            url,
            out_dir,
            stem=stem,
            video_id=video_id,
            cookies=cookies,
            riva_client=riva_client,
            pool=asr_pool,
            translator=translator,
            language_code=language_code,
            translate_workers=translate_workers,
            whisper_workers=whisper_workers,
            quiet=quiet,
            log=_log,
            keep_audio=keep_audio,
        )
    except Exception as e:
        return YoutubeProcessResult(
            mode="whisper",
            video_id=video_id,
            title=title,
            chosen_lang=None,
            error=f"音频 Whisper 失败: {type(e).__name__}: {e}",
        )

    return YoutubeProcessResult(
        mode="whisper",
        video_id=video_id,
        title=title,
        chosen_lang="whisper",
        outputs=outputs,
    )
