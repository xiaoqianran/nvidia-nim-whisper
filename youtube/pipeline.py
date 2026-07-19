"""YouTube URL → 简体字幕/文本流水线。"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from youtube.captions import (
    Cue,
    CaptionTrack,
    cues_to_plain_text,
    cues_to_srt,
    parse_subtitle_file,
    pick_caption_track,
    safe_stem,
)
from youtube.ytdlp_fetch import download_captions, dump_lang_list_json, list_caption_langs

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass
class YoutubeProcessResult:
    mode: str  # zh-Hans | en | none
    video_id: str
    title: str
    chosen_lang: str | None
    outputs: dict[str, Path] = field(default_factory=dict)
    error: str | None = None


def _translate_cues(
    cues: list[Cue],
    translator: Any,
    *,
    workers: int = 4,
    quiet: bool = False,
) -> list[Cue]:
    """用现有 translate 模块按 cue 译成中文。"""
    if not cues:
        return []
    segs = [{"text": c.text, "i": i} for i, c in enumerate(cues)]
    translated = translator.translate_segments(
        segs,
        text_key="text",
        out_key="text_zh",
        workers=max(1, workers),
        quiet=quiet,
    )
    out: list[Cue] = []
    for c, t in zip(cues, translated):
        zh = (t.get("text_zh") or "").strip() or c.text
        out.append(Cue(start=c.start, end=c.end, text=zh))
    return out


def process_youtube_url(
    url: str,
    out_dir: Path,
    *,
    cookies: str | Path | None = None,
    translator: Any | None = None,
    translate_workers: int = 4,
    quiet: bool = False,
    log: Callable[[str], None] | None = None,
) -> YoutubeProcessResult:
    """
    主流程：
    - 有简体字幕 → 下简体 + 简体 txt
    - 否则有英文 → 下英文 + 英文 txt + 译简体 srt + 简体 txt
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

    meta = list_caption_langs(url, cookies=cookies)
    video_id = meta["id"] or "unknown"
    title = meta["title"] or video_id
    stem = safe_stem(title, video_id)
    dump_lang_list_json(meta, out_dir / f"{stem}.langs.json")

    mode, track = pick_caption_track(meta["manual"], meta["auto"])
    if mode == "none" or track is None:
        return YoutubeProcessResult(
            mode="none",
            video_id=video_id,
            title=title,
            chosen_lang=None,
            error="未找到简体中文或英文字幕",
        )

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
        return YoutubeProcessResult(
            mode=mode,
            video_id=video_id,
            title=title,
            chosen_lang=track.lang,
            error=f"字幕下载失败: {track.lang}",
        )

    src_path = files[track.lang]
    cues = parse_subtitle_file(src_path)
    plain = cues_to_plain_text(cues)
    outputs: dict[str, Path] = {}

    # 保留原始字幕副本到 out_dir
    raw_copy = out_dir / f"{stem}.{track.lang}{src_path.suffix}"
    raw_copy.write_bytes(src_path.read_bytes())
    outputs["source_sub"] = raw_copy

    if mode == "zh-Hans":
        zh_txt = out_dir / f"{stem}.zh.txt"
        zh_txt.write_text(plain if plain.endswith("\n") else plain + "\n", encoding="utf-8")
        outputs["zh_txt"] = zh_txt
        # 若源是 vtt，额外给一份 srt 方便挂载
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

    # EN path
    en_txt = out_dir / f"{stem}.en.txt"
    en_txt.write_text(plain if plain.endswith("\n") else plain + "\n", encoding="utf-8")
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
    zh_cues = _translate_cues(
        cues, translator, workers=translate_workers, quiet=quiet
    )
    zh_srt = out_dir / f"{stem}.zh.srt"
    zh_srt.write_text(cues_to_srt(zh_cues), encoding="utf-8")
    outputs["zh_srt"] = zh_srt
    zh_txt = out_dir / f"{stem}.zh.txt"
    zh_plain = cues_to_plain_text(zh_cues)
    zh_txt.write_text(zh_plain if zh_plain.endswith("\n") else zh_plain + "\n", encoding="utf-8")
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
