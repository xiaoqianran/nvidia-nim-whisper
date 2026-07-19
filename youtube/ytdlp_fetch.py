"""yt-dlp 适配：列出/下载字幕，不拉整视频。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _require_yt_dlp():
    try:
        import yt_dlp
    except ImportError as e:
        raise ImportError("需要 yt-dlp: pip install yt-dlp") from e
    return yt_dlp


def list_caption_langs(
    url: str,
    *,
    cookies: str | Path | None = None,
) -> dict[str, Any]:
    """
    返回 {
      title, id, manual: [lang...], auto: [lang...]
    }
    """
    yt_dlp = _require_yt_dlp()
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignore_no_formats_error": True,
        "noplaylist": True,
    }
    if cookies:
        opts["cookiefile"] = str(cookies)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    manual = sorted((info.get("subtitles") or {}).keys())
    auto = sorted((info.get("automatic_captions") or {}).keys())
    return {
        "title": info.get("title") or "",
        "id": info.get("id") or "",
        "manual": manual,
        "auto": auto,
        "raw_info_keys": list(info.keys())[:20],
    }


def download_captions(
    url: str,
    out_dir: Path,
    langs: list[str],
    *,
    cookies: str | Path | None = None,
    prefer_ext: str = "vtt",
) -> dict[str, Path]:
    """
    下载指定语言字幕到 out_dir，返回 {lang: path}。
    不下载视频本体。
    """
    yt_dlp = _require_yt_dlp()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignore_no_formats_error": True,
        "noplaylist": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": list(langs),
        "subtitlesformat": prefer_ext,
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
    }
    if cookies:
        opts["cookiefile"] = str(cookies)

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id") or ""

    found: dict[str, Path] = {}
    for lang in langs:
        # yt-dlp 命名: id.lang.ext
        for ext in (prefer_ext, "vtt", "srt", "ttml", "srv3"):
            p = out_dir / f"{video_id}.{lang}.{ext}"
            if p.is_file():
                found[lang] = p
                break
    # 宽松匹配（lang 含点等情况）
    if len(found) < len(langs):
        for p in out_dir.glob(f"{video_id}.*"):
            name = p.name
            if not name.startswith(video_id + "."):
                continue
            mid = name[len(video_id) + 1 :]
            # mid = lang.ext
            if "." not in mid:
                continue
            lang_part, ext = mid.rsplit(".", 1)
            if ext.lower() not in ("vtt", "srt", "ttml", "srv3", "json3"):
                continue
            for want in langs:
                if lang_part == want and want not in found:
                    found[want] = p
    return found


def dump_lang_list_json(meta: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "id": meta.get("id"),
                "title": meta.get("title"),
                "manual": meta.get("manual"),
                "auto": meta.get("auto"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
