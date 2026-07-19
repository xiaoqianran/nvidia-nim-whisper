"""YouTube 频道：最近 N 个视频元数据 + 字幕/Whisper 流水线。"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.lang_detect import looks_chinese, needs_translate_to_zh
from common.translate_cues import translate_plain_text
from youtube.pipeline import YoutubeProcessResult, process_youtube_url


def _require_yt_dlp():
    try:
        import yt_dlp
    except ImportError as e:
        raise ImportError("需要 yt-dlp") from e
    return yt_dlp


def list_channel_videos(
    channel_url: str,
    *,
    limit: int = 5,
    cookies: str | Path | None = None,
) -> dict[str, Any]:
    """列出频道最近 limit 个视频（flat）。"""
    from youtube.ytdlp_fetch import base_ydl_opts

    yt_dlp = _require_yt_dlp()
    url = channel_url.rstrip("/")
    if not url.endswith("/videos"):
        if "/@" in url or url.rstrip("/").count("/") <= 3:
            url = url + "/videos"

    opts: dict[str, Any] = base_ydl_opts(cookies=cookies)
    opts.update(
        {
            "extract_flat": True,
            "playlistend": max(1, limit),
            "skip_download": True,
            "noplaylist": False,
        }
    )

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = [e for e in (info.get("entries") or []) if e][:limit]
    return {
        "channel": info.get("channel") or info.get("uploader") or info.get("title") or "",
        "channel_url": channel_url,
        "entries": entries,
    }


def fetch_video_meta(
    video_id: str,
    *,
    cookies: str | Path | None = None,
) -> dict[str, Any]:
    from youtube.ytdlp_fetch import base_ydl_opts

    yt_dlp = _require_yt_dlp()
    url = f"https://www.youtube.com/watch?v={video_id}"
    opts: dict[str, Any] = base_ydl_opts(cookies=cookies)
    opts["skip_download"] = True
    with yt_dlp.YoutubeDL(opts) as ydl:
        v = ydl.extract_info(url, download=False)
    return {
        "id": v.get("id") or video_id,
        "title": v.get("title") or "",
        "description": v.get("description") or "",
        "webpage_url": v.get("webpage_url") or url,
        "manual": sorted((v.get("subtitles") or {}).keys()),
        "auto": sorted((v.get("automatic_captions") or {}).keys()),
    }


def video_done_marker(video_dir: Path, video_id: str) -> Path:
    return video_dir / f"{video_id}.done.json"


def is_video_done(video_dir: Path, video_id: str) -> bool:
    """断点：存在 done 标记，或已有非空 zh_txt。"""
    marker = video_done_marker(video_dir, video_id)
    if marker.is_file():
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
            if data.get("status") == "ok":
                return True
        except Exception:
            pass
    zh = video_dir / f"{video_id}.zh.txt"
    if zh.is_file() and zh.stat().st_size > 10:
        return True
    return False


def mark_video_done(
    video_dir: Path,
    video_id: str,
    *,
    mode: str,
    outputs: dict[str, str],
    error: str | None = None,
) -> None:
    payload = {
        "video_id": video_id,
        "status": "ok" if not error else "error",
        "mode": mode,
        "outputs": outputs,
        "error": error,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    video_done_marker(video_dir, video_id).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


@dataclass
class ChannelVideoResult:
    index: int
    video_id: str
    url: str
    title: str
    title_zh: str
    description: str
    description_zh: str
    transcript_mode: str
    chosen_lang: str | None
    outputs: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    skipped: bool = False


def _maybe_translate_field(text: str, translator: Any, label: str, _log) -> str:
    if not text:
        return ""
    if not needs_translate_to_zh(text):
        _log(f"{label}已是中文，跳过翻译")
        return text
    if translator is None:
        return ""
    _log(f"翻译{label}…")
    try:
        if len(text) < 2500:
            return translate_plain_text(text, translator)
        parts, buf, n = [], [], 0
        for para in text.replace("\r", "").split("\n"):
            if n + len(para) > 1800 and buf:
                parts.append(translate_plain_text("\n".join(buf), translator))
                buf, n = [para], len(para)
            else:
                buf.append(para)
                n += len(para) + 1
        if buf:
            parts.append(translate_plain_text("\n".join(buf), translator))
        return "\n".join(parts)
    except Exception as ex:
        _log(f"{label}翻译失败: {ex}")
        return ""


def process_channel(
    channel_url: str,
    out_dir: Path,
    *,
    limit: int = 5,
    cookies: str | Path | None = None,
    translator: Any = None,
    riva_client: Any = None,
    asr_pool: Any = None,
    translate_workers: int = 4,
    whisper_workers: int = 8,
    fallback_audio: bool = True,
    resume: bool = True,
    quiet: bool = False,
    log: Callable[[str], None] | None = None,
) -> list[ChannelVideoResult]:
    def _log(msg: str) -> None:
        if log:
            log(msg)
        elif not quiet:
            print(msg, flush=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    listing = list_channel_videos(channel_url, limit=limit, cookies=cookies)
    _log(f"频道: {listing['channel']} | 最近 {len(listing['entries'])} 个视频")
    if resume:
        _log("断点续跑: 已成功 video_id 将跳过")

    results: list[ChannelVideoResult] = []
    for i, e in enumerate(listing["entries"], 1):
        vid = e.get("id") or ""
        url = e.get("webpage_url") or f"https://www.youtube.com/watch?v={vid}"
        if not str(url).startswith("http"):
            url = f"https://www.youtube.com/watch?v={vid}"

        _log(f"\n======== [{i}/{limit}] {vid} ========")
        # 短目录名：仅 video_id
        video_out = out_dir / vid
        video_out.mkdir(parents=True, exist_ok=True)

        if resume and is_video_done(video_out, vid):
            _log(f"跳过（已完成）: {vid}")
            # 尽量从 done / 旧 md 恢复摘要
            marker = video_done_marker(video_out, vid)
            mode, outs, err = "skipped", {}, None
            if marker.is_file():
                try:
                    d = json.loads(marker.read_text(encoding="utf-8"))
                    mode = d.get("mode") or "skipped"
                    outs = d.get("outputs") or {}
                    err = d.get("error")
                except Exception:
                    pass
            zh_txt = video_out / f"{vid}.zh.txt"
            if zh_txt.is_file():
                outs.setdefault("zh_txt", str(zh_txt))
            results.append(
                ChannelVideoResult(
                    index=i,
                    video_id=vid,
                    url=url,
                    title=e.get("title") or "",
                    title_zh="",
                    description="",
                    description_zh="",
                    transcript_mode=mode,
                    chosen_lang=None,
                    outputs=outs,
                    error=err,
                    skipped=True,
                )
            )
            continue

        try:
            meta = fetch_video_meta(vid, cookies=cookies)
        except Exception as ex:
            results.append(
                ChannelVideoResult(
                    index=i,
                    video_id=vid,
                    url=url,
                    title=e.get("title") or "",
                    title_zh="",
                    description="",
                    description_zh="",
                    transcript_mode="none",
                    chosen_lang=None,
                    error=f"元数据失败: {ex}",
                )
            )
            continue

        title = meta["title"]
        desc = meta["description"]
        url = meta["webpage_url"]

        title_zh = _maybe_translate_field(title, translator, "标题", _log)
        desc_zh = _maybe_translate_field(desc, translator, "简介", _log)

        prefetched = {
            "id": meta["id"],
            "title": title,
            "description": desc,
            "manual": meta["manual"],
            "auto": meta["auto"],
        }
        proc: YoutubeProcessResult = process_youtube_url(
            url,
            video_out,
            cookies=cookies,
            translator=translator,
            riva_client=riva_client,
            asr_pool=asr_pool,
            translate_workers=translate_workers,
            whisper_workers=whisper_workers,
            fallback_audio=fallback_audio,
            quiet=quiet,
            log=_log,
            prefetched_meta=prefetched,
        )

        outs = {k: str(v) for k, v in proc.outputs.items()}

        # 不再写单独的「简介 md」；全部汇总进 README 全文

        if not proc.error and (outs.get("zh_txt") or outs.get("en_txt")):
            mark_video_done(video_out, vid, mode=proc.mode, outputs=outs)
        elif proc.error:
            mark_video_done(video_out, vid, mode=proc.mode or "none", outputs=outs, error=proc.error)

        results.append(
            ChannelVideoResult(
                index=i,
                video_id=vid,
                url=url,
                title=title,
                title_zh=title_zh,
                description=desc,
                description_zh=desc_zh,
                transcript_mode=proc.mode,
                chosen_lang=proc.chosen_lang,
                outputs=outs,
                error=proc.error,
            )
        )

    write_channel_readme(
        out_dir,
        channel_name=listing["channel"] or channel_url,
        channel_url=channel_url,
        limit=limit,
        results=results,
    )
    json_path = out_dir / "latest.json"
    payload = [
        {
            "index": r.index,
            "video_id": r.video_id,
            "url": r.url,
            "title": r.title,
            "title_zh": r.title_zh,
            "description": r.description,
            "description_zh": r.description_zh,
            "transcript_mode": r.transcript_mode,
            "chosen_lang": r.chosen_lang,
            "outputs": r.outputs,
            "error": r.error,
            "skipped": r.skipped,
        }
        for r in results
    ]
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _log(f"\n完成。README（全文）: {out_dir / 'README.md'}")
    return results


def write_channel_readme(
    out_dir: Path,
    *,
    channel_name: str,
    channel_url: str,
    limit: int,
    results: list[ChannelVideoResult],
) -> Path:
    """
    单一 README：标题/简介原文与简体均为**全文**，不再另写每条简介 md。
    （字幕/转写产物若有，只列路径，不在此展开。）
    """
    lines = [
        f"# {channel_name} 最近 {limit} 个视频\n\n",
        f"更新时间: {datetime.now(timezone.utc).isoformat()}\n\n",
        f"频道: {channel_url}\n\n",
        "> 标题与简介均为**完整全文**（不再单独生成简介 md）。\n\n",
        "---\n",
    ]
    for r in results:
        lines.append(f"\n## {r.index}. {r.title_zh or r.title}\n\n")
        lines.append(f"- **链接**: {r.url}\n")
        lines.append(f"- **video_id**: `{r.video_id}`\n")
        if r.transcript_mode:
            lines.append(f"- **转写模式**: `{r.transcript_mode}`")
            if r.chosen_lang:
                lines.append(f" / `{r.chosen_lang}`")
            if r.skipped:
                lines.append("（跳过/续跑）")
            lines.append("\n")
        if r.error:
            lines.append(f"- **错误**: {r.error}\n")
        lines.append(f"\n### 标题（原文）\n\n{r.title or ''}\n\n")
        lines.append(f"### 标题（简体中文）\n\n{r.title_zh or ''}\n\n")
        lines.append(f"### 简介（原文）\n\n{r.description or ''}\n\n")
        lines.append(f"### 简介（简体中文）\n\n{r.description_zh or ''}\n\n")
        lines.append("---\n")

    path = Path(out_dir) / "README.md"
    path.write_text("".join(lines), encoding="utf-8")
    return path
