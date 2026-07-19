"""字幕语言选择、VTT/SRT 解析与纯文本转换（无网络依赖）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# 简体中文（手动轨优先）
ZH_HANS_MANUAL = ("zh-Hans", "zh-CN", "zh", "cmn-Hans", "zh-Hans-CN")
# 自动轨中带简体标记
ZH_HANS_AUTO_PREFIXES = ("zh-Hans", "zh-CN", "cmn-Hans")

EN_MANUAL = ("en", "en-US", "en-GB", "en-orig")
EN_AUTO_PREFIXES = ("en",)


@dataclass(frozen=True)
class CaptionTrack:
    lang: str
    kind: str  # "manual" | "auto"
    ext: str = "vtt"


@dataclass
class Cue:
    start: float  # seconds
    end: float
    text: str


def is_traditional_chinese_lang(code: str) -> bool:
    c = (code or "").replace("_", "-")
    low = c.lower()
    if "hant" in low:
        return True
    if low in ("zh-tw", "zh-hk", "zh-mo", "zh-hant"):
        return True
    if low.startswith("zh-tw") or low.startswith("zh-hk"):
        return True
    return False


def is_simplified_chinese_lang(code: str) -> bool:
    """是否为简体中文相关语言码（排除繁体）。"""
    if not code or is_traditional_chinese_lang(code):
        return False
    c = code.replace("_", "-")
    low = c.lower()
    if low in ("zh", "zh-cn", "zh-sg", "cmn", "cmn-hans"):
        return True
    if "hans" in low:
        return True
    if low.startswith("zh-cn") or low.startswith("zh-sg"):
        return True
    # yt-dlp auto: zh-Hans-en
    if low.startswith("zh-hans"):
        return True
    return False


def is_english_lang(code: str) -> bool:
    if not code:
        return False
    c = code.replace("_", "-")
    low = c.lower()
    if low == "en" or low.startswith("en-") or low.startswith("en."):
        return True
    return False


def _rank_zh(track: CaptionTrack) -> tuple:
    """越小越优先。"""
    manual = 0 if track.kind == "manual" else 1
    # 精确匹配优先
    try:
        exact = list(ZH_HANS_MANUAL).index(track.lang)
    except ValueError:
        exact = 50
        if track.lang.lower().startswith("zh-hans"):
            exact = 10
        elif track.lang.lower() in ("zh-cn", "zh"):
            exact = 5
    return (manual, exact, track.lang)


def _rank_en(track: CaptionTrack) -> tuple:
    manual = 0 if track.kind == "manual" else 1
    try:
        exact = list(EN_MANUAL).index(track.lang)
    except ValueError:
        exact = 50
        if track.lang.lower() == "en" or track.lang.lower().startswith("en-"):
            exact = 20
    return (manual, exact, track.lang)


def pick_caption_track(
    manual_langs: Iterable[str],
    auto_langs: Iterable[str],
) -> tuple[str, CaptionTrack | None]:
    """
    选择字幕轨。

    返回 (path_mode, track):
      path_mode: "zh-Hans" | "en" | "none"
      track: 选中的语言轨，或 None
    """
    tracks: list[CaptionTrack] = []
    for lang in manual_langs or []:
        tracks.append(CaptionTrack(lang=lang, kind="manual"))
    for lang in auto_langs or []:
        tracks.append(CaptionTrack(lang=lang, kind="auto"))

    zh = [t for t in tracks if is_simplified_chinese_lang(t.lang)]
    if zh:
        best = sorted(zh, key=_rank_zh)[0]
        return "zh-Hans", best

    en = [t for t in tracks if is_english_lang(t.lang)]
    if en:
        best = sorted(en, key=_rank_en)[0]
        return "en", best

    return "none", None


_TS_VTT = re.compile(
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})\.(?P<ms>\d{3})\s*-->\s*"
    r"(?P<h2>\d{2}):(?P<m2>\d{2}):(?P<s2>\d{2})\.(?P<ms2>\d{3})"
)
_TS_SRT = re.compile(
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{3})\s*-->\s*"
    r"(?P<h2>\d{2}):(?P<m2>\d{2}):(?P<s2>\d{2}),(?P<ms2>\d{3})"
)
_TAG = re.compile(r"<[^>]+>")
_NOTE = re.compile(r"^(NOTE|STYLE|REGION)\b", re.I)


def _ts_to_sec(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _clean_text_line(line: str) -> str:
    line = _TAG.sub("", line)
    line = line.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    line = line.strip()
    return line


def parse_vtt(content: str) -> list[Cue]:
    cues: list[Cue] = []
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.upper().startswith("WEBVTT") or _NOTE.match(line):
            i += 1
            continue
        m = _TS_VTT.search(line)
        if not m:
            i += 1
            continue
        start = _ts_to_sec(m["h"], m["m"], m["s"], m["ms"])
        end = _ts_to_sec(m["h2"], m["m2"], m["s2"], m["ms2"])
        i += 1
        text_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            if _TS_VTT.search(lines[i]):
                break
            text_lines.append(_clean_text_line(lines[i]))
            i += 1
        text = " ".join(t for t in text_lines if t).strip()
        if text:
            cues.append(Cue(start=start, end=end, text=text))
    return cues


def parse_srt(content: str) -> list[Cue]:
    cues: list[Cue] = []
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n").replace("\r", "\n").strip())
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        if not lines:
            continue
        # optional index line
        idx = 0
        if lines[0].strip().isdigit():
            idx = 1
        if idx >= len(lines):
            continue
        m = _TS_SRT.search(lines[idx])
        if not m:
            # try vtt style in srt
            m = _TS_VTT.search(lines[idx])
            if not m:
                continue
            start = _ts_to_sec(m["h"], m["m"], m["s"], m["ms"])
            end = _ts_to_sec(m["h2"], m["m2"], m["s2"], m["ms2"])
        else:
            start = _ts_to_sec(m["h"], m["m"], m["s"], m["ms"])
            end = _ts_to_sec(m["h2"], m["m2"], m["s2"], m["ms2"])
        text = " ".join(_clean_text_line(x) for x in lines[idx + 1 :]).strip()
        if text:
            cues.append(Cue(start=start, end=end, text=text))
    return cues


def parse_subtitle_file(path: Path | str) -> list[Cue]:
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="replace")
    suffix = p.suffix.lower()
    if suffix == ".srt":
        return parse_srt(raw)
    # default vtt / srv3 often saved as vtt by yt-dlp
    if "-->" in raw and "," in raw.split("-->", 1)[0][-10:]:
        # heuristic: srt uses comma
        try:
            return parse_srt(raw)
        except Exception:
            pass
    return parse_vtt(raw)


# 时间轴 / 序号行（写入 .txt 时必须剔除）
_TS_LINE = re.compile(
    r"^\d{1,2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[.,]\d{3}"
)
_INDEX_ONLY = re.compile(r"^\d+$")


def _is_timestamp_or_index_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if _TS_LINE.match(s) or "-->" in s:
        return True
    if _INDEX_ONLY.match(s):
        return True
    if s.upper().startswith("WEBVTT"):
        return True
    return False


def cues_to_plain_text(cues: list[Cue]) -> str:
    """
    合并为纯文本（仅台词，无时间轴、无序号）。
    相邻重复行去重（自动字幕常见）。
    """
    parts: list[str] = []
    prev = ""
    for c in cues:
        t = c.text.strip()
        if not t or _is_timestamp_or_index_line(t):
            continue
        # 若 cue 文本里误带多行，只保留非时间轴行
        cleaned_lines = [
            ln.strip()
            for ln in t.replace("\r", "\n").split("\n")
            if ln.strip() and not _is_timestamp_or_index_line(ln)
        ]
        t = " ".join(cleaned_lines).strip()
        if not t:
            continue
        if t == prev:
            continue
        # 滚动字幕：若新行包含旧行前缀则替换
        if prev and t.startswith(prev):
            parts[-1] = t
            prev = t
            continue
        if prev and prev.startswith(t):
            continue
        parts.append(t)
        prev = t
    return "\n".join(parts).strip() + ("\n" if parts else "")


def strip_to_plain_text(content: str) -> str:
    """
    将任意 SRT/VTT/混杂文本收成纯台词（保险层，供写 .txt 前调用）。
    """
    if not content or not content.strip():
        return ""
    # 若看起来像字幕文件，先解析
    if "-->" in content or content.lstrip().upper().startswith("WEBVTT"):
        try:
            if re.search(r"\d{2}:\d{2}:\d{2},\d{3}\s*-->", content):
                cues = parse_srt(content)
            else:
                cues = parse_vtt(content)
            if cues:
                return cues_to_plain_text(cues)
        except Exception:
            pass
    lines_out: list[str] = []
    for ln in content.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if _is_timestamp_or_index_line(ln):
            continue
        t = _clean_text_line(ln)
        if t:
            lines_out.append(t)
    return "\n".join(lines_out).strip() + ("\n" if lines_out else "")


def _fmt_srt_ts(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    whole = int(s)
    ms = int(round((s - whole) * 1000))
    if ms == 1000:
        whole += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{whole:02d},{ms:03d}"


def cues_to_srt(cues: list[Cue]) -> str:
    lines: list[str] = []
    for i, c in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{_fmt_srt_ts(c.start)} --> {_fmt_srt_ts(c.end)}")
        lines.append(c.text.strip())
        lines.append("")
    return "\n".join(lines)


def safe_stem(title: str, video_id: str) -> str:
    base = (title or video_id or "youtube").strip()
    base = re.sub(r"[\\/:*?\"<>|]+", "_", base)
    base = re.sub(r"\s+", " ", base).strip()[:80]
    if not base:
        base = video_id or "youtube"
    return f"{base}_{video_id}" if video_id and video_id not in base else base
