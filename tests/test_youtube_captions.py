"""YouTube 字幕纯函数单元测试（无网络）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from youtube.captions import (
    cues_to_plain_text,
    cues_to_srt,
    is_english_lang,
    is_simplified_chinese_lang,
    parse_subtitle_file,
    pick_caption_track,
    strip_to_plain_text,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_lang_detect_simplified_vs_traditional():
    assert is_simplified_chinese_lang("zh-Hans")
    assert is_simplified_chinese_lang("zh-CN")
    assert is_simplified_chinese_lang("zh-Hans-en")
    assert is_simplified_chinese_lang("zh")
    assert not is_simplified_chinese_lang("zh-Hant")
    assert not is_simplified_chinese_lang("zh-TW")
    assert not is_simplified_chinese_lang("zh-Hant-en")
    assert is_english_lang("en")
    assert is_english_lang("en-US")
    assert is_english_lang("en-en")
    assert not is_english_lang("fr")


def test_pick_prefers_simplified_over_english():
    mode, track = pick_caption_track(
        manual_langs=["en", "zh-Hans", "de"],
        auto_langs=["en-en", "zh-Hans-en"],
    )
    assert mode == "zh-Hans"
    assert track is not None
    assert track.lang == "zh-Hans"
    assert track.kind == "manual"


def test_pick_falls_back_to_english_auto():
    mode, track = pick_caption_track(
        manual_langs=["de", "fr"],
        auto_langs=["en-en", "zh-Hant-en"],
    )
    assert mode == "en"
    assert track is not None
    assert is_english_lang(track.lang)


def test_pick_none_when_only_traditional():
    mode, track = pick_caption_track(
        manual_langs=["zh-Hant", "zh-TW"],
        auto_langs=["zh-Hant-en"],
    )
    assert mode == "none"
    assert track is None


def test_parse_vtt_and_plain_text():
    path = FIXTURES / "sample_en.vtt"
    cues = parse_subtitle_file(path)
    assert len(cues) == 2
    assert cues[0].text == "Hello world."
    assert "test subtitle" in cues[1].text
    plain = cues_to_plain_text(cues)
    assert "Hello world." in plain
    assert "<c>" not in plain
    assert "-->" not in plain
    assert "00:00:00" not in plain
    srt = cues_to_srt(cues)
    assert "-->" in srt
    assert "Hello world." in srt


def test_txt_never_contains_timestamps():
    """.txt 只要文字：不得出现时间轴行。"""
    raw = (FIXTURES / "sample_zh.srt").read_text(encoding="utf-8")
    assert "-->" in raw  # fixture 本身是 srt
    plain = strip_to_plain_text(raw)
    assert "-->" not in plain
    assert "00:00:00" not in plain
    assert "你好世界" in plain
    assert not any(line.strip().isdigit() for line in plain.splitlines() if line.strip())


def test_parse_srt_zh_fixture():
    path = FIXTURES / "sample_zh.srt"
    cues = parse_subtitle_file(path)
    assert len(cues) == 2
    plain = cues_to_plain_text(cues)
    assert "你好世界" in plain
    assert "测试字幕" in plain


def test_translate_path_invokes_existing_translator(monkeypatch):
    """确保 EN 路径调用现有 translate_segments，而非自造翻译逻辑。"""
    from common.translate_cues import translate_cues
    from youtube.captions import Cue

    calls: list = []

    class FakeTranslator:
        def translate_segments(self, segs, text_key="text", out_key="text_zh", workers=4, quiet=False):
            calls.append(list(segs))
            out = []
            for s in segs:
                d = dict(s)
                d[out_key] = f"译:{s[text_key]}"
                out.append(d)
            return out

    cues = [Cue(0.0, 1.0, "Hello"), Cue(1.0, 2.0, "World")]
    zh = translate_cues(
        cues,
        FakeTranslator(),
        workers=1,
        quiet=True,
        cue_factory=lambda s, e, t: Cue(s, e, t),
    )
    assert len(calls) == 1
    assert calls[0][0]["text"] == "Hello"
    assert zh[0].text == "译:Hello"
    assert zh[1].text == "译:World"


def test_segments_to_cues_estimated():
    from youtube.audio_whisper import segments_to_cues

    segs = [{"text": "Hello"}, {"text": "World"}]
    cues = segments_to_cues(segs, duration=10.0)
    assert len(cues) == 2
    assert cues[0].text == "Hello"
    assert cues[-1].end <= 10.0 + 1e-6
