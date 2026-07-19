#!/usr/bin/env python3
"""
使用 NVIDIA build.nvidia.com 托管的 OpenAI Whisper Large V3 转录音视频。

依赖:
  - ffmpeg / ffprobe
  - nvidia-riva-client（见 requirements.txt）

API Key（任选其一，勿提交到 Git）:
  export NVIDIA_API_KEY='nvapi-...'
  或在项目根目录放置 .env（见 .env.example）
  或: python transcribe_whisper_nvidia.py media.mp4 --api-key nvapi-...

用法:
  python transcribe_whisper_nvidia.py video.mp4
  python transcribe_whisper_nvidia.py audio.wav -o out_dir --language en-US
  ./transcribe.sh video.mp4 --keep-wav --no-srt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


# build.nvidia.com / openai/whisper-large-v3 的 NVCF function-id
DEFAULT_FUNCTION_ID = "b702f636-f60c-4a3d-a6f4-f3568c13bd7d"
DEFAULT_SERVER = "grpc.nvcf.nvidia.com:443"
DEFAULT_SAMPLE_RATE = 16000

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".flv", ".ts", ".mpeg", ".mpg"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".wma"}


def die(msg: str, code: int = 1) -> None:
    print(f"错误: {msg}", file=sys.stderr)
    raise SystemExit(code)


def load_dotenv(paths: list[Path]) -> None:
    """轻量加载 .env：不覆盖已有环境变量。无需 python-dotenv。"""
    for path in paths:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            # 已存在的环境变量优先（export 覆盖 .env）
            if key not in os.environ or os.environ.get(key) == "":
                os.environ[key] = value


def require_cmd(name: str) -> str:
    path = shutil.which(name)
    if not path:
        die(f"未找到命令 `{name}`，请先安装。")
    return path


def import_riva():
    try:
        import riva.client  # type: ignore

        return riva.client
    except ImportError:
        die(
            "未安装 nvidia-riva-client。\n"
            "  推荐: python3 -m venv .venv && source .venv/bin/activate\n"
            "        pip install -r requirements.txt\n"
            "  或:   uv venv && uv pip install -r requirements.txt"
        )


def probe_duration(path: Path) -> float | None:
    ffprobe = require_cmd("ffprobe")
    try:
        out = subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            text=True,
        ).strip()
        return float(out) if out else None
    except (subprocess.CalledProcessError, ValueError):
        return None


def extract_wav(input_path: Path, wav_path: Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
    """用 ffmpeg 抽取/转码为 16kHz 单声道 PCM WAV（Whisper/Riva 推荐格式）。"""
    ffmpeg = require_cmd("ffmpeg")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(wav_path),
    ]
    print(f"正在提取音频: {input_path.name} -> {wav_path.name}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        die(f"ffmpeg 失败:\n{e.stderr[-2000:] if e.stderr else e}")


def ensure_wav(input_path: Path, work_dir: Path, sample_rate: int, keep_wav: bool) -> Path:
    """转码为 16kHz mono WAV，返回路径。"""
    suffix = input_path.suffix.lower()
    if suffix not in VIDEO_EXTS | AUDIO_EXTS and suffix != ".wav":
        print(f"警告: 未识别扩展名 {suffix}，仍尝试用 ffmpeg 处理。", file=sys.stderr)

    wav_path = work_dir / f"{input_path.stem}_16k_mono.wav"
    extract_wav(input_path, wav_path, sample_rate)
    return wav_path


def build_asr_service(riva_client, api_key: str, server: str, function_id: str, max_mb: int):
    max_len = max_mb * 1024 * 1024
    auth = riva_client.Auth(
        use_ssl=True,
        uri=server,
        metadata_args=[
            ("function-id", function_id),
            ("authorization", f"Bearer {api_key}"),
        ],
        options=[
            ("grpc.max_receive_message_length", max_len),
            ("grpc.max_send_message_length", max_len),
        ],
    )
    return riva_client.ASRService(auth)


def offline_transcribe(
    riva_client,
    asr,
    audio: bytes,
    language_code: str,
    sample_rate: int,
    word_offsets: bool,
) -> Any:
    config = riva_client.RecognitionConfig(
        language_code=language_code,
        max_alternatives=1,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=word_offsets,
        audio_channel_count=1,
        sample_rate_hertz=sample_rate,
        encoding=riva_client.AudioEncoding.LINEAR_PCM,
    )
    return asr.offline_recognize(audio, config)


def parse_response(resp) -> tuple[str, list[dict[str, Any]]]:
    segments: list[dict[str, Any]] = []
    full_parts: list[str] = []

    for result in resp.results:
        for alt in result.alternatives:
            text = (alt.transcript or "").strip()
            if not text:
                continue
            full_parts.append(text)
            words = []
            for w in getattr(alt, "words", []) or []:
                words.append(
                    {
                        "word": w.word,
                        "start": getattr(w, "start_time", None),
                        "end": getattr(w, "end_time", None),
                        "confidence": getattr(w, "confidence", None),
                    }
                )
            start = words[0]["start"] if words else None
            end = words[-1]["end"] if words else None
            segments.append(
                {
                    "text": text,
                    "confidence": getattr(alt, "confidence", None),
                    "start": start,
                    "end": end,
                    "words": words,
                }
            )

    full_text = " ".join(full_parts)
    return full_text, segments


def max_word_time(segments: list[dict[str, Any]]) -> float:
    m = 0.0
    for seg in segments:
        for w in seg.get("words") or []:
            for k in ("start", "end"):
                v = w.get(k)
                if v is not None:
                    try:
                        m = max(m, float(v))
                    except (TypeError, ValueError):
                        pass
    return m


def estimate_segment_times(segments: list[dict[str, Any]], duration: float | None) -> list[dict[str, Any]]:
    """API 未返回词级时间戳时，按字符比例估算片段起止时间。"""
    if not segments:
        return segments
    if max_word_time(segments) > 0:
        return segments
    if not duration or duration <= 0:
        return segments

    total_chars = sum(len(s.get("text") or "") for s in segments) or 1
    t = 0.0
    for s in segments:
        n = len(s.get("text") or "")
        dur = duration * (n / total_chars)
        s["start"] = round(t, 3)
        s["end"] = round(min(duration, t + dur), 3)
        s["timing"] = "estimated"
        t = s["end"]
    return segments


def fmt_srt_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    sec = seconds % 60
    whole = int(sec)
    ms = int(round((sec - whole) * 1000))
    if ms == 1000:
        whole += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{whole:02d},{ms:03d}"


def write_srt(
    path: Path,
    segments: list[dict[str, Any]],
    duration: float | None,
    max_line_chars: int = 72,
) -> int:
    """写出 SRT。优先用词级时间；否则用估算的片段时间并按长度切行。"""
    has_words = max_word_time(segments) > 0
    scale = 1.0
    if has_words:
        if duration and max_word_time(segments) > duration * 5:
            scale = 0.001

    cues: list[tuple[float, float, str]] = []

    if has_words:
        for seg in segments:
            words = seg.get("words") or []
            if not words:
                continue
            chunk: list[dict] = []
            chunk_start = None
            for w in words:
                if not chunk:
                    chunk_start = float(w["start"] or 0) * scale
                chunk.append(w)
                end = float(w["end"] or 0) * scale
                text_len = len(" ".join(x["word"] for x in chunk))
                if len(chunk) >= 12 or (end - chunk_start) >= 4.0 or text_len >= max_line_chars:
                    text = " ".join(x["word"] for x in chunk)
                    cues.append((chunk_start, end, text))
                    chunk = []
                    chunk_start = None
            if chunk and chunk_start is not None:
                text = " ".join(x["word"] for x in chunk)
                end = float(chunk[-1]["end"] or 0) * scale
                cues.append((chunk_start, end, text))
    else:
        segs = estimate_segment_times([dict(s) for s in segments], duration)
        for s in segs:
            text = s.get("text") or ""
            start = float(s.get("start") or 0)
            end = float(s.get("end") or start)
            words = text.split()
            chunks: list[str] = []
            buf: list[str] = []
            for w in words:
                buf.append(w)
                if len(" ".join(buf)) >= max_line_chars:
                    chunks.append(" ".join(buf))
                    buf = []
            if buf:
                chunks.append(" ".join(buf))
            if not chunks:
                continue
            total = sum(len(c) for c in chunks) or 1
            t = start
            for c in chunks:
                sub = (end - start) * (len(c) / total)
                te = min(end, t + sub)
                cues.append((t, te, c))
                t = te

    with path.open("w", encoding="utf-8") as f:
        for i, (a, b, text) in enumerate(cues, 1):
            f.write(f"{i}\n{fmt_srt_ts(a)} --> {fmt_srt_ts(b)}\n{text}\n\n")
    return len(cues)


def default_out_stem(input_path: Path) -> str:
    return input_path.stem.replace(" ", "_")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NVIDIA Whisper Large V3 音视频转录（build.nvidia.com NIM）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", type=Path, help="输入音视频文件路径")
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录（默认与输入文件同目录）",
    )
    p.add_argument(
        "--stem",
        default=None,
        help="输出文件名前缀（默认取输入文件名，空格变下划线）",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="NVIDIA API Key；默认读 NVIDIA_API_KEY / NGC_API_KEY / .env",
    )
    p.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="指定 .env 路径（默认自动查找当前目录与脚本目录）",
    )
    p.add_argument("--function-id", default=DEFAULT_FUNCTION_ID, help="NVCF function-id")
    p.add_argument("--server", default=DEFAULT_SERVER, help="gRPC 服务地址")
    p.add_argument(
        "--language",
        default="en-US",
        help="语言代码，如 en-US / zh-CN / multi",
    )
    p.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="转码采样率 Hz")
    p.add_argument(
        "--max-message-mb",
        type=int,
        default=128,
        help="gRPC 单条消息上限（MB）",
    )
    p.add_argument("--keep-wav", action="store_true", help="保留中间 WAV 文件")
    p.add_argument("--no-srt", action="store_true", help="不生成 SRT 字幕")
    p.add_argument("--no-json", action="store_true", help="不生成 JSON")
    p.add_argument("--no-txt", action="store_true", help="不生成纯文本")
    p.add_argument(
        "--word-offsets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="请求词级时间戳（API 可能不返回）",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="减少日志")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    if args.env_file:
        load_dotenv([args.env_file.expanduser().resolve()])
    else:
        load_dotenv([cwd / ".env", script_dir / ".env"])

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        die(f"输入文件不存在: {input_path}")

    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_API_KEY")
    if not api_key:
        die(
            "未找到 API Key。请任选其一:\n"
            "  export NVIDIA_API_KEY='nvapi-...'\n"
            "  在项目根目录创建 .env（参考 .env.example）\n"
            "  或传入 --api-key"
        )

    out_dir = (args.output_dir or input_path.parent).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or default_out_stem(input_path)

    riva_client = import_riva()
    duration = probe_duration(input_path)

    with tempfile.TemporaryDirectory(prefix="whisper_nv_") as tmp:
        tmp_dir = Path(tmp)
        work_dir = out_dir if args.keep_wav else tmp_dir
        wav_path = ensure_wav(input_path, work_dir, args.sample_rate, keep_wav=args.keep_wav)
        audio = wav_path.read_bytes()
        if not args.quiet:
            print(f"音频大小: {len(audio) / 1e6:.2f} MB")
            if duration:
                print(f"时长约: {duration:.1f} s")

        asr = build_asr_service(
            riva_client,
            api_key=api_key,
            server=args.server,
            function_id=args.function_id,
            max_mb=args.max_message_mb,
        )

        if not args.quiet:
            print("正在调用 NVIDIA Whisper Large V3 …")
        t0 = time.time()
        try:
            resp = offline_transcribe(
                riva_client,
                asr,
                audio,
                language_code=args.language,
                sample_rate=args.sample_rate,
                word_offsets=args.word_offsets,
            )
        except Exception as e:
            die(f"转录失败: {type(e).__name__}: {e}")
        elapsed = time.time() - t0
        if not args.quiet:
            print(f"完成，耗时 {elapsed:.1f} s")

    full_text, segments = parse_response(resp)
    if not full_text:
        die("未得到任何转写文本（空结果）")

    timing_note = None
    if max_word_time(segments) <= 0:
        segments = estimate_segment_times(segments, duration)
        timing_note = "片段 start/end 为按字符比例估算（API 未返回词级时间戳）"

    written: list[Path] = []

    if not args.no_txt:
        txt_path = out_dir / f"{stem}_transcript.txt"
        txt_path.write_text(full_text + "\n", encoding="utf-8")
        written.append(txt_path)

    if not args.no_json:
        json_path = out_dir / f"{stem}_transcript.json"
        payload = {
            "model": "openai/whisper-large-v3",
            "source": str(input_path),
            "language": args.language,
            "duration_sec": duration,
            "timing_note": timing_note,
            "text": full_text,
            "segments": segments,
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(json_path)

    if not args.no_srt:
        srt_path = out_dir / f"{stem}.srt"
        n = write_srt(srt_path, segments, duration)
        written.append(srt_path)
        if not args.quiet:
            print(f"SRT 字幕条数: {n}" + ("（时间戳为估算）" if timing_note else ""))

    if not args.quiet:
        print(f"字数约: {len(full_text.split())} 词 / {len(full_text)} 字符")
        print("已写入:")
        for p in written:
            print(f"  - {p}")
        print("--- 预览（前 400 字）---")
        print(full_text[:400] + ("…" if len(full_text) > 400 else ""))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
