#!/usr/bin/env python3
"""
使用 NVIDIA build.nvidia.com 托管的 OpenAI Whisper Large V3 转录音视频。

单文件分段：将整段音频切成固定时长的 chunk，默认并行调用 API
（滑动窗口限速 40 次/分钟），再按时间偏移合并文本 / JSON / SRT。

可选：OpenAI 兼容 Chat Completions 翻译模块（translate_openai.py），
将转写结果译成中文等目标语言（Whisper 自带 translate 仅能到英文）。

依赖:
  - ffmpeg / ffprobe
  - nvidia-riva-client（见 requirements.txt）

API Key（任选其一，勿提交到 Git）:
  export NVIDIA_API_KEY='nvapi-...'
  翻译另需: export OPENAI_API_KEY=... OPENAI_BASE_URL=... OPENAI_MODEL=...
  或在项目根目录放置 .env（见 .env.example）

用法:
  python transcribe_whisper_nvidia.py audio.mp3
  python transcribe_whisper_nvidia.py audio.mp3 --translate
  python transcribe_whisper_nvidia.py audio.mp3 --translate --to zh-CN
  ./transcribe.sh audio.mp3 --chunk-seconds 45 --translate
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# build.nvidia.com / openai/whisper-large-v3 的 NVCF function-id
DEFAULT_FUNCTION_ID = "b702f636-f60c-4a3d-a6f4-f3568c13bd7d"
DEFAULT_SERVER = "grpc.nvcf.nvidia.com:443"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_SECONDS = 30.0
# NVIDIA API Trial 常见限速：滑动窗口 40 次/分钟
DEFAULT_RATE_LIMIT = 40
DEFAULT_RATE_WINDOW_SEC = 60.0
DEFAULT_WORKERS = 8

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


def ensure_wav(input_path: Path, work_dir: Path, sample_rate: int) -> Path:
    suffix = input_path.suffix.lower()
    if suffix not in VIDEO_EXTS | AUDIO_EXTS and suffix != ".wav":
        print(f"警告: 未识别扩展名 {suffix}，仍尝试用 ffmpeg 处理。", file=sys.stderr)

    wav_path = work_dir / f"{input_path.stem}_16k_mono.wav"
    extract_wav(input_path, wav_path, sample_rate)
    return wav_path


def wav_duration_and_rate(wav_path: Path) -> tuple[float, int, int]:
    """返回 (duration_sec, sample_rate, n_channels)。"""
    with wave.open(str(wav_path), "rb") as w:
        rate = w.getframerate()
        nframes = w.getnframes()
        channels = w.getnchannels()
        duration = nframes / float(rate) if rate else 0.0
        return duration, rate, channels


@dataclass
class AudioChunk:
    index: int
    start_sec: float
    end_sec: float
    path: Path


def split_wav_chunks(
    wav_path: Path,
    chunk_dir: Path,
    chunk_seconds: float,
    overlap_seconds: float,
    sample_rate: int,
) -> list[AudioChunk]:
    """
    将整段 WAV 按时间切成多个 chunk 文件（串行后续调用 API）。

    chunk_seconds <= 0 时表示不切分，整文件作为一个 chunk。
    """
    duration, rate, channels = wav_duration_and_rate(wav_path)
    if rate != sample_rate:
        print(
            f"警告: WAV 采样率 {rate} != 期望 {sample_rate}，仍按文件实际采样率切分。",
            file=sys.stderr,
        )
        sample_rate = rate

    if chunk_seconds <= 0 or duration <= chunk_seconds:
        # 整段作为一个 chunk（可直接复用原 wav，避免多余拷贝）
        return [
            AudioChunk(
                index=0,
                start_sec=0.0,
                end_sec=duration,
                path=wav_path,
            )
        ]

    if overlap_seconds < 0:
        die("--overlap-seconds 不能为负")
    if overlap_seconds >= chunk_seconds:
        die("--overlap-seconds 必须小于 --chunk-seconds")

    step = chunk_seconds - overlap_seconds
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[AudioChunk] = []
    with wave.open(str(wav_path), "rb") as src:
        sampwidth = src.getsampwidth()
        n_channels = src.getnchannels()
        assert n_channels == 1, "期望单声道 WAV"

        idx = 0
        start = 0.0
        while start < duration - 1e-6:
            end = min(duration, start + chunk_seconds)
            start_frame = int(round(start * sample_rate))
            end_frame = int(round(end * sample_rate))
            n_frames = max(0, end_frame - start_frame)
            if n_frames <= 0:
                break

            src.setpos(start_frame)
            frames = src.readframes(n_frames)

            out = chunk_dir / f"chunk_{idx:04d}_{start:.2f}-{end:.2f}.wav"
            with wave.open(str(out), "wb") as dst:
                dst.setnchannels(1)
                dst.setsampwidth(sampwidth)
                dst.setframerate(sample_rate)
                dst.writeframes(frames)

            chunks.append(AudioChunk(index=idx, start_sec=start, end_sec=end, path=out))
            idx += 1

            if end >= duration - 1e-6:
                break
            start += step

    return chunks


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


def shift_segments(segments: list[dict[str, Any]], offset_sec: float) -> list[dict[str, Any]]:
    """把片段内相对时间平移到全局时间轴。"""
    out: list[dict[str, Any]] = []
    for seg in segments:
        s = dict(seg)
        words = []
        for w in seg.get("words") or []:
            nw = dict(w)
            if nw.get("start") is not None:
                try:
                    nw["start"] = float(nw["start"]) + offset_sec
                except (TypeError, ValueError):
                    pass
            if nw.get("end") is not None:
                try:
                    nw["end"] = float(nw["end"]) + offset_sec
                except (TypeError, ValueError):
                    pass
            words.append(nw)
        s["words"] = words
        if s.get("start") is not None:
            try:
                s["start"] = float(s["start"]) + offset_sec
            except (TypeError, ValueError):
                pass
        if s.get("end") is not None:
            try:
                s["end"] = float(s["end"]) + offset_sec
            except (TypeError, ValueError):
                pass
        out.append(s)
    return out


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
        for k in ("start", "end"):
            v = seg.get(k)
            if v is not None:
                try:
                    m = max(m, float(v))
                except (TypeError, ValueError):
                    pass
    return m


def estimate_segment_times(
    segments: list[dict[str, Any]],
    duration: float | None,
    window_start: float = 0.0,
    window_end: float | None = None,
) -> list[dict[str, Any]]:
    """
    API 未返回词级时间戳时，在 [window_start, window_end] 内按字符比例估算。
    单文件分段场景下，每个 chunk 在自己的时间窗内估算，再合并。
    """
    if not segments:
        return segments
    if max_word_time(segments) > 0:
        return segments

    end = window_end if window_end is not None else duration
    if end is None or end <= window_start:
        return segments

    span = end - window_start
    total_chars = sum(len(s.get("text") or "") for s in segments) or 1
    t = window_start
    for s in segments:
        n = len(s.get("text") or "")
        dur = span * (n / total_chars)
        s["start"] = round(t, 3)
        s["end"] = round(min(end, t + dur), 3)
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
    """写出 SRT。优先用词级时间；否则用片段估算时间并按长度切行。"""
    has_words = any((seg.get("words") or []) for seg in segments) and max_word_time(segments) > 0
    scale = 1.0
    if has_words and duration and max_word_time(segments) > duration * 5:
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
        segs = segments
        # 若完全没有时间，整段按总时长估算
        if max_word_time(segs) <= 0:
            segs = estimate_segment_times([dict(s) for s in segments], duration, 0.0, duration)
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
                sub = (end - start) * (len(c) / total) if end > start else 0
                te = min(end, t + sub) if end > start else end
                if te <= t:
                    te = t + 0.01
                cues.append((t, te, c))
                t = te

    with path.open("w", encoding="utf-8") as f:
        for i, (a, b, text) in enumerate(cues, 1):
            f.write(f"{i}\n{fmt_srt_ts(a)} --> {fmt_srt_ts(b)}\n{text}\n\n")
    return len(cues)


def default_out_stem(input_path: Path) -> str:
    return input_path.stem.replace(" ", "_")


class SlidingWindowRateLimiter:
    """滑动窗口限速：window_sec 内最多 max_calls 次。"""

    def __init__(self, max_calls: int, window_sec: float = 60.0) -> None:
        self.max_calls = max_calls
        self.window_sec = window_sec
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self._times and now - self._times[0] >= self.window_sec:
            self._times.popleft()

    def remaining(self) -> int:
        if self.max_calls <= 0:
            return 10**9
        with self._lock:
            self._prune(time.monotonic())
            return max(0, self.max_calls - len(self._times))

    def wait_seconds(self) -> float:
        """若当前无槽位，需等待多久；有槽位则 0。"""
        if self.max_calls <= 0:
            return 0.0
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            if len(self._times) < self.max_calls:
                return 0.0
            return max(0.0, self.window_sec - (now - self._times[0]) + 0.02)

    def try_acquire(self) -> bool:
        """非阻塞：有名额则占用并返回 True。"""
        if self.max_calls <= 0:
            return True
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            if len(self._times) < self.max_calls:
                self._times.append(now)
                return True
            return False

    def acquire(self, quiet: bool = True) -> float:
        """阻塞直到拿到名额，返回等待秒数。"""
        waited = 0.0
        if self.max_calls <= 0:
            return 0.0
        while True:
            if self.try_acquire():
                return waited
            sleep_for = self.wait_seconds()
            if sleep_for <= 0:
                sleep_for = 0.05
            if not quiet:
                print(
                    f"  限速等待 {sleep_for:.1f}s"
                    f"（滑动窗口 {self.max_calls}/{self.window_sec:g}s）…",
                    flush=True,
                )
            time.sleep(sleep_for)
            waited += sleep_for


def mask_api_key(key: str) -> str:
    k = key.strip()
    if len(k) <= 16:
        return k[:4] + "…"
    return f"{k[:12]}…{k[-4:]}"


def parse_api_keys_blob(text: str) -> list[str]:
    """解析逗号/空白/换行分隔的 key 列表。"""
    if not text:
        return []
    parts: list[str] = []
    for line in text.replace(";", "\n").replace(",", "\n").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts.append(line)
    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def load_nvidia_api_keys(
    *,
    cli_key: str | None,
    cli_keys: str | None,
    keys_file: Path | None,
) -> list[str]:
    """
    收集 API Key，优先级合并（去重保序）:
      --api-key / --api-keys / --api-keys-file
      NVIDIA_API_KEY / NVIDIA_API_KEYS / NGC_API_KEY
      NVIDIA_API_KEYS_FILE 指向的文件
    """
    keys: list[str] = []
    if cli_key:
        keys.extend(parse_api_keys_blob(cli_key))
    if cli_keys:
        keys.extend(parse_api_keys_blob(cli_keys))
    if keys_file and keys_file.is_file():
        keys.extend(parse_api_keys_blob(keys_file.read_text(encoding="utf-8")))

    env_file = os.environ.get("NVIDIA_API_KEYS_FILE") or os.environ.get("NVAPI_KEYS_FILE")
    if env_file:
        p = Path(env_file).expanduser()
        if p.is_file():
            keys.extend(parse_api_keys_blob(p.read_text(encoding="utf-8")))

    keys.extend(parse_api_keys_blob(os.environ.get("NVIDIA_API_KEYS") or ""))
    keys.extend(parse_api_keys_blob(os.environ.get("NVAPI_KEYS") or ""))
    single = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_API_KEY") or ""
    if single.strip():
        keys.extend(parse_api_keys_blob(single))

    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


class NvidiaApiKeyPool:
    """
    多 Key 负载均衡 + 每 Key 独立滑动窗口限速。

    有效吞吐 ≈ n_keys × rate_limit / window（例如 6×40 = 240 次/分钟）。
    调度：优先选有剩余配额的 Key（轮询）；全满则等待最先腾出名额的 Key。
    """

    def __init__(
        self,
        keys: list[str],
        *,
        riva_client: Any,
        server: str,
        function_id: str,
        max_mb: int,
        rate_limit: int = DEFAULT_RATE_LIMIT,
        rate_window_sec: float = DEFAULT_RATE_WINDOW_SEC,
    ) -> None:
        if not keys:
            raise ValueError("API Key 池为空")
        self.keys = list(keys)
        self.rate_limit = rate_limit
        self.rate_window_sec = rate_window_sec
        self._riva = riva_client
        self._server = server
        self._function_id = function_id
        self._max_mb = max_mb
        self._limiters = {
            k: SlidingWindowRateLimiter(rate_limit, rate_window_sec) for k in self.keys
        }
        self._services: dict[str, Any] = {}
        self._svc_lock = threading.Lock()
        self._rr = 0
        self._rr_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats: dict[str, int] = {k: 0 for k in self.keys}

    @property
    def size(self) -> int:
        return len(self.keys)

    def effective_rpm(self) -> int:
        if self.rate_limit <= 0:
            return 0
        return self.rate_limit * len(self.keys)

    def _get_service(self, key: str):
        with self._svc_lock:
            svc = self._services.get(key)
            if svc is None:
                svc = build_asr_service(
                    self._riva,
                    api_key=key,
                    server=self._server,
                    function_id=self._function_id,
                    max_mb=self._max_mb,
                )
                self._services[key] = svc
            return svc

    def acquire(self, quiet: bool = True) -> tuple[str, Any]:
        """占用一个 Key 名额，返回 (key, asr_service)。"""
        if self.rate_limit <= 0:
            with self._rr_lock:
                idx = self._rr % len(self.keys)
                self._rr += 1
            key = self.keys[idx]
            with self._stats_lock:
                self._stats[key] = self._stats.get(key, 0) + 1
            return key, self._get_service(key)

        waited_total = 0.0
        while True:
            # 从 rr 起点轮询，优先 try_acquire 成功的 Key
            with self._rr_lock:
                start = self._rr
                self._rr += 1
            n = len(self.keys)
            for off in range(n):
                key = self.keys[(start + off) % n]
                if self._limiters[key].try_acquire():
                    with self._stats_lock:
                        self._stats[key] = self._stats.get(key, 0) + 1
                    return key, self._get_service(key)

            # 全部占满：睡到「最早释放」的那个
            sleep_for = min(self._limiters[k].wait_seconds() for k in self.keys)
            sleep_for = max(0.05, sleep_for)
            if not quiet:
                print(
                    f"  Key 池限速等待 {sleep_for:.1f}s"
                    f"（{self.size} keys × {self.rate_limit}/{self.rate_window_sec:g}s"
                    f" ≈ {self.effective_rpm()}/min）…",
                    flush=True,
                )
            time.sleep(sleep_for)
            waited_total += sleep_for

    def stats_summary(self) -> list[dict[str, Any]]:
        with self._stats_lock:
            return [
                {
                    "key": mask_api_key(k),
                    "requests": self._stats.get(k, 0),
                    "remaining_slots": self._limiters[k].remaining(),
                }
                for k in self.keys
            ]


def _wav_to_pcm(wav_path: Path) -> bytes | None:
    """读取 WAV 的 PCM 帧；失败返回 None。"""
    try:
        with wave.open(str(wav_path), "rb") as w:
            return w.readframes(w.getnframes())
    except wave.Error:
        return None


def _is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    needles = (
        "rate limit",
        "rate_limit",
        "too many requests",
        "resource_exhausted",
        "quota",
        "429",
        "throttl",
    )
    return any(n in msg for n in needles)


def transcribe_one_chunk(
    riva_client,
    chunk: AudioChunk,
    language_code: str,
    sample_rate: int,
    word_offsets: bool,
    pool: NvidiaApiKeyPool,
    quiet: bool = True,
    max_retries: int = 5,
) -> tuple[AudioChunk, str, list[dict[str, Any]], float, str]:
    """转录单个 chunk，返回 (chunk, text, segments_global, elapsed, key_mask)."""
    raw = chunk.path.read_bytes()
    pcm = _wav_to_pcm(chunk.path)
    audio = pcm if pcm is not None else raw

    t0 = time.time()
    last_err: BaseException | None = None
    used_key = ""
    for attempt in range(1, max_retries + 1):
        key, asr = pool.acquire(quiet=quiet)
        used_key = key
        try:
            resp = offline_transcribe(
                riva_client,
                asr,
                audio,
                language_code=language_code,
                sample_rate=sample_rate,
                word_offsets=word_offsets,
            )
            break
        except Exception as e:
            last_err = e
            if attempt >= max_retries or not _is_rate_limit_error(e):
                raise
            backoff = min(30.0, 1.5 * attempt + 0.5)
            if not quiet:
                print(
                    f"  chunk#{chunk.index} key={mask_api_key(key)} 疑似限速，"
                    f"{backoff:.1f}s 后换 Key 重试（{attempt}/{max_retries}）: {e}",
                    flush=True,
                )
            time.sleep(backoff)
    else:
        raise last_err or RuntimeError("转录失败")

    elapsed = time.time() - t0
    text, segs = parse_response(resp)

    if segs and max_word_time(segs) <= 0:
        segs = estimate_segment_times(segs, None, chunk.start_sec, chunk.end_sec)
    else:
        segs = shift_segments(segs, chunk.start_sec)

    for s in segs:
        s["chunk_index"] = chunk.index
        s["chunk_start"] = chunk.start_sec
        s["chunk_end"] = chunk.end_sec

    return chunk, text, segs, elapsed, mask_api_key(used_key)


def run_chunks(
    riva_client,
    pool: NvidiaApiKeyPool,
    chunks: list[AudioChunk],
    language_code: str,
    sample_rate: int,
    word_offsets: bool,
    workers: int,
    quiet: bool,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """
    对 chunk 列表做识别。

    workers=1: 串行
    workers>1: 线程池并行；Key 池负载均衡 + 每 Key 独立限速
    """
    results: dict[int, tuple[str, list[dict[str, Any]], float, str]] = {}
    total = len(chunks)

    def _work(ch: AudioChunk):
        return transcribe_one_chunk(
            riva_client,
            ch,
            language_code,
            sample_rate,
            word_offsets,
            pool=pool,
            quiet=quiet,
        )

    if workers <= 1:
        for i, ch in enumerate(chunks, 1):
            if not quiet:
                print(
                    f"[{i}/{total}] chunk#{ch.index} "
                    f"{ch.start_sec:.1f}s–{ch.end_sec:.1f}s …",
                    flush=True,
                )
            try:
                chunk, text, segs, elapsed, key_m = _work(ch)
            except Exception as e:
                die(f"chunk#{ch.index} 转录失败: {type(e).__name__}: {e}")
            results[chunk.index] = (text, segs, elapsed, key_m)
            if not quiet:
                preview = (text[:80] + "…") if len(text) > 80 else text
                print(f"    完成 {elapsed:.1f}s key={key_m} | {preview}", flush=True)
    else:
        if not quiet:
            rpm = pool.effective_rpm()
            rl = (
                f"{pool.size} keys × {pool.rate_limit}/{pool.rate_window_sec:g}s"
                f" ≈ {rpm}/min"
                if pool.rate_limit > 0
                else f"{pool.size} keys 无限速"
            )
            print(f"并行 workers={workers}，共 {total} 个 chunk，Key 池 {rl} …", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_work, ch): ch for ch in chunks}
            done = 0
            for fut in as_completed(futs):
                ch = futs[fut]
                try:
                    chunk, text, segs, elapsed, key_m = fut.result()
                except Exception as e:
                    die(f"chunk#{ch.index} 转录失败: {type(e).__name__}: {e}")
                results[chunk.index] = (text, segs, elapsed, key_m)
                done += 1
                if not quiet:
                    preview = (text[:80] + "…") if len(text) > 80 else text
                    print(
                        f"[{done}/{total}] chunk#{chunk.index} "
                        f"{chunk.start_sec:.1f}s–{chunk.end_sec:.1f}s "
                        f"完成 {elapsed:.1f}s key={key_m} | {preview}",
                        flush=True,
                    )

    texts: list[str] = []
    all_segments: list[dict[str, Any]] = []
    chunk_meta: list[dict[str, Any]] = []
    for ch in sorted(chunks, key=lambda c: c.index):
        text, segs, elapsed, key_m = results[ch.index]
        if text:
            texts.append(text)
        all_segments.extend(segs)
        chunk_meta.append(
            {
                "index": ch.index,
                "start_sec": ch.start_sec,
                "end_sec": ch.end_sec,
                "path": str(ch.path.name),
                "elapsed_sec": round(elapsed, 3),
                "chars": len(text),
                "api_key": key_m,
            }
        )

    full_text = " ".join(texts)
    return full_text, all_segments, chunk_meta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NVIDIA Whisper Large V3 音视频转录（单文件分段 + build.nvidia.com NIM）",
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
        help="单个 NVIDIA API Key（可与 --api-keys 合并）",
    )
    p.add_argument(
        "--api-keys",
        default=None,
        help="多个 Key，逗号/换行分隔；用于负载均衡突破单 Key 40/min",
    )
    p.add_argument(
        "--api-keys-file",
        type=Path,
        default=None,
        help="Key 列表文件（每行一个）；也可用环境变量 NVIDIA_API_KEYS_FILE",
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
    p.add_argument(
        "--chunk-seconds",
        type=float,
        default=DEFAULT_CHUNK_SECONDS,
        help="单文件分段时长（秒）。<=0 表示不切分、整段一次请求",
    )
    p.add_argument(
        "--overlap-seconds",
        type=float,
        default=0.0,
        help="相邻 chunk 重叠秒数（减少边界吞字；合并时文本仍简单拼接）",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行请求线程数。1=串行；默认按 Key 数量自动：min(48, max(8, n_keys*6))",
    )
    p.add_argument(
        "--rate-limit",
        type=int,
        default=None,
        help="Whisper ASR 滑动窗口最大请求数（默认 40/min；可用 WHISPER_RATE_LIMIT）。0=关闭",
    )
    p.add_argument(
        "--rate-window-sec",
        type=float,
        default=None,
        help="Whisper ASR 限速窗口秒数（默认 60；可用 WHISPER_RATE_WINDOW_SEC）",
    )
    p.add_argument("--keep-wav", action="store_true", help="保留中间整段 WAV")
    p.add_argument("--keep-chunks", action="store_true", help="保留分段 WAV 文件")
    p.add_argument("--no-srt", action="store_true", help="不生成 SRT 字幕")
    p.add_argument("--no-json", action="store_true", help="不生成 JSON")
    p.add_argument("--no-txt", action="store_true", help="不生成纯文本")
    p.add_argument(
        "--word-offsets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="请求词级时间戳（API 可能不返回）",
    )
    # —— 翻译（OpenAI 兼容）——
    p.add_argument(
        "--translate",
        action="store_true",
        help="转写后用 OpenAI 兼容接口翻译（默认目标简体中文）",
    )
    p.add_argument(
        "--to",
        dest="translate_to",
        default=None,
        help="翻译目标语言，默认 zh-CN / TRANSLATE_TO",
    )
    p.add_argument(
        "--openai-api-key",
        default=None,
        help="翻译 API Key；默认 OPENAI_API_KEY / LLM_API_KEY",
    )
    p.add_argument(
        "--openai-base-url",
        default=None,
        help="OpenAI 兼容 base URL，如 https://api.openai.com/v1",
    )
    p.add_argument(
        "--openai-model",
        default=None,
        help="翻译模型名，如 gpt-4o-mini",
    )
    p.add_argument(
        "--translate-workers",
        type=int,
        default=4,
        help="按字幕片段并行翻译的线程数",
    )
    p.add_argument(
        "--translate-rate-limit",
        type=int,
        default=None,
        help="翻译 API 滑动窗口限速（次），默认 40；0=关闭。可用 TRANSLATE_RATE_LIMIT",
    )
    p.add_argument(
        "--translate-rate-window-sec",
        type=float,
        default=None,
        help="翻译限速窗口秒数，默认 60",
    )
    p.add_argument(
        "--translate-only-zh-outputs",
        action="store_true",
        help="开启翻译时，SRT/TXT 默认只写中文文件（仍保留原文 txt/json 中的 text 字段）",
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

    # Whisper ASR 限速：CLI > 环境变量 > 默认 40/60s（按「每个 Key」计）
    if args.rate_limit is None:
        args.rate_limit = int(
            os.environ.get("WHISPER_RATE_LIMIT")
            or os.environ.get("NVIDIA_RATE_LIMIT")
            or os.environ.get("ASR_RATE_LIMIT")
            or DEFAULT_RATE_LIMIT
        )
    if args.rate_window_sec is None:
        args.rate_window_sec = float(
            os.environ.get("WHISPER_RATE_WINDOW_SEC")
            or os.environ.get("NVIDIA_RATE_WINDOW_SEC")
            or os.environ.get("ASR_RATE_WINDOW_SEC")
            or DEFAULT_RATE_WINDOW_SEC
        )

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        die(f"输入文件不存在: {input_path}")

    api_keys = load_nvidia_api_keys(
        cli_key=args.api_key,
        cli_keys=args.api_keys,
        keys_file=args.api_keys_file.expanduser() if args.api_keys_file else None,
    )
    if not api_keys:
        die(
            "未找到 API Key。请任选其一:\n"
            "  export NVIDIA_API_KEY='nvapi-...'\n"
            "  export NVIDIA_API_KEYS='key1,key2,key3'   # 多 Key 负载均衡\n"
            "  或 --api-keys-file keys.txt（每行一个）\n"
            "  或项目 .env（参考 .env.example）"
        )

    # workers：默认按 Key 数放大，吃满多 Key 配额
    if args.workers is None:
        args.workers = min(48, max(DEFAULT_WORKERS, len(api_keys) * 6))
    if args.workers < 1:
        die("--workers 至少为 1")

    out_dir = (args.output_dir or input_path.parent).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or default_out_stem(input_path)

    riva_client = import_riva()
    duration = probe_duration(input_path)

    pool = NvidiaApiKeyPool(
        api_keys,
        riva_client=riva_client,
        server=args.server,
        function_id=args.function_id,
        max_mb=args.max_message_mb,
        rate_limit=args.rate_limit,
        rate_window_sec=args.rate_window_sec,
    )

    with tempfile.TemporaryDirectory(prefix="whisper_nv_") as tmp:
        tmp_dir = Path(tmp)
        wav_work = out_dir if args.keep_wav else tmp_dir
        wav_path = ensure_wav(input_path, wav_work, args.sample_rate)

        wav_dur, wav_rate, _ = wav_duration_and_rate(wav_path)
        if duration is None:
            duration = wav_dur

        chunk_work = out_dir / f"{stem}_chunks" if args.keep_chunks else tmp_dir / "chunks"
        chunks = split_wav_chunks(
            wav_path,
            chunk_work,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            sample_rate=args.sample_rate,
        )

        if not args.quiet:
            size_mb = wav_path.stat().st_size / 1e6
            print(f"音频大小: {size_mb:.2f} MB | 时长约: {duration:.1f} s | 采样率: {wav_rate}")
            print(
                f"API Key 池: {pool.size} 个"
                + (
                    f" | 每 Key {args.rate_limit}/{args.rate_window_sec:g}s"
                    f" | 合计约 {pool.effective_rpm()}/min"
                    if args.rate_limit > 0
                    else " | 无限速"
                )
            )
            for i, k in enumerate(api_keys, 1):
                print(f"  [{i}] {mask_api_key(k)}")
            if args.chunk_seconds <= 0:
                print("分段: 关闭（整段一次请求）")
            else:
                mode = "并行" if args.workers > 1 else "串行"
                print(
                    f"分段: {len(chunks)} 片 × {args.chunk_seconds:g}s"
                    f"（重叠 {args.overlap_seconds:g}s）"
                    f" | workers={args.workers} ({mode})"
                )

        if not args.quiet:
            print("正在调用 NVIDIA Whisper Large V3 …")
        t0 = time.time()
        full_text, segments, chunk_meta = run_chunks(
            riva_client,
            pool,
            chunks,
            language_code=args.language,
            sample_rate=args.sample_rate,
            word_offsets=args.word_offsets,
            workers=args.workers,
            quiet=args.quiet,
        )
        elapsed = time.time() - t0
        if not args.quiet:
            print(f"全部完成，总耗时 {elapsed:.1f} s")
            print("Key 使用统计:")
            for row in pool.stats_summary():
                print(f"  {row['key']}: {row['requests']} 次")
    if not full_text:
        die("未得到任何转写文本（空结果）")

    timing_note = None
    if max_word_time(segments) <= 0:
        timing_note = "时间戳为按各 chunk 时间窗内字符比例估算（API 未返回词级时间戳）"
    elif any(s.get("timing") == "estimated" for s in segments):
        timing_note = "部分片段时间戳为估算"

    # —— 可选：OpenAI 兼容翻译 ——
    full_text_zh: str | None = None
    translation_meta: dict[str, Any] | None = None
    if args.translate:
        try:
            from translate_openai import OpenAICompatTranslator
        except ImportError:
            # 同目录导入
            sys.path.insert(0, str(script_dir))
            from translate_openai import OpenAICompatTranslator  # type: ignore

        try:
            translator = OpenAICompatTranslator.from_env(
                api_key=args.openai_api_key,
                base_url=args.openai_base_url,
                model=args.openai_model,
                target=args.translate_to,
                rate_limit=args.translate_rate_limit,
                rate_window_sec=args.translate_rate_window_sec,
            )
        except ValueError as e:
            die(str(e) + "\n  翻译需设置 OPENAI_API_KEY（及可选 OPENAI_BASE_URL / OPENAI_MODEL）")

        if not args.quiet:
            rl = (
                f"限速 {translator.config.rate_limit}/{translator.config.rate_window_sec:g}s"
                if translator.config.rate_limit > 0
                else "无限速"
            )
            print(
                f"正在翻译 → {translator.config.target} "
                f"| model={translator.config.model} "
                f"| base={translator.config.base_url} "
                f"| {rl}",
                flush=True,
            )
        t1 = time.time()
        full_text_zh, segments = translator.translate_segments_as_text_list(
            segments,
            text_key="text",
            out_key="text_zh",
            workers=max(1, args.translate_workers),
            quiet=args.quiet,
        )
        # 无分段时退回整篇翻译
        if not full_text_zh and full_text:
            full_text_zh = translator.translate(full_text)
        translation_meta = {
            "target": translator.config.target,
            "model": translator.config.model,
            "base_url": translator.config.base_url,
            "workers": args.translate_workers,
            "rate_limit": translator.config.rate_limit,
            "rate_window_sec": translator.config.rate_window_sec,
            "elapsed_sec": round(time.time() - t1, 3),
        }
        if not args.quiet:
            print(f"翻译完成，耗时 {translation_meta['elapsed_sec']:.1f} s", flush=True)

    written: list[Path] = []

    if not args.no_txt:
        txt_path = out_dir / f"{stem}_transcript.txt"
        txt_path.write_text(full_text + "\n", encoding="utf-8")
        written.append(txt_path)
        if full_text_zh is not None:
            zh_txt = out_dir / f"{stem}_transcript.zh.txt"
            zh_txt.write_text(full_text_zh + "\n", encoding="utf-8")
            written.append(zh_txt)

    if not args.no_json:
        json_path = out_dir / f"{stem}_transcript.json"
        payload: dict[str, Any] = {
            "model": "openai/whisper-large-v3",
            "source": str(input_path),
            "language": args.language,
            "duration_sec": duration,
            "chunk_seconds": args.chunk_seconds,
            "overlap_seconds": args.overlap_seconds,
            "workers": args.workers,
            "rate_limit_per_key": args.rate_limit,
            "rate_window_sec": args.rate_window_sec,
            "api_key_count": pool.size,
            "effective_rpm": pool.effective_rpm(),
            "api_key_stats": pool.stats_summary(),
            "chunks": chunk_meta,
            "timing_note": timing_note,
            "text": full_text,
            "segments": segments,
        }
        if full_text_zh is not None:
            payload["text_zh"] = full_text_zh
            payload["translation"] = translation_meta
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(json_path)

    if not args.no_srt:
        # 原文 SRT（除非只要中文输出且开启了 translate-only 风格——仍写原文轨便于对照）
        if not (args.translate and args.translate_only_zh_outputs):
            srt_path = out_dir / f"{stem}.srt"
            n = write_srt(srt_path, segments, duration)
            written.append(srt_path)
            if not args.quiet:
                print(f"SRT 字幕条数: {n}" + ("（时间戳含估算）" if timing_note else ""))
        if full_text_zh is not None:
            # 用 text_zh 覆盖 text 写中文字幕轨
            zh_segments = []
            for s in segments:
                ns = dict(s)
                if ns.get("text_zh"):
                    ns["text"] = ns["text_zh"]
                # 词级英文时间戳对中文无意义，清掉以免 SRT 用词级英文
                ns["words"] = []
                zh_segments.append(ns)
            zh_srt = out_dir / f"{stem}.zh.srt"
            n_zh = write_srt(zh_srt, zh_segments, duration)
            written.append(zh_srt)
            if not args.quiet:
                print(f"中文 SRT 字幕条数: {n_zh}")

    if not args.quiet:
        print(f"字数约: {len(full_text.split())} 词 / {len(full_text)} 字符")
        if full_text_zh is not None:
            print(f"译文约: {len(full_text_zh)} 字符")
        print("已写入:")
        for p in written:
            print(f"  - {p}")
        print("--- 原文预览（前 400 字）---")
        print(full_text[:400] + ("…" if len(full_text) > 400 else ""))
        if full_text_zh:
            print("--- 译文预览（前 400 字）---")
            print(full_text_zh[:400] + ("…" if len(full_text_zh) > 400 else ""))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
