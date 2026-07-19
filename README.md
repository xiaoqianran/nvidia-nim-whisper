# nvidia-nim-whisper

通过 [NVIDIA build.nvidia.com](https://build.nvidia.com/openai/whisper-large-v3) 托管的 **OpenAI Whisper Large V3**（NIM / Riva gRPC）转录本地音视频，输出：

- `*_transcript.txt` — 纯文本
- `*_transcript.json` — 分段 + 元数据
- `*.srt` — 字幕（无词级时间戳时按比例估算）

模型页: https://build.nvidia.com/openai/whisper-large-v3/api

## 依赖

- Python 3.10+
- 系统安装 [ffmpeg](https://ffmpeg.org/)（含 `ffprobe`）
- NVIDIA API Key（[获取](https://build.nvidia.com/)）

## 安装

```bash
git clone https://github.com/xiaoqianran/nvidia-nim-whisper.git
cd nvidia-nim-whisper

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

也可用 [uv](https://github.com/astral-sh/uv)：

```bash
uv venv && uv pip install -r requirements.txt
```

## 配置 API Key

**不要**把 Key 写进代码或提交到 Git。任选一种方式：

### 1. 环境变量

```bash
export NVIDIA_API_KEY='nvapi-...'
```

### 2. 本地 `.env`

```bash
cp .env.example .env
# 编辑 .env，填入 NVIDIA_API_KEY=nvapi-...
```

`.env` 已在 `.gitignore` 中，不会被提交。

### 3. 命令行参数

```bash
python transcribe_whisper_nvidia.py media.mp4 --api-key 'nvapi-...'
```

优先级：`--api-key` > 已 export 的环境变量 > `.env`。

## 使用

```bash
# 推荐
./transcribe.sh your_video.mp4

# 或
python transcribe_whisper_nvidia.py your_video.mp4
python transcribe_whisper_nvidia.py audio.wav -o ./out --language en-US
python transcribe_whisper_nvidia.py video.mp4 --keep-wav --stem my_talk
```

### 常用参数

| 参数 | 说明 |
|------|------|
| `-o / --output-dir` | 输出目录（默认与输入同目录） |
| `--stem` | 输出文件名前缀 |
| `--language` | 如 `en-US`、`zh-CN`、`multi` |
| `--keep-wav` | 保留中间 16 kHz mono WAV |
| `--no-srt` / `--no-json` / `--no-txt` | 跳过对应输出 |
| `--env-file PATH` | 指定 `.env` 路径 |
| `-q` | 安静模式 |

## 单文件分段

默认将**单个**音视频按时间切成多段，再 **并行** 调 API（`--workers 8`），并遵守滑动窗口限速：

1. `ffmpeg` 转为 16 kHz 单声道 PCM WAV  
2. 按 `--chunk-seconds`（默认 30s）切 chunk，可选 `--overlap-seconds` 重叠  
3. 线程池并行 gRPC `offline_recognize`（客户端限速 **40 次 / 60s 滑动窗口**，对齐 NVIDIA Trial）  
4. 按 chunk 起始时间偏移合并文本 / JSON / SRT  

```bash
# 默认：30s 分段 + 8 并行 + 40/min 限速
./transcribe.sh talk.mp3

# 强制串行（便于对比）
./transcribe.sh talk.mp3 --workers 1

# 60 秒一片，边界重叠 1 秒，16 并行
./transcribe.sh talk.mp3 --chunk-seconds 60 --overlap-seconds 1 --workers 16

# 不切分（整段一次请求）
./transcribe.sh talk.mp3 --chunk-seconds 0

# 关闭客户端限速（不推荐，易 429）
./transcribe.sh talk.mp3 --rate-limit 0

# 保留分段 WAV 便于调试
./transcribe.sh talk.mp3 --keep-chunks
```

| 参数 | 说明 |
|------|------|
| `--chunk-seconds` | 每段时长（秒），`<=0` 关闭分段 |
| `--overlap-seconds` | 相邻段重叠，减轻边界吞字 |
| `--workers` | 并行线程数，默认 `8`；`1`=串行 |
| `--rate-limit` | 滑动窗口最大请求数，默认 `40`；`0`=关闭 |
| `--rate-window-sec` | 限速窗口秒数，默认 `60` |
| `--keep-chunks` | 保留 `*_chunks/` 下分段 WAV |

> 串行分段通常不比「整段一次」更快；**并行 + 限速** 才能明显缩短墙钟时间（受 API 并发与配额约束）。

## 许可与条款

- 本仓库脚本代码：MIT（见下方）  
- 调用 NVIDIA 托管 API 须遵守 [NVIDIA API Trial Terms](https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf) 及 build.nvidia.com 相关条款  

## License

MIT
