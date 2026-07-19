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

## 工作原理

当前实现为 **串行、整段 offline 识别**：

1. `ffmpeg` 将输入转为 16 kHz 单声道 PCM WAV  
2. 通过 gRPC 一次请求 `grpc.nvcf.nvidia.com:443`  
3. metadata：`function-id` + `authorization: Bearer <API_KEY>`  
4. 写出 txt / json / srt  

不适合超长文件时，可能受 gRPC 消息大小与超时限制；可按需自行分片。

## 许可与条款

- 本仓库脚本代码：MIT（见下方）  
- 调用 NVIDIA 托管 API 须遵守 [NVIDIA API Trial Terms](https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf) 及 build.nvidia.com 相关条款  

## License

MIT
