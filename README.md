# nvidia-nim-whisper

通过 [NVIDIA build.nvidia.com](https://build.nvidia.com/openai/whisper-large-v3) 托管的 **OpenAI Whisper Large V3**（NIM / Riva gRPC）转录本地音视频，输出：

- `*_transcript.txt` — 纯文本（原文）
- `*_transcript.json` — 分段 + 元数据
- `*.srt` — 字幕（无词级时间戳时按比例估算）
- 可选 `--translate`：`*_transcript.zh.txt` / `*.zh.srt`（OpenAI 兼容 LLM 译成中文等）

模型页: https://build.nvidia.com/openai/whisper-large-v3/api

> Whisper 自带 `task=translate` **只能译成英文**。要中文请用本仓库的 **OpenAI 兼容翻译模块**。

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
# 单 Key
export NVIDIA_API_KEY='nvapi-...'

# 多 Key 负载均衡（推荐）：总配额 ≈ n × 60/min
export NVIDIA_API_KEYS='nvapi-key1,nvapi-key2,nvapi-key3'
# 或每行一个的文件
export NVIDIA_API_KEYS_FILE=./nvidia_api_keys.txt
```

#### 多 Key 方案（突破 40/min）

| 概念 | 说明 |
|------|------|
| 每 Key 限速 | 默认 40 次 / 60s 滑动窗口 |
| 池化调度 | 有余量的 Key 轮询使用；全满则等最早释放的 Key |
| 有效吞吐 | `n_keys × 40 / min`（6 个 Key ≈ **300/min**） |
| 并发 | 默认 `workers = min(48, max(8, n_keys×6))` |
| 失败 | 疑似限流时换 Key 重试 |

```bash
# keys.txt 每行一个 nvapi-...
./transcribe.sh talk.mp3 --api-keys-file keys.txt
./transcribe.sh talk.mp3 --api-keys 'k1,k2,k3' --workers 36
```

### 2. 本地 `.env`

```bash
cp .env.example .env
# 编辑 .env，填入 NVIDIA_API_KEY=nvapi-...
# 若需要翻译，再填 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
```

`.env` 已在 `.gitignore` 中，不会被提交。

### 3. 命令行参数

```bash
python transcribe_whisper_nvidia.py media.mp4 --api-key 'nvapi-...'
```

优先级：`--api-key` > 已 export 的环境变量 > `.env`。

### 4. 翻译（NVIDIA 多 Key 负载均衡）

与 Whisper 相同思路：多把 `nvapi-`，**每把独立 60/min**，客户端轮询。

```bash
# nvidia_translate_api_keys.txt 每行一个 nvapi-...
# OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
# OPENAI_MODEL=mistralai/mistral-small-4-119b-2603
# TRANSLATE_RATE_LIMIT=50   # 每 Key

./transcribe.sh talk.mp3 --translate
python translate_openai.py --text "Hello, Spring Boot"
```

6 把翻译 Key → 理论 **≈ 360 次/分钟**。
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
# 默认：30s 分段 + 8 并行 + 60/min 限速
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
| `--rate-limit` | **Whisper ASR** 滑动窗口次数，默认 `50`；`0`=关闭（也可用 `WHISPER_RATE_LIMIT`） |
| `--rate-window-sec` | Whisper 限速窗口秒数，默认 `60`（`WHISPER_RATE_WINDOW_SEC`） |
| `--keep-chunks` | 保留 `*_chunks/` 下分段 WAV |
| `--translate` | 转写后翻译（需 OpenAI 兼容 Key） |
| `--to` | 目标语言，默认 `zh-CN` |
| `--openai-base-url` / `--openai-model` / `--openai-api-key` | 翻译端点 |
| `--translate-workers` | 按片段并行翻译线程数，默认 4 |
| `--translate-rate-limit` | 翻译 **每 Key** 滑动窗口次数，默认 **50**/min |
| `--translate-rate-window-sec` | 翻译限速窗口，默认 60s |

> 串行分段通常不比「整段一次」更快；**并行 + 限速** 才能明显缩短墙钟时间（受 API 并发与配额约束）。

翻译产物：

| 文件 | 内容 |
|------|------|
| `*_transcript.txt` / `*.srt` | 原文（Whisper） |
| `*_transcript.zh.txt` / `*.zh.srt` | 译文 |
| `*_transcript.json` | 含 `text`、`text_zh`、`segments[].text_zh` |

## YouTube 字幕模块（yt-dlp）

给定视频链接：

1. **有简体中文字幕**（`zh-Hans` / `zh-CN` 等）→ 下载简体字幕 + 写出 `*.zh.txt`（及 `*.zh.srt`）  
2. **没有简体、有英文** → 下载英文 → `*.en.txt` / `*.en.srt`，再用现有 **NVIDIA 翻译库** 生成 `*.zh.srt` + `*.zh.txt`  
3. **不下载整段视频**；可用 cookies 提高成功率  

```bash
# 依赖: pip install yt-dlp
chmod +x run_youtube.sh

./run_youtube.sh "https://www.youtube.com/watch?v=VIDEO_ID" \
  --cookies /root/Desktop/www.youtube.com_cookies.txt \
  -o ./out/youtube

# 或
python -m youtube.cli "URL" --cookies /path/to/cookies.txt -o ./out/youtube
```

输出示例：

- 简体路径：`标题_ID.zh.txt`、`标题_ID.zh.srt`  
- 英文路径：`标题_ID.en.txt`、`标题_ID.en.srt`、`标题_ID.zh.txt`、`标题_ID.zh.srt`  
- `标题_ID.langs.json`：可用语言列表  

单元测试（无网络）：

```bash
pytest tests/test_youtube_captions.py -q
```

### 频道最近 N 视频（元数据 + 字幕/Whisper）

对频道页抓最近 N 条：标题/简介（原文+简体）+ 正文转写：

1. 有**简体字幕** → 下载简体 txt/srt  
2. 有**英文字幕** → 英文 txt/srt + 译简体  
3. **都没有** → 下音频 → Whisper → 译简体（用完删音频）

```bash
chmod +x run_youtube_channel.sh

./run_youtube_channel.sh "https://www.youtube.com/@AIsuperdomain" \
  --limit 5 \
  --cookies /root/Desktop/www.youtube.com_cookies.txt \
  -o ./out/channel_AIsuperdomain
```

公共逻辑：`common/translate_cues.py`（cue 翻译）、`common/lang_detect.py`（中文启发/Whisper 语种）、`youtube/audio_whisper.py`（音频 Whisper）。

频道输出约定：

- **README.md**：标题/简介原文与简体均为**全文**（不再单独写简介 md）  
- **latest.json**：机器可读全量  
- **短目录名**（字幕/转写）：`out/{video_id}/` → `{video_id}.zh.txt` 等  
- **仅给总结用中文**：`./run_youtube_channel.sh ... --zh-only`（不写英文 en.*）  
- **断点续跑**：默认 `--resume`  
- **语种自动**：中文片 Whisper `zh-CN`；已是中文则跳过翻译  

### 总结子模块（公共 + YouTube）

公共：`common/summarize.py` + `common/llm_chat.py`  
- 模型默认 **`stepfun-ai/step-3.5-flash`**  
- Key 池：`nvidia_summarize_api_keys.txt`，**每 Key 40/min**  
- 超长文稿自动分块 map-reduce（默认每块 ≤24000 字符）

```bash
# 对已有中文台词/字幕做总结
./run_youtube_summarize.sh path/to/xxx.zh.txt -o path/to/xxx.summary.md
# 或目录
./run_youtube_summarize.sh out/channel_xxx/VIDEO_ID --video-id VIDEO_ID
```

## 数据集模块：GigaSpeech 增量流水线

面向 [speechcolab/gigaspeech](https://huggingface.co/datasets/speechcolab/gigaspeech)，**不整库下载**：

```text
HF streaming → 内存 PCM → Whisper(Key 池) →（可选）中文翻译 → JSONL
音频处理完即丢弃；磁盘 mainly 只有结果 + 进度库
```

与本地媒体模块的区别：

| | `transcribe_whisper_nvidia.py` | `dataset.gigaspeech_pipeline` |
|--|--------------------------------|-------------------------------|
| 输入 | 本地音视频文件 | HuggingFace streaming |
| 切分 | 默认 30s | **官方 segment**，默认不再切 |
| 输出 | 每文件 txt/srt | 流式 **JSONL** + SQLite 断点 |

### 安装额外依赖

```bash
pip install -r requirements.txt   # 含 datasets / numpy
huggingface-cli login             # 或 export HF_TOKEN=hf_...
# 在 HF 网页同意 GigaSpeech 使用条款
```

### 运行

**推荐：持续拉取 + 自动 Whisper + 译中 + 清理缓存**

```bash
# 配置 .env: HF_TOKEN / NVIDIA keys / OPENAI_*
chmod +x run_gigaspeech_continuous.sh

# 一直跑 xs（Ctrl+C 可停，--resume 可续）
./run_gigaspeech_continuous.sh

# 或限制条数试跑
./run_gigaspeech_continuous.sh --max-samples 100

# 换子集
GIGASPEECH_SUBSET=s ./run_gigaspeech_continuous.sh
```

行为：

1. HF **streaming** 按条拉取（不整库下载）  
2. 内存里 PCM → Whisper（多 Key 池）→ 中文翻译  
3. 写出 JSONL 一行后 **丢掉音频数组**  
4. 每 20 条（可调）清理 `.hf_cache`，结束时再清一遍  
5. 磁盘长期只剩：`*.jsonl` + `*.state.sqlite`

```bash
# 等价 Python 入口
python -m dataset.run_continuous --subset xs --max-samples 50

# 底层参数全开时
python -m dataset.gigaspeech_pipeline \
  --subset xs --max-samples 50 --translate \
  --out-dir ./out/gigaspeech_xs \
  --cleanup-every 20 --max-cache-gb 1.5 --min-free-gb 2

# 跳过 ASR，只用官方英文 ref 译中（省 NVIDIA 配额）
python -m dataset.gigaspeech_pipeline --subset xs --skip-whisper-use-ref --translate --max-samples 100
```

产物：

- `out/.../gigaspeech_{subset}_{split}.jsonl` — 每行一条 segment  
- `out/.../gigaspeech_{subset}_{split}.state.sqlite` — 已完成 id（resume）

JSONL 字段含：`segment_id`, `ref_text`, `whisper_text`, `text_zh`, `duration_sec`, `asr_key`, `error` 等。

省磁盘建议：

- 始终 `streaming`（默认）
- `--hf-cache-dir` 指到可清理的小目录 / tmp
- 不要用非 streaming 的 `load_dataset` 全量下载

## 许可与条款

- 本仓库脚本代码：MIT（见下方）  
- 调用 NVIDIA 托管 API 须遵守 [NVIDIA API Trial Terms](https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf) 及 build.nvidia.com 相关条款  
- GigaSpeech 须遵守 SpeechColab / HuggingFace 数据集条款  

## License

MIT
