#!/usr/bin/env bash
# 持续拉取 HuggingFace GigaSpeech → Whisper → 中文翻译
# 音频仅驻内存；HF 缓存定期删除，只保留 JSONL + 进度库。
#
# 用法:
#   ./run_gigaspeech_continuous.sh
#   ./run_gigaspeech_continuous.sh --subset xs --max-samples 0
#   ./run_gigaspeech_continuous.sh --subset s --max-samples 1000
#
# 依赖 .env: HF_TOKEN, NVIDIA_API_KEYS_FILE / keys, OPENAI_* 翻译配置
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "错误: 请设置 HF_TOKEN（GigaSpeech 为 gated 数据集）" >&2
  exit 1
fi

if [[ -z "${NVIDIA_API_KEY:-}" && -z "${NVIDIA_API_KEYS:-}" && -z "${NVIDIA_API_KEYS_FILE:-}" ]]; then
  if [[ ! -f "$ROOT/nvidia_api_keys.txt" ]]; then
    echo "错误: 请配置 NVIDIA API Key（NVIDIA_API_KEYS_FILE 或 nvidia_api_keys.txt）" >&2
    exit 1
  fi
  export NVIDIA_API_KEYS_FILE="${NVIDIA_API_KEYS_FILE:-$ROOT/nvidia_api_keys.txt}"
fi

if [[ -z "${OPENAI_API_KEY:-}" && -z "${LLM_API_KEY:-}" ]]; then
  echo "警告: 未设置 OPENAI_API_KEY，翻译可能失败。请检查 .env" >&2
fi

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

OUT_DIR="${GIGASPEECH_OUT_DIR:-$ROOT/out/gigaspeech_continuous}"
SUBSET="${GIGASPEECH_SUBSET:-xs}"
CACHE_DIR="${GIGASPEECH_HF_CACHE:-$OUT_DIR/.hf_cache}"

mkdir -p "$OUT_DIR" "$CACHE_DIR"

echo "=========================================="
echo " GigaSpeech 持续流水线"
echo " subset=$SUBSET"
echo " out=$OUT_DIR"
echo " cache=$CACHE_DIR (会定期删除)"
echo " 流程: 拉取 → Whisper → 译中 → 丢弃音频/缓存"
echo "=========================================="

# 默认：无限条、开启翻译、resume、定期清理
# 其余参数可透传，例如 --max-samples 100 --subset s
exec "$PY" -m dataset.gigaspeech_pipeline \
  --subset "$SUBSET" \
  --split train \
  --out-dir "$OUT_DIR" \
  --hf-cache-dir "$CACHE_DIR" \
  --translate \
  --resume \
  --cleanup-every 20 \
  --min-free-gb 2 \
  --max-cache-gb 1.5 \
  --max-in-flight 8 \
  "$@"
