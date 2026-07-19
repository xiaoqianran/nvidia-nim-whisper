#!/usr/bin/env bash
# 便捷包装：加载本地 .env，优先使用本目录 .venv
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# 若尚未 export，则尝试从 .env 注入（不覆盖已有环境变量）
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if [[ -z "${NVIDIA_API_KEY:-}" && -z "${NGC_API_KEY:-}" ]]; then
  echo "错误: 请 export NVIDIA_API_KEY='nvapi-...' 或在项目根目录创建 .env" >&2
  echo "参考: .env.example" >&2
  exit 1
fi

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

exec "$PY" "$ROOT/transcribe_whisper_nvidia.py" "$@"
