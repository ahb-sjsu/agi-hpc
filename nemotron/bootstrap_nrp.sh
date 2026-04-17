#!/usr/bin/env bash
# Nemotron NRP bootstrap — runs inside pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel
set -euo pipefail

echo "[bootstrap] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no GPU')"

echo "[bootstrap] pip install deps (--no-deps for torch-dependent packages)"
pip install --quiet --no-cache-dir --no-deps \
    transformers peft accelerate trl
pip install --quiet --no-cache-dir \
    safetensors tokenizers huggingface-hub \
    datasets bitsandbytes sentencepiece protobuf polars numpy \
    jinja2 regex requests tqdm pyyaml packaging \
    xxhash multiprocess dill pyarrow fsspec aiohttp einops

echo "[bootstrap] compile causal-conv1d + mamba-ssm (CUDA, ~5-10 min)"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0 8.0 9.0}"
pip install --quiet --no-cache-dir --no-build-isolation causal-conv1d
pip install --quiet --no-cache-dir --no-build-isolation mamba-ssm

echo "[bootstrap] fetch training script + data"
REPO_URL="${REPO_URL:-https://github.com/ahb-sjsu/agi-hpc.git}"
git clone --depth 1 "$REPO_URL" /tmp/agi-hpc 2>/dev/null || true
cp /tmp/agi-hpc/nemotron/nemotron_nrp.py /app/ 2>/dev/null || true
cp /tmp/agi-hpc/nemotron/nemotron_v3_kaggle.py /app/ 2>/dev/null || true

# Competition data — download or use mounted /data
if [ ! -f /data/train.csv ]; then
  echo "[bootstrap] downloading competition data"
  mkdir -p /data
  # Fetch from a hosted URL (user provides via DATA_URL env)
  if [ -n "${DATA_URL:-}" ]; then
    curl -sL "$DATA_URL" | tar xz -C /data
  else
    echo "[bootstrap] WARNING: no /data/train.csv and no DATA_URL set"
  fi
fi
echo "[bootstrap] data: $(wc -l /data/train.csv 2>/dev/null || echo missing)"

echo "[bootstrap] starting training"
cd /app
exec python3 nemotron_nrp.py "$@"
