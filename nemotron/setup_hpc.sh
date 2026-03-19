#!/bin/bash
# Nemotron HPC Setup — Run on login node ONCE
# Then: sbatch submit_nemotron.sh
set -e

echo "=== Nemotron HPC Setup ==="
WORK_DIR=~/nemotron
mkdir -p $WORK_DIR/data $WORK_DIR/output

module load anaconda/3.9
module load cuda/12.2

# Create conda env
if [ ! -d $WORK_DIR/env ]; then
    echo "Creating conda environment..."
    conda create -y -p $WORK_DIR/env python=3.11 numpy pandas pip
fi

source activate $WORK_DIR/env

# PyTorch with CUDA 12.1 (compatible with cuda/12.2 module)
pip install --no-build-isolation torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -5

# Core training packages
pip install --no-build-isolation \
    transformers peft accelerate bitsandbytes datasets trl \
    sentencepiece protobuf 2>&1 | tail -5

# Nemotron requires mamba-ssm and causal-conv1d (need CUDA to compile)
echo "Building mamba-ssm (requires CUDA, may take 5-10 min)..."
pip install causal-conv1d 2>&1 | tail -5
pip install mamba-ssm 2>&1 | tail -5

# Download model weights (do this on login node with internet)
echo "Downloading Nemotron-3-Nano-30B model weights..."
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16', trust_remote_code=True)
print('Downloading model (this will take a while)...')
AutoModelForCausalLM.from_pretrained('nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16', trust_remote_code=True)
print('Model cached.')
"

# Copy competition data (upload train.csv and test.csv to ~/nemotron/data/ first)
if [ ! -f $WORK_DIR/data/train.csv ]; then
    echo ""
    echo "WARNING: No training data found!"
    echo "Upload train.csv and test.csv to $WORK_DIR/data/"
    echo "  scp train.csv test.csv coe-hpc1.sjsu.edu:~/nemotron/data/"
fi

echo ""
echo "Data files:"
ls -la $WORK_DIR/data/*.csv 2>/dev/null || echo "  (none — upload data first)"
echo ""
echo "Setup complete! Next steps:"
echo "  1. Upload data:  scp data/train.csv data/test.csv coe-hpc1.sjsu.edu:~/nemotron/data/"
echo "  2. Upload script: scp train_hpc.py coe-hpc1.sjsu.edu:~/nemotron/"
echo "  3. Submit job:   sbatch ~/nemotron/submit_nemotron.sh"
