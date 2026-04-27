#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

mkdir -p data/input data/output

echo "=== Step 1: Generate synthetic noisy signal dataset ==="
python3 src/generate_dataset.py \
    --output data/input \
    --count  200        \
    --length 1024       \
    --noise  0.25

echo ""
echo "=== Step 2: Build CUDA smoother ==="
make

echo ""
echo "=== Step 3: Run GPU batch smoothing ==="
./bin/batch_signal_smoother data/input data/output

echo ""
echo "Run complete. Results in data/output/  |  Log: data/output/run_log.txt"
