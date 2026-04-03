#!/bin/bash
set -euo pipefail

MODEL="${1:?Usage: $0 <model_path> [n_gpu_layers] [ubatch_sizes]}"
NGL="${2:-99}"
UB_SIZES="${3:-512,1024,2048}"

echo "=== GDN Chunked Kernel Benchmark ==="
echo "Model: $MODEL"
echo "GPU layers: $NGL"
echo "Ubatch sizes: $UB_SIZES"
echo ""

IFS=',' read -ra UB_ARRAY <<< "$UB_SIZES"

for ub in "${UB_ARRAY[@]}"; do
    echo "--- ubatch=$ub ---"
    ./bin/llama-bench \
        -m "$MODEL" \
        -ngl "$NGL" \
        -ub "$ub" \
        -fa 1 \
        -t 1 \
        -p 512,2048 \
        -n 128 \
        -r 3 2>&1 || echo "FAILED with ubatch=$ub"
    echo ""
done
