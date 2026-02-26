#!/usr/bin/env bash
# Download a GGUF model file from HuggingFace.
#
# Usage: scripts/download-model.sh <hf-repo> <filename> [output-dir]
#
# Examples:
#   scripts/download-model.sh Qwen/Qwen3-0.6B-GGUF qwen3-0.6b-q8_0.gguf
#   scripts/download-model.sh bartowski/Qwen3-Coder-Next-3B-GGUF qwen3-coder-next-3b-q4_k_m.gguf data/models

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <hf-repo> <filename> [output-dir]" >&2
    exit 1
fi

HF_REPO="$1"
FILENAME="$2"
OUTPUT_DIR="${3:-data/models}"

mkdir -p "$OUTPUT_DIR"

OUTPUT_PATH="$OUTPUT_DIR/$FILENAME"

if [[ -f "$OUTPUT_PATH" ]]; then
    echo "Already exists: $OUTPUT_PATH (skipping)"
    exit 0
fi

URL="https://huggingface.co/${HF_REPO}/resolve/main/${FILENAME}"

echo "Downloading: $URL"
echo "        to: $OUTPUT_PATH"

# Download with resume support
curl -fL --retry 3 --retry-delay 5 -C - -o "$OUTPUT_PATH.tmp" "$URL"
mv "$OUTPUT_PATH.tmp" "$OUTPUT_PATH"

echo "Done: $OUTPUT_PATH ($(du -h "$OUTPUT_PATH" | cut -f1))"
