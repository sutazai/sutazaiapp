#!/bin/bash
# Preload models on startup

MODELS=("qwen2.5-coder:3b" "qwen2.5:3b" "nomic-embed-text")

for model in "${MODELS[@]}"; do
    echo "Preloading $model..."
    curl -X POST http://localhost:11434/api/generate \
        -d "{\"model\": \"$model\", \"prompt\": \"test\", \"stream\": false}" \
        > /dev/null 2>&1 &
done

wait
echo "All models preloaded"
