#!/bin/bash
# Preload models on startup

MODELS=("codellama:7b" "llama3.2:1b" "nomic-embed-text")

for model in "${MODELS[@]}"; do
    echo "Preloading $model..."
    curl -X POST http://localhost:11434/api/generate \
        -d "{\"model\": \"$model\", \"prompt\": \"test\", \"stream\": false}" \
        > /dev/null 2>&1 &
done

wait
echo "All models preloaded"
