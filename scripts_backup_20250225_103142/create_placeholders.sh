#!/bin/bash
set -euo pipefail

# List of essential subdirectories to create as placeholders
PLACEHOLDER_DIRS=(
    "ai_agents/supreme_ai"
    "ai_agents/auto_gpt"
    "ai_agents/superagi"
    "model_management/GPT4All"
    "model_management/DeepSeek-R1"
    "model_management/Molmo"
    "web_ui/src"
    "web_ui/public"
    "packages/wheels"
    "packages/node"
    "doc_data/pdfs"
    "doc_data/diagrams"
)

for d in "${PLACEHOLDER_DIRS[@]}"; do
    if [ ! -d "$d" ]; then
         echo "Creating directory $d"
         mkdir -p "$d"
         touch "$d/.gitkeep"
    else
         echo "Directory $d already exists."
    fi
done

echo "Placeholder directories created." 
 