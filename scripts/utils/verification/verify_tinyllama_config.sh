#!/bin/bash

echo "🔍 Verifying TinyLlama configuration..."
echo "======================================="

# Check environment files
echo "📝 Checking environment files..."
for env_file in .env .env.agents .env.example .env.production .env.ollama .env.tinyllama; do
    if [ -f "$env_file" ]; then
        echo -n "  $env_file: "
        if grep -q "DEFAULT_MODEL=tinyllama" "$env_file" 2>/dev/null; then
            echo "✅ TinyLlama configured"
        else
            echo "⚠️  Missing DEFAULT_MODEL=tinyllama"
        fi
    fi
done

# Check for any remaining deepseek references
echo ""
echo "🔍 Checking for remaining deepseek-r1 references..."
deepseek_count=$(find . -type f \( -name "*.env*" -o -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "*.toml" -o -name "*.py" -o -name "*.sh" \) -not -path "./.git/*" -exec grep -l "deepseek-r1" {} \; 2>/dev/null | wc -l)

if [ "$deepseek_count" -eq 0 ]; then
    echo "  ✅ No deepseek-r1 references found!"
else
    echo "  ⚠️  Found $deepseek_count files with deepseek-r1 references"
fi

# Check Docker configurations
echo ""
echo "🐳 Checking Docker configurations..."
for compose_file in docker-compose.yml docker-compose.tinyllama.yml docker-compose.agents.yml; do
    if [ -f "$compose_file" ]; then
        echo -n "  $compose_file: "
        if grep -q "DEFAULT_MODEL" "$compose_file" 2>/dev/null; then
            if grep -q "tinyllama" "$compose_file" 2>/dev/null; then
                echo "✅ TinyLlama configured"
            else
                echo "⚠️  Not using TinyLlama"
            fi
        else
            echo "✅ No model configuration (inherits from .env)"
        fi
    fi
done

# Check agent configurations
echo ""
echo "🤖 Checking agent configurations..."
agent_count=$(find agents -name "*_ollama.json" 2>/dev/null | wc -l)
tinyllama_count=$(find agents -name "*_ollama.json" -exec grep -l "tinyllama" {} \; 2>/dev/null | wc -l)
echo "  Total Ollama configs: $agent_count"
echo "  Using TinyLlama: $tinyllama_count"
if [ "$agent_count" -eq "$tinyllama_count" ]; then
    echo "  ✅ All agents configured for TinyLlama!"
else
    echo "  ⚠️  Some agents not using TinyLlama"
fi

# Check Ollama service
echo ""
echo "🔧 Checking Ollama service configuration..."
if docker ps --format "table {{.Names}}" | grep -q "ollama"; then
    echo "  ✅ Ollama container is running"
    # Check if tinyllama model is available
    if docker exec ollama ollama list 2>/dev/null | grep -q "tinyllama"; then
        echo "  ✅ TinyLlama model is available in Ollama"
    else
        echo "  ⚠️  TinyLlama model not found in Ollama"
        echo "     Run: docker exec ollama ollama pull tinyllama:latest"
    fi
else
    echo "  ⚠️  Ollama container not running"
fi

echo ""
echo "======================================="
echo "📊 Configuration Summary:"
echo "  - Model: tinyllama (637MB)"
echo "  - API: Native Ollama (port 11434)"
echo "  - LiteLLM: Removed completely"
echo "  - Agents: Using _ollama.json configs"
echo ""
echo "🚀 To start the system: ./start_tinyllama.sh"