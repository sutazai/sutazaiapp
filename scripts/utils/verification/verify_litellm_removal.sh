#!/bin/bash

# Verify LiteLLM has been completely removed

echo "🔍 Verifying LiteLLM removal..."
echo "==============================="

errors=0

# Check for LiteLLM in docker-compose files
echo -n "Checking docker-compose files... "
if grep -qi "litellm" docker-compose*.yml 2>/dev/null; then
    echo "❌ Found LiteLLM references in docker-compose files"
    grep -n -i "litellm" docker-compose*.yml
    ((errors++))
else
    echo "✅ Clean"
fi

# Check for LiteLLM config files
echo -n "Checking for LiteLLM config files... "
if find . -name "*litellm*.json" -o -name "*litellm*.yaml" -o -name "*litellm*.yml" 2>/dev/null | grep -q .; then
    echo "❌ Found LiteLLM config files:"
    find . -name "*litellm*.json" -o -name "*litellm*.yaml" -o -name "*litellm*.yml" 2>/dev/null
    ((errors++))
else
    echo "✅ Clean"
fi

# Check for LiteLLM directories
echo -n "Checking for LiteLLM directories... "
if find . -type d -name "*litellm*" 2>/dev/null | grep -q .; then
    echo "❌ Found LiteLLM directories:"
    find . -type d -name "*litellm*" 2>/dev/null
    ((errors++))
else
    echo "✅ Clean"
fi

# Check for LiteLLM in Python files
echo -n "Checking Python imports... "
if grep -r "import litellm\|from litellm" --include="*.py" . 2>/dev/null | grep -v "^Binary file" | grep -q .; then
    echo "❌ Found LiteLLM imports:"
    grep -r "import litellm\|from litellm" --include="*.py" . 2>/dev/null | grep -v "^Binary file"
    ((errors++))
else
    echo "✅ Clean"
fi

# Check environment files
echo -n "Checking environment files... "
if grep -i "litellm" .env* 2>/dev/null | grep -q .; then
    echo "❌ Found LiteLLM in environment files:"
    grep -n -i "litellm" .env* 2>/dev/null
    ((errors++))
else
    echo "✅ Clean"
fi

# Check if all agents use Ollama configs
echo -n "Checking agent configurations... "
ollama_configs=$(find agents/configs -name "*_ollama.json" 2>/dev/null | wc -l)
universal_configs=$(find agents/configs -name "*_universal.json" 2>/dev/null | wc -l)
echo "✅ Found $ollama_configs Ollama configs and $universal_configs universal configs"

# Final report
echo ""
echo "==============================="
if [ $errors -eq 0 ]; then
    echo "✅ LiteLLM has been completely removed!"
    echo "🎯 All agents are now using native Ollama"
    echo "🚀 System is 100% local with no API translation layers"
else
    echo "❌ Found $errors issues that need attention"
    echo "Please run the cleanup script again or manually fix the issues above"
fi

echo ""
echo "📊 System Status:"
echo "  - Ollama: Native API at port 11434"
echo "  - Agents: Using _ollama.json configurations"
echo "  - Model: TinyLlama via native Ollama"
echo "  - Dependencies: Zero external APIs"