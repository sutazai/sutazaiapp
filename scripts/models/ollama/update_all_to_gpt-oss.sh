#!/bin/bash

echo "🔄 Updating all model references to gpt-oss..."
echo "============================================"

# Update environment files
echo "📝 Updating environment files..."
for env_file in .env .env.agents .env.example .env.production .env.ollama .env.gpt-oss; do
    if [ -f "$env_file" ]; then
        echo "  - Updating $env_file"
        sed -i 's/gpt-oss/gpt-oss/g' "$env_file"
        sed -i 's/gpt-oss/gpt-oss/g' "$env_file"
        sed -i 's/gpt-oss/gpt-oss/g' "$env_file"
    fi
done

# Update YAML configuration files
echo "📄 Updating YAML configuration files..."
find . -name "*.yaml" -o -name "*.yml" | while read -r file; do
    if grep -q "gpt-oss" "$file" 2>/dev/null; then
        echo "  - Updating $file"
        sed -i 's/gpt-oss/gpt-oss/g' "$file"
        sed -i 's/gpt-oss/gpt-oss/g' "$file"
        sed -i 's/gpt-oss/gpt-oss/g' "$file"
    fi
done

# Update JSON configuration files
echo "📋 Updating JSON configuration files..."
find . -name "*.json" | while read -r file; do
    if grep -q "gpt-oss" "$file" 2>/dev/null; then
        echo "  - Updating $file"
        sed -i 's/gpt-oss/gpt-oss/g' "$file"
        sed -i 's/gpt-oss/gpt-oss/g' "$file"
        sed -i 's/gpt-oss/gpt-oss/g' "$file"
    fi
done

# Update Python files
echo "🐍 Updating Python files..."
find . -name "*.py" | while read -r file; do
    if grep -q "gpt-oss" "$file" 2>/dev/null; then
        echo "  - Updating $file"
        sed -i 's/gpt-oss/gpt-oss/g' "$file"
        sed -i 's/gpt-oss/gpt-oss/g' "$file"
        sed -i 's/gpt-oss/gpt-oss/g' "$file"
    fi
done

# Update shell scripts
echo "🐚 Updating shell scripts..."
find . -name "*.sh" | while read -r file; do
    if grep -q "gpt-oss" "$file" 2>/dev/null; then
        echo "  - Updating $file"
        sed -i 's/gpt-oss/gpt-oss/g' "$file"
        sed -i 's/gpt-oss/gpt-oss/g' "$file"
        sed -i 's/gpt-oss/gpt-oss/g' "$file"
    fi
done

# Update markdown documentation
echo "📚 Updating documentation..."
find . -name "*.md" | while read -r file; do
    if grep -q "gpt-oss" "$file" 2>/dev/null; then
        echo "  - Updating $file"
        sed -i 's/gpt-oss/gpt-oss/g' "$file"
        sed -i 's/gpt-oss/gpt-oss/g' "$file"
        sed -i 's/gpt-oss/gpt-oss/g' "$file"
    fi
done

# Update TOML files
echo "📦 Updating TOML files..."
find . -name "*.toml" | while read -r file; do
    if grep -q "gpt-oss" "$file" 2>/dev/null; then
        echo "  - Updating $file"
        sed -i 's/gpt-oss/gpt-oss/g' "$file"
        sed -i 's/gpt-oss/gpt-oss/g' "$file"
        sed -i 's/gpt-oss/gpt-oss/g' "$file"
    fi
done

echo ""
echo "✅ All model references have been updated to gpt-oss!"
echo ""
echo "📊 Summary of changes:"
echo "  - Environment files: Updated all DEFAULT_MODEL, LLM_MODEL, CHAT_MODEL references"
echo "  - Configuration files: Updated YAML, JSON, TOML files"
echo "  - Source code: Updated Python and Shell scripts"
echo "  - Documentation: Updated all markdown files"
echo ""
echo "🚀 The system is now configured to use gpt-oss as the default model everywhere"