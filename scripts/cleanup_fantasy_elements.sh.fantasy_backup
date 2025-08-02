#!/bin/bash

echo "🧹 Cleaning up fantasy elements from SutazAI codebase..."

# Remove fantasy-themed documentation files
echo "📄 Removing fantasy documentation..."
find docs/ -name "*.md" -type f | grep -E "(AGI|ASI|brain|consciousness|quantum|neural|genetic|evolution)" -i | while read file; do
    echo "Removing: $file"
    rm -f "$file"
done

# Remove AGI/ASI architecture docs
rm -rf docs/system/architecture/ADVANCED_AGI_ASI_ARCHITECTURE.md
rm -rf docs/system/architecture/AGI_*
rm -rf docs/system/architecture/*AGI*
rm -rf docs/project-docs/*AGI*
rm -rf docs/*BRAIN*
rm -rf docs/*CONSCIOUSNESS*

# Remove fantasy agent documentation
echo "🤖 Cleaning fantasy agent docs..."
find .claude/agents/ -name "*.md" | grep -E "(quantum|neural-architecture|genetic|evolution|brain|agi-system|consciousness)" -i | while read file; do
    echo "Removing agent doc: $file"
    rm -f "$file"
done

# Remove fantasy-related Python files
echo "🐍 Checking Python files for fantasy imports..."
find . -name "*.py" -type f | while read file; do
    if grep -l -E "(consciousness|self_aware|sentient|quantum_|genetic_algorithm|evolution_|brain_computer|agi_system)" "$file" > /dev/null 2>&1; then
        echo "File contains fantasy elements: $file"
        # Don't auto-remove Python files, just report them
    fi
done

# Clean up configuration files
echo "⚙️ Cleaning configuration files..."
find config/ -name "*.yaml" -o -name "*.yml" -o -name "*.json" | while read file; do
    if grep -l -E "(consciousness|quantum|genetic|evolution|brain_computer|agi_system)" "$file" > /dev/null 2>&1; then
        echo "Config contains fantasy elements: $file"
    fi
done

# Remove fantasy workflow files
echo "🔄 Removing fantasy workflows..."
rm -f workflows/*consciousness*.py
rm -f workflows/*quantum*.py
rm -f workflows/*genetic*.py
rm -f workflows/*brain*.py

# Clean up archive
echo "📦 Cleaning archive..."
rm -rf archive/agi-docs/
rm -rf archive/consciousness/
rm -rf archive/quantum/

# Update pyproject.toml to remove fantasy dependencies
echo "📦 Checking dependencies..."
if grep -E "(quantum|consciousness|genetic-algorithm|brain-interface)" pyproject.toml > /dev/null 2>&1; then
    echo "Found fantasy dependencies in pyproject.toml - please review manually"
fi

# Create a summary report
echo "📊 Creating cleanup report..."
cat > FANTASY_CLEANUP_REPORT.md << EOF
# Fantasy Elements Cleanup Report

Date: $(date)

## Summary
Removed fantasy and speculative elements from the SutazAI codebase to focus on practical, working implementation.

## Actions Taken
1. Removed AGI/ASI documentation
2. Removed consciousness-related files
3. Removed quantum computing references
4. Removed genetic algorithm agents
5. Removed brain-computer interface references
6. Cleaned up speculative architecture documents

## Remaining Tasks
- Review Python files for fantasy imports
- Update agent configurations
- Focus on working agents: deployment, testing, infrastructure, security
- Update documentation to reflect actual capabilities

## Working Components
- Ollama integration for local LLM
- Docker deployment
- Task automation workflows
- Practical AI agents for development tasks
EOF

echo "✅ Fantasy cleanup complete! See FANTASY_CLEANUP_REPORT.md for details."
echo "🎯 Focus is now on practical, working implementation with local models."