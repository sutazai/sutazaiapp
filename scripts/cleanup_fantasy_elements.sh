#!/bin/bash

echo "🧹 Cleaning up fantasy elements from SutazAI codebase..."

# Remove fantasy-themed documentation files
echo "📄 Removing fantasy documentation..."
find docs/ -name "*.md" -type f | grep -E "(automation|advanced automation|coordinator|system_state|advanced|processing|genetic|evolution)" -i | while read file; do
    echo "Removing: $file"
    rm -f "$file"
done

# Remove automation/advanced automation architecture docs
rm -rf docs/system/architecture/ADVANCED_AGI_ASI_ARCHITECTURE.md
rm -rf docs/system/architecture/AGI_*
rm -rf docs/system/architecture/*automation*
rm -rf docs/project-docs/*automation*
rm -rf docs/*BRAIN*
rm -rf docs/*state management*

# Remove fantasy agent documentation
echo "🤖 Cleaning fantasy agent docs..."
find .claude/agents/ -name "*.md" | grep -E "(advanced|processing-architecture|genetic|evolution|coordinator|agi-system|system_state)" -i | while read file; do
    echo "Removing agent doc: $file"
    rm -f "$file"
done

# Remove fantasy-related Python files
echo "🐍 Checking Python files for fantasy imports..."
find . -name "*.py" -type f | while read file; do
    if grep -l -E "(system_state|self_monitoring|sentient|advanced_|genetic_algorithm|evolution_|coordinator_computer|agi_system)" "$file" > /dev/null 2>&1; then
        echo "File contains fantasy elements: $file"
        # Don't auto-remove Python files, just report them
    fi
done

# Clean up configuration files
echo "⚙️ Cleaning configuration files..."
find config/ -name "*.yaml" -o -name "*.yml" -o -name "*.json" | while read file; do
    if grep -l -E "(system_state|advanced|genetic|evolution|coordinator_computer|agi_system)" "$file" > /dev/null 2>&1; then
        echo "Config contains fantasy elements: $file"
    fi
done

# Remove fantasy workflow files
echo "🔄 Removing fantasy workflows..."
rm -f workflows/*system_state*.py
rm -f workflows/*advanced*.py
rm -f workflows/*genetic*.py
rm -f workflows/*coordinator*.py

# Clean up archive
echo "📦 Cleaning archive..."
rm -rf archive/agi-docs/
rm -rf archive/system_state/
rm -rf archive/advanced/

# Update pyproject.toml to remove fantasy dependencies
echo "📦 Checking dependencies..."
if grep -E "(advanced|system_state|genetic-algorithm|coordinator-interface)" pyproject.toml > /dev/null 2>&1; then
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
1. Removed automation/advanced automation documentation
2. Removed system_state-related files
3. Removed advanced computing references
4. Removed genetic algorithm agents
5. Removed coordinator-computer interface references
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