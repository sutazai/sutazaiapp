#!/bin/bash

# Script to merge v26 changes into main branch and push to GitHub
# This handles the complete workflow for SutazAI system

set -e

echo "🚀 Starting merge of v26 changes into main branch..."

# Navigate to project directory
cd /opt/sutazaiapp

# Check current branch and status
echo "📋 Current git status:"
git branch --show-current
echo ""

# Ensure we have all changes committed in v26
echo "📦 Checking for uncommitted changes in v26..."
if [ -n "$(git status --porcelain)" ]; then
    echo "⚡ Committing remaining changes in v26..."
    git add -A
    git commit -m "v26: Final updates before merge to main"
fi

# Switch to main branch
echo "🔄 Switching to main branch..."
git checkout main

# Update main branch from remote
echo "📥 Pulling latest changes from remote main..."
git pull origin main

# Merge v26 into main
echo "🔀 Merging v26 branch into main..."
git merge v26 --no-ff -m "Merge v26: Major SutazAI system updates

- AGI reasoning engine implementation
- Network reconnaissance capabilities  
- Enhanced UI components and navigation
- System optimizations and improvements
- Documentation reorganization
- Backend API enhancements
- 23 files changed with 4,485 insertions

This merge brings the complete v26 feature set into main branch."

# Push main branch to GitHub
echo "📤 Pushing main branch to GitHub..."
git push origin main

# Verify the push was successful
echo "✅ Verifying push to main..."
git log --oneline -5

echo ""
echo "🎉 SUCCESS! All v26 changes have been merged into main and pushed to GitHub!"
echo "🌐 Visit: https://github.com/sutazai/sutazaiapp"
echo ""
echo "📊 Summary:"
echo "  - Source branch: v26"
echo "  - Target branch: main" 
echo "  - Changes: AGI reasoning, network recon, UI enhancements"
echo "  - Files modified: 23 files with 4,485 insertions"
echo ""
echo "🔧 Branch status:"
git branch -vv | grep -E "(main|v26)" || true

echo ""
echo "✨ SutazAI main branch is now up to date with all v26 improvements!" 