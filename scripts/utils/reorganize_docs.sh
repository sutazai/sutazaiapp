#!/bin/bash

# Script to reorganize all documentation files to /opt/sutazaiapp/docs
# Excludes .claude directory

set -e

DOCS_DIR="/opt/sutazaiapp/docs"
BASE_DIR="/opt/sutazaiapp"

echo "Starting documentation reorganization..."

# Move system-level documentation
echo "Moving system documentation..."
find "$BASE_DIR" -maxdepth 1 -type f \( -name "*.md" -o -name "*.txt" \) ! -path "*/.claude/*" ! -path "*/.git/*" | while read -r file; do
    filename=$(basename "$file")
    case "$filename" in
        README*|GUIDE*|DOCUMENTATION*)
            mv "$file" "$DOCS_DIR/guides/" 2>/dev/null || true
            ;;
        AGENT*|AI_*)
            mv "$file" "$DOCS_DIR/agents/" 2>/dev/null || true
            ;;
        DEPLOYMENT*|DEPLOY*)
            mv "$file" "$DOCS_DIR/deployment/" 2>/dev/null || true
            ;;
        SECURITY*|AUDIT*)
            mv "$file" "$DOCS_DIR/security/" 2>/dev/null || true
            ;;
        SYSTEM*|ARCHITECTURE*|BRAIN*)
            mv "$file" "$DOCS_DIR/architecture/" 2>/dev/null || true
            ;;
        *)
            mv "$file" "$DOCS_DIR/system/" 2>/dev/null || true
            ;;
    esac
done

# Move requirements files
echo "Moving requirements files..."
find "$BASE_DIR" -name "requirements*.txt" ! -path "*/.claude/*" ! -path "*/.git/*" ! -path "*/node_modules/*" | while read -r file; do
    dir=$(dirname "$file")
    dir_name=$(basename "$dir")
    mkdir -p "$DOCS_DIR/requirements/$dir_name"
    mv "$file" "$DOCS_DIR/requirements/$dir_name/" 2>/dev/null || true
done

# Move script documentation
echo "Moving script documentation..."
find "$BASE_DIR/scripts" -type f \( -name "*.md" -o -name "*.txt" \) ! -path "*/.claude/*" | while read -r file; do
    mv "$file" "$DOCS_DIR/scripts/" 2>/dev/null || true
done

# Move backend documentation
echo "Moving backend documentation..."
find "$BASE_DIR/backend" -type f \( -name "*.md" -o -name "*.txt" \) ! -name "requirements*.txt" ! -path "*/.claude/*" | while read -r file; do
    mv "$file" "$DOCS_DIR/api/" 2>/dev/null || true
done

# Move demo and workflow documentation
echo "Moving demo and workflow documentation..."
find "$BASE_DIR/demos" "$BASE_DIR/workflows" -type f \( -name "*.md" -o -name "*.txt" \) ! -path "*/.claude/*" 2>/dev/null | while read -r file; do
    mv "$file" "$DOCS_DIR/guides/" 2>/dev/null || true
done

# Move coordinator architecture documentation
echo "Moving coordinator architecture documentation..."
find "$BASE_DIR/coordinator" -type f \( -name "*.md" -o -name "*.txt" \) ! -name "requirements*.txt" ! -path "*/.claude/*" 2>/dev/null | while read -r file; do
    mv "$file" "$DOCS_DIR/architecture/" 2>/dev/null || true
done

# Move any remaining documentation files
echo "Moving remaining documentation..."
find "$BASE_DIR" -type f \( -name "*.md" -o -name "*.txt" \) ! -path "*/.claude/*" ! -path "*/.git/*" ! -path "*/node_modules/*" ! -path "$DOCS_DIR/*" | while read -r file; do
    # Skip secrets directory
    if [[ "$file" =~ /secrets/ ]]; then
        continue
    fi
    mv "$file" "$DOCS_DIR/misc/" 2>/dev/null || true
done

# Move docx and pdf files
echo "Moving docx and pdf files..."
find "$BASE_DIR" -type f \( -name "*.docx" -o -name "*.pdf" \) ! -path "*/.claude/*" ! -path "*/.git/*" ! -path "*/node_modules/*" ! -path "$DOCS_DIR/*" | while read -r file; do
    mv "$file" "$DOCS_DIR/misc/" 2>/dev/null || true
done

echo "Documentation reorganization complete!"
echo "Total files in docs directory:"
find "$DOCS_DIR" -type f | wc -l

echo -e "\nDocumentation structure:"
tree -d "$DOCS_DIR" 2>/dev/null || ls -la "$DOCS_DIR/"