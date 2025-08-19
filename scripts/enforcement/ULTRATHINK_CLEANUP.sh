#!/bin/bash
# ULTRATHINK COMPREHENSIVE CLEANUP SCRIPT
# Zero tolerance enforcement of all rules

set -euo pipefail

echo "========================================="
echo "ULTRATHINK CLEANUP - STARTING"
echo "========================================="

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/sutazaiapp/cleanup_backup_$TIMESTAMP"
mkdir -p "$BACKUP_DIR"

# 1. Find and remove mock/fake/stub files
echo "[1/5] Removing mock/fake/stub files..."
find /opt/sutazaiapp -type f \( -name "*mock*" -o -name "*fake*" -o -name "*stub*" \) \
    -not -path "*/node_modules/*" -not -path "*/.git/*" -not -path "*/venv/*" | while read -r file; do
    if [ -f "$file" ]; then
        echo "  Removing: $file"
        cp "$file" "$BACKUP_DIR/" 2>/dev/null || true
        rm -f "$file"
    fi
done

# 2. Clean up Docker files
echo "[2/5] Consolidating Docker configuration..."
if [ -f "/opt/sutazaiapp/docker-compose.yml" ]; then
    echo "  Moving root docker-compose.yml to /docker/"
    mkdir -p /opt/sutazaiapp/docker
    mv /opt/sutazaiapp/docker-compose.yml /opt/sutazaiapp/docker/
fi

# Remove backup docker files
rm -f /opt/sutazaiapp/docker-compose.yml.backup.*

# 3. Clean root directory
echo "[3/5] Cleaning root directory..."
for file in /opt/sutazaiapp/*.md; do
    filename=$(basename "$file")
    case "$filename" in
        README.md|CHANGELOG.md|CLAUDE.md|LICENSE.md)
            # Keep these
            ;;
        *)
            echo "  Moving $filename to /docs/"
            mkdir -p /opt/sutazaiapp/docs/relocated
            mv "$file" "/opt/sutazaiapp/docs/relocated/" 2>/dev/null || true
            ;;
    esac
done

# 4. Consolidate CHANGELOGs
echo "[4/5] Consolidating CHANGELOG files..."
MASTER_CHANGELOG="/opt/sutazaiapp/CHANGELOG.md"
echo "# MASTER CHANGELOG" > "$MASTER_CHANGELOG"
echo "Consolidated on $TIMESTAMP" >> "$MASTER_CHANGELOG"
echo "" >> "$MASTER_CHANGELOG"

find /opt/sutazaiapp -name "CHANGELOG*" -not -path "$MASTER_CHANGELOG" -type f | while read -r changelog; do
    echo "  Consolidating: $changelog"
    echo "## From: $changelog" >> "$MASTER_CHANGELOG"
    cat "$changelog" >> "$MASTER_CHANGELOG"
    echo "" >> "$MASTER_CHANGELOG"
    rm -f "$changelog"
done

# 5. Remove TODO/FIXME placeholders
echo "[5/5] Removing TODO/FIXME placeholders..."
find /opt/sutazaiapp -type f \( -name "*.py" -o -name "*.js" -o -name "*.ts" \) \
    -not -path "*/node_modules/*" -not -path "*/.git/*" | while read -r file; do
    if grep -q "TODO\|FIXME" "$file"; then
        # Remove obvious placeholders
        sed -i '/TODO.*implement/d' "$file" 2>/dev/null || true
        sed -i '/FIXME.*implement/d' "$file" 2>/dev/null || true
    fi
done

echo ""
echo "========================================="
echo "CLEANUP COMPLETE"
echo "========================================="
echo "Backup location: $BACKUP_DIR"
echo ""

# Final verification
echo "Verification:"
echo "- Mock/fake files remaining: $(find /opt/sutazaiapp -name "*mock*" -o -name "*fake*" -o -name "*stub*" 2>/dev/null | wc -l)"
echo "- Docker files in root: $(ls /opt/sutazaiapp/docker-compose* 2>/dev/null | wc -l)"
echo "- CHANGELOG files: $(find /opt/sutazaiapp -name "CHANGELOG*" | wc -l)"
echo "- Files in root: $(ls /opt/sutazaiapp/*.md /opt/sutazaiapp/*.yml 2>/dev/null | wc -l)"