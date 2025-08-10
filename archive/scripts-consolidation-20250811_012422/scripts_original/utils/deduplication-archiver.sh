#!/bin/bash
# SutazAI Deduplication Archive System
# Archives all original files before deduplication for rollback capability
# Author: DevOps Manager - Deduplication Operation
# Date: August 10, 2025

set -euo pipefail

# Configuration

# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ARCHIVE_DIR="${PROJECT_ROOT}/archive/deduplication-backup-$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${PROJECT_ROOT}/logs/deduplication_archive_$(date +%Y%m%d_%H%M%S).log"

# Logging
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "$LOG_FILE" >&2
}

# Create archive structure
create_archive_structure() {
    log "Creating archive directory structure..."
    
    mkdir -p "$ARCHIVE_DIR"/{dockerfiles,scripts,templates,configs}
    mkdir -p "$ARCHIVE_DIR"/dockerfiles/{agents,base,services,individual}
    mkdir -p "$ARCHIVE_DIR"/scripts/{deployment,monitoring,maintenance,utils}
    
    log "Archive directory created: $ARCHIVE_DIR"
}

# Archive duplicate Dockerfiles
archive_dockerfiles() {
    log "Archiving duplicate Dockerfiles..."
    
    local archived_count=0
    
    # Archive all Dockerfiles from /docker/agents/ (exact duplicates)
    if [[ -d "${PROJECT_ROOT}/docker/agents" ]]; then
        log "Archiving agent Dockerfiles..."
        cp -r "${PROJECT_ROOT}/docker/agents/" "${ARCHIVE_DIR}/dockerfiles/agents/"
        archived_count=$((archived_count + $(find "${PROJECT_ROOT}/docker/agents" -name "Dockerfile*" -type f | wc -l)))
    fi
    
    # Archive base image Dockerfiles that will be consolidated
    if [[ -d "${PROJECT_ROOT}/docker/base" ]]; then
        log "Archiving base image Dockerfiles..."
        cp -r "${PROJECT_ROOT}/docker/base/" "${ARCHIVE_DIR}/dockerfiles/base/"
        archived_count=$((archived_count + $(find "${PROJECT_ROOT}/docker/base" -name "Dockerfile*" -type f | wc -l)))
    fi
    
    # Archive individual service Dockerfiles that have duplicates
    local duplicate_dockerfiles=(
        "agents/ai_agent_orchestrator/Dockerfile"
        "agents/jarvis-hardware-resource-optimizer/Dockerfile"
        "docker/agentgpt/Dockerfile"
        "docker/autogpt/Dockerfile"
        "docker/crewai/Dockerfile"
        "docker/langchain-agents/Dockerfile"
        "docker/llamaindex/Dockerfile"
        # Add more as identified
    )
    
    for dockerfile in "${duplicate_dockerfiles[@]}"; do
        if [[ -f "${PROJECT_ROOT}/${dockerfile}" ]]; then
            local dest_dir="${ARCHIVE_DIR}/dockerfiles/individual/$(dirname "$dockerfile")"
            mkdir -p "$dest_dir"
            cp "${PROJECT_ROOT}/${dockerfile}" "$dest_dir/"
            log "Archived: $dockerfile"
            ((archived_count++))
        fi
    done
    
    log "Archived $archived_count Dockerfiles"
}

# Archive duplicate scripts
archive_scripts() {
    log "Archiving duplicate scripts..."
    
    local archived_count=0
    
    # Archive deployment scripts (47 variations)
    local deployment_scripts=(
        "scripts/deployment/deploy.sh"
        "scripts/deployment/build-all-images.sh"
        "scripts/deployment/start-minimal.sh"
        "scripts/deployment/fast_start.sh"
        "scripts/deployment/ultimate-deployment-master.py"
        "scripts/deployment/prepare-20-agents.py"
        # Archive ALL scripts in deployment directory since we're consolidating
    )
    
    if [[ -d "${PROJECT_ROOT}/scripts/deployment" ]]; then
        log "Archiving deployment scripts..."
        find "${PROJECT_ROOT}/scripts/deployment" -type f \( -name "*.sh" -o -name "*.py" \) | while read -r script; do
            local rel_path=$(realpath --relative-to="${PROJECT_ROOT}" "$script")
            local dest_dir="${ARCHIVE_DIR}/scripts/$(dirname "$rel_path")"
            mkdir -p "$dest_dir"
            cp "$script" "$dest_dir/"
            ((archived_count++))
        done
    fi
    
    # Archive monitoring scripts (38 variations)
    if [[ -d "${PROJECT_ROOT}/scripts/monitoring" ]]; then
        log "Archiving monitoring scripts..."
        find "${PROJECT_ROOT}/scripts/monitoring" -type f \( -name "*.sh" -o -name "*.py" \) | while read -r script; do
            local rel_path=$(realpath --relative-to="${PROJECT_ROOT}" "$script")
            local dest_dir="${ARCHIVE_DIR}/scripts/$(dirname "$rel_path")"
            mkdir -p "$dest_dir"
            cp "$script" "$dest_dir/"
            ((archived_count++))
        done
    fi
    
    # Archive maintenance scripts (15 variations)
    if [[ -d "${PROJECT_ROOT}/scripts/maintenance" ]]; then
        log "Archiving maintenance scripts..."
        find "${PROJECT_ROOT}/scripts/maintenance" -type f \( -name "*.sh" -o -name "*.py" \) | while read -r script; do
            local rel_path=$(realpath --relative-to="${PROJECT_ROOT}" "$script")
            local dest_dir="${ARCHIVE_DIR}/scripts/$(dirname "$rel_path")"
            mkdir -p "$dest_dir"
            cp "$script" "$dest_dir/"
            ((archived_count++))
        done
    fi
    
    log "Archived $archived_count scripts"
}

# Create archive manifest
create_manifest() {
    log "Creating archive manifest..."
    
    local manifest_file="${ARCHIVE_DIR}/ARCHIVE_MANIFEST.md"
    
    cat > "$manifest_file" << EOF
# SutazAI Deduplication Archive Manifest

**Archive Date:** $(date)  
**Operation:** Infrastructure Deduplication  
**Operator:** DevOps Manager - Claude Code  

## Archive Purpose
This archive contains all original files before the massive infrastructure deduplication operation.
It enables complete rollback if any issues are discovered after consolidation.

## Archive Contents

### Dockerfiles
- **agents/**: All agent service Dockerfiles (exact duplicates)
- **base/**: Base image Dockerfiles being consolidated
- **individual/**: Individual service Dockerfiles with duplicates

### Scripts  
- **deployment/**: All deployment script variations (47+ files)
- **monitoring/**: All monitoring script variations (38+ files)
- **maintenance/**: All maintenance script variations (15+ files)

## Statistics
- **Total Dockerfiles Archived:** $(find "$ARCHIVE_DIR/dockerfiles" -name "Dockerfile*" -type f | wc -l)
- **Total Scripts Archived:** $(find "$ARCHIVE_DIR/scripts" -name "*.sh" -o -name "*.py" | wc -l)
- **Archive Size:** $(du -sh "$ARCHIVE_DIR" | cut -f1)

## Rollback Instructions

### Complete Rollback
\`\`\`bash
# Stop all services
docker-compose down

# Restore all files from archive
cd $PROJECT_ROOT
rsync -av "${ARCHIVE_DIR}/dockerfiles/" docker/
rsync -av "${ARCHIVE_DIR}/scripts/" scripts/

# Restart services
docker-compose up -d
\`\`\`

### Partial Rollback (Dockerfiles only)
\`\`\`bash
# Restore specific Dockerfile categories
rsync -av "${ARCHIVE_DIR}/dockerfiles/agents/" docker/agents/
rsync -av "${ARCHIVE_DIR}/dockerfiles/base/" docker/base/
\`\`\`

### Partial Rollback (Scripts only)
\`\`\`bash  
# Restore specific script categories
rsync -av "${ARCHIVE_DIR}/scripts/deployment/" scripts/deployment/
rsync -av "${ARCHIVE_DIR}/scripts/monitoring/" scripts/monitoring/
\`\`\`

## Validation
After rollback, run these commands to validate:
\`\`\`bash
# Test Docker builds
docker-compose build --parallel

# Test script execution
bash scripts/deployment/deploy.sh --dry-run minimal

# Verify service health
python3 scripts/monitoring/monitoring-master.py --mode check
\`\`\`

## Archive Integrity
- **MD5 Checksums:** See checksums.md5 file
- **Git Commit:** $(git rev-parse HEAD)
- **Git Branch:** $(git branch --show-current)

## Notes
This archive is automatically created before deduplication.
Keep this archive until the new system has been validated in production for at least 30 days.

EOF

    # Create checksums
    log "Generating checksums..."
    find "$ARCHIVE_DIR" -type f -not -name "*.md5" -exec md5sum {} \; > "${ARCHIVE_DIR}/checksums.md5"
    
    log "Archive manifest created: $manifest_file"
}

# Compress archive
compress_archive() {
    log "Compressing archive for storage efficiency..."
    
    local archive_parent=$(dirname "$ARCHIVE_DIR")
    local archive_name=$(basename "$ARCHIVE_DIR")
    
    cd "$archive_parent"
    
    # Create compressed tar
    local compressed_file="${archive_name}.tar.gz"
    tar -czf "$compressed_file" "$archive_name"
    
    # Verify compression
    if [[ -f "$compressed_file" ]]; then
        local original_size=$(du -sh "$archive_name" | cut -f1)
        local compressed_size=$(du -sh "$compressed_file" | cut -f1)
        
        log "Archive compressed: $original_size → $compressed_size"
        log "Compressed archive: ${archive_parent}/${compressed_file}"
        
        # Keep both compressed and uncompressed for safety
        return 0
    else
        log_error "Archive compression failed"
        return 1
    fi
}

# Main execution
main() {
    log "Starting deduplication archive process"
    log "Project root: $PROJECT_ROOT"
    log "Archive location: $ARCHIVE_DIR"
    
    # Create logs directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Check if we're in the correct directory
    if [[ ! -f "${PROJECT_ROOT}/CLAUDE.md" ]]; then
        log_error "Not in SutazAI project root directory"
        exit 1
    fi
    
    # Execute archive steps
    create_archive_structure
    archive_dockerfiles
    archive_scripts
    create_manifest
    compress_archive
    
    # Final summary
    local total_size=$(du -sh "$ARCHIVE_DIR" | cut -f1)
    local file_count=$(find "$ARCHIVE_DIR" -type f | wc -l)
    
    log "Archive process completed successfully"
    log "Archive size: $total_size"
    log "Files archived: $file_count"
    log "Archive location: $ARCHIVE_DIR"
    log "Manifest: ${ARCHIVE_DIR}/ARCHIVE_MANIFEST.md"
    
    echo
    echo "=== DEDUPLICATION ARCHIVE COMPLETE ==="
    echo "Archive: $ARCHIVE_DIR"
    echo "Size: $total_size"
    echo "Files: $file_count"
    echo
    echo "Ready to proceed with deduplication!"
    echo "Rollback instructions in: ${ARCHIVE_DIR}/ARCHIVE_MANIFEST.md"
}

# Execute main function
main "$@"