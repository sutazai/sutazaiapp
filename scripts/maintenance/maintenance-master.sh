#!/bin/bash
# Master Maintenance Script - Orchestrates all maintenance operations
# Usage: ./maintenance-master.sh [cleanup|optimization|validation|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] MAINT: $1"
}

OPERATION="${1:-all}"

log "Starting maintenance operation: $OPERATION"

case "$OPERATION" in
    "cleanup")
        for script in "$SCRIPT_DIR/cleanup"/*.sh; do
            [[ -x "$script" ]] && "$script"
        done
        ;;
    "optimization")
        for script in "$SCRIPT_DIR/optimization"/*.sh; do
            [[ -x "$script" ]] && "$script"
        done
        ;;
    "validation")
        for script in "$SCRIPT_DIR/validation"/*.sh; do
            [[ -x "$script" ]] && "$script"
        done
        ;;
    "all"|*)
        for subdir in cleanup optimization validation; do
            for script in "$SCRIPT_DIR/$subdir"/*.sh; do
                [[ -x "$script" ]] && "$script"
            done
        done
        ;;
esac

log "Maintenance operation complete: $OPERATION"
