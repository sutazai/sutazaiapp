#!/bin/bash
# Common utilities for all scripts

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Error handling
handle_error() {
    echo -e "${RED}[ERROR]${NC} ${BOLD}$1${NC} at line $2" >&2
    exit 1
}

# Logging functions
log() {
    echo -e "$1"
}

log_info() {
    log "${BLUE}[INFO]${NC} $1"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    log "${YELLOW}[WARNING]${NC} $1"
}

# Command validation
require_root() {
    if [ "$EUID" -ne 0 ]; then
        handle_error "This script must be run as root" ${LINENO}
    fi
}

# Package verification
verify_package() {
    dpkg -l "$1" &> /dev/null || {
        handle_error "Package $1 not installed" ${LINENO}
    }
}