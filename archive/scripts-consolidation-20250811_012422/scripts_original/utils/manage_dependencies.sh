#!/bin/bash

# Strict error handling
set -euo pipefail

# Systematically manage all dependencies with version control and hash verification

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

PYTHON_REQUIRED="3.11"
VIRTUALENV_VERSION="20.16.7"
VIRTUALENV_HASH="sha256:330618a7970b6596b79c4b45a60a51bfbaa5fcf0d77ac7e0697b1307eafde145"

# Import common utilities
source "$(dirname "$0")/common.sh"

validate_python() {
    if ! python3 -c "import sys; assert sys.version_info >= (3,11)"; then
        log_error "Python 3.11+ required"
        exit 1
    fi
}

install_system_deps() {
    apt-get update
    apt-get install -y \
        python3.11=3.11.8-1~22.04 \
        python3.11-venv=3.11.8-1~22.04 \
        python3.11-dev=3.11.8-1~22.04
}

secure_install_virtualenv() {
    pip install "virtualenv==${VIRTUALENV_VERSION}" \
        --require-hashes \
        --hash="${VIRTUALENV_HASH}"
}

verify_requirements() {
    python -m pip hash -r requirements.txt > requirements.sha256 || {
        log_error "Dependency integrity check failed"
        exit 1
    }
}

main() {
    validate_python
    install_system_deps
    secure_install_virtualenv
    verify_requirements
}

main "$@"