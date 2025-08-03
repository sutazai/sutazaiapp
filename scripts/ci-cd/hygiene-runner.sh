#!/bin/bash
# Universal CI/CD Hygiene Runner Script
# Purpose: Provides a consistent interface for running hygiene checks across different CI/CD platforms
# Usage: ./hygiene-runner.sh --rules "1,2,3" --priority "critical" --output "report.json" [--dry-run]
# Requirements: Python 3.8+, git, jq

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_VENV="${PROJECT_ROOT}/venv"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${PROJECT_ROOT}/logs/hygiene-runner-${TIMESTAMP}.log"

# Default values
RULES=""
PRIORITY="medium"
OUTPUT_FILE="hygiene-report.json"
DRY_RUN=false
CONFIG_FILE=""
VERBOSE=false
CI_PLATFORM="unknown"
ANALYSIS_ONLY=false
MAX_PARALLEL_JOBS=4

# Logging functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[${timestamp}] [${level}] ${message}" | tee -a "${LOG_FILE}"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_debug() { [[ "${VERBOSE}" == "true" ]] && log "DEBUG" "$@" || true; }

# Help function
show_help() {
    cat << EOF
Universal CI/CD Hygiene Runner Script

Usage: ${0##*/} [OPTIONS]

Options:
    -r, --rules RULES           Comma-separated list of rule numbers to check (e.g., "1,2,3")
    -p, --priority PRIORITY     Priority level: critical, high, medium, low (default: medium)
    -o, --output FILE          Output file for the report (default: hygiene-report.json)
    -c, --config FILE          Configuration file with additional settings
    -d, --dry-run              Run in dry-run mode (no changes will be made)
    -a, --analysis-only        Run analysis only, skip enforcement
    -v, --verbose              Enable verbose output
    -j, --jobs NUMBER          Maximum parallel jobs (default: 4)
    -h, --help                 Show this help message

Environment Variables:
    CI_PLATFORM               Auto-detected or set manually (github, gitlab, jenkins)
    PROJECT_ROOT              Project root directory (default: auto-detected)
    PYTHON_VENV               Python virtual environment path

Examples:
    # Run critical rules analysis
    ${0##*/} --rules "1,2,3" --priority critical --output critical-report.json

    # Run comprehensive check in dry-run mode
    ${0##*/} --priority high --dry-run --verbose

    # Run specific rules with custom config
    ${0##*/} --rules "7,12,13" --config custom-config.json
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -r|--rules)
                RULES="$2"
                shift 2
                ;;
            -p|--priority)
                PRIORITY="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_FILE="$2"
                shift 2
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -a|--analysis-only)
                ANALYSIS_ONLY=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -j|--jobs)
                MAX_PARALLEL_JOBS="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Detect CI platform
detect_ci_platform() {
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        CI_PLATFORM="github"
    elif [[ -n "${GITLAB_CI:-}" ]]; then
        CI_PLATFORM="gitlab"
    elif [[ -n "${JENKINS_HOME:-}" ]]; then
        CI_PLATFORM="jenkins"
    elif [[ -n "${CIRCLECI:-}" ]]; then
        CI_PLATFORM="circleci"
    elif [[ -n "${TRAVIS:-}" ]]; then
        CI_PLATFORM="travis"
    else
        CI_PLATFORM="local"
    fi
    
    log_info "Detected CI platform: ${CI_PLATFORM}"
}

# Setup Python environment
setup_python_env() {
    log_info "Setting up Python environment..."
    
    # Check if virtual environment exists
    if [[ ! -d "${PYTHON_VENV}" ]]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv "${PYTHON_VENV}"
    fi
    
    # Activate virtual environment
    source "${PYTHON_VENV}/bin/activate"
    
    # Upgrade pip
    pip install --quiet --upgrade pip
    
    # Install required packages
    local requirements=(
        "pyyaml>=6.0"
        "click>=8.0"
        "gitpython>=3.1"
        "pathlib>=1.0"
        "jsonschema>=4.0"
    )
    
    for req in "${requirements[@]}"; do
        pip install --quiet "${req}"
    done
    
    # Install project-specific requirements if available
    if [[ -f "${PROJECT_ROOT}/requirements/hygiene.txt" ]]; then
        pip install --quiet -r "${PROJECT_ROOT}/requirements/hygiene.txt"
    fi
}

# Validate environment
validate_environment() {
    log_info "Validating environment..."
    
    # Check required commands
    local required_commands=("python3" "git" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Required command not found: $cmd"
            exit 1
        fi
    done
    
    # Check project structure
    if [[ ! -f "${PROJECT_ROOT}/CLAUDE.md" ]]; then
        log_error "CLAUDE.md not found. Are you in the correct project directory?"
        exit 1
    fi
    
    # Ensure log directory exists
    mkdir -p "$(dirname "${LOG_FILE}")"
    
    # Check git repository
    if ! git -C "${PROJECT_ROOT}" rev-parse --git-dir > /dev/null 2>&1; then
        log_warn "Not in a git repository. Some features may be limited."
    fi
}

# Load configuration
load_config() {
    local config="{}"
    
    # Load default configuration
    if [[ -f "${PROJECT_ROOT}/.hygiene-config.json" ]]; then
        config=$(jq -s '.[0] * .[1]' <(echo "$config") "${PROJECT_ROOT}/.hygiene-config.json")
    fi
    
    # Load user-specified configuration
    if [[ -n "${CONFIG_FILE}" ]] && [[ -f "${CONFIG_FILE}" ]]; then
        config=$(jq -s '.[0] * .[1]' <(echo "$config") "${CONFIG_FILE}")
    fi
    
    # Apply configuration
    if [[ -n "$(echo "$config" | jq -r '.rules // empty')" ]] && [[ -z "${RULES}" ]]; then
        RULES=$(echo "$config" | jq -r '.rules | join(",")')
    fi
    
    if [[ -n "$(echo "$config" | jq -r '.priority // empty')" ]] && [[ "${PRIORITY}" == "medium" ]]; then
        PRIORITY=$(echo "$config" | jq -r '.priority')
    fi
    
    log_debug "Configuration loaded: $(echo "$config" | jq -c .)"
}

# Get rules to check
get_rules_to_check() {
    if [[ -n "${RULES}" ]]; then
        echo "${RULES}"
    else
        # Default rules based on priority
        case "${PRIORITY}" in
            critical)
                echo "1,2,3,13"
                ;;
            high)
                echo "1,2,3,4,5,8,11,12,13"
                ;;
            medium)
                echo "1,2,3,4,5,6,7,8,9,10,11,12,13"
                ;;
            low|*)
                echo "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"
                ;;
        esac
    fi
}

# Run hygiene analysis
run_hygiene_analysis() {
    local rules="$1"
    local output_file="$2"
    
    log_info "Running hygiene analysis for rules: ${rules}"
    
    # Create temporary directory for intermediate results
    local temp_dir="${PROJECT_ROOT}/.hygiene-temp-${TIMESTAMP}"
    mkdir -p "${temp_dir}"
    
    # Split rules and run in parallel
    IFS=',' read -ra RULE_ARRAY <<< "$rules"
    local pids=()
    local job_count=0
    
    for rule in "${RULE_ARRAY[@]}"; do
        # Wait if we've reached max parallel jobs
        while [[ ${job_count} -ge ${MAX_PARALLEL_JOBS} ]]; do
            wait -n
            job_count=$((job_count - 1))
        done
        
        (
            log_debug "Analyzing rule ${rule}..."
            python3 "${SCRIPT_DIR}/analyze-rule.py" \
                --rule "${rule}" \
                --project-root "${PROJECT_ROOT}" \
                --output "${temp_dir}/rule-${rule}.json" \
                ${DRY_RUN:+--dry-run} \
                ${VERBOSE:+--verbose}
        ) &
        
        pids+=($!)
        job_count=$((job_count + 1))
    done
    
    # Wait for all analysis jobs to complete
    for pid in "${pids[@]}"; do
        wait "${pid}" || log_warn "Analysis job ${pid} failed"
    done
    
    # Consolidate results
    log_info "Consolidating analysis results..."
    python3 "${SCRIPT_DIR}/consolidate-analysis.py" \
        --input-dir "${temp_dir}" \
        --output "${output_file}" \
        --priority "${PRIORITY}" \
        --ci-platform "${CI_PLATFORM}"
    
    # Cleanup temporary directory
    rm -rf "${temp_dir}"
}

# Run agent enforcement
run_agent_enforcement() {
    local analysis_file="$1"
    
    if [[ "${ANALYSIS_ONLY}" == "true" ]]; then
        log_info "Skipping enforcement (analysis-only mode)"
        return 0
    fi
    
    log_info "Running agent enforcement based on analysis..."
    
    # Determine which agents to run based on violations
    local agents_to_run=$(python3 -c "
import json
with open('${analysis_file}') as f:
    data = json.load(f)
    violations = data.get('violations', {})
    
    agents = []
    if any(r in violations for r in ['13']):
        agents.append('garbage-collector')
    if any(r in violations for r in ['7', '12']):
        agents.append('script-consolidator')
    if any(r in violations for r in ['11']):
        agents.append('docker-optimizer')
    if any(r in violations for r in ['1', '2', '3']):
        agents.append('code-auditor')
    if any(r in violations for r in ['6', '15']):
        agents.append('documentation-manager')
    
    print(' '.join(agents))
")
    
    if [[ -z "${agents_to_run}" ]]; then
        log_info "No agents needed for enforcement"
        return 0
    fi
    
    # Run agents
    for agent in ${agents_to_run}; do
        log_info "Running ${agent} agent..."
        
        python3 "${PROJECT_ROOT}/scripts/agents/hygiene-agent-orchestrator.py" \
            --agent "${agent}" \
            --analysis-file "${analysis_file}" \
            --output "${OUTPUT_FILE%.json}-enforcement-${agent}.json" \
            ${DRY_RUN:+--dry-run} \
            ${VERBOSE:+--verbose} || log_warn "Agent ${agent} failed"
    done
}

# Generate CI-specific output
generate_ci_output() {
    local report_file="$1"
    
    case "${CI_PLATFORM}" in
        github)
            # Generate GitHub Actions output
            python3 -c "
import json
with open('${report_file}') as f:
    data = json.load(f)
    summary = data.get('summary', {})
    
    # Set outputs
    print(f\"::set-output name=hygiene_score::{summary.get('hygiene_score', 0)}\")
    print(f\"::set-output name=critical_violations::{summary.get('critical_violations', 0)}\")
    print(f\"::set-output name=high_violations::{summary.get('high_violations', 0)}\")
    
    # Add annotations
    for violation in data.get('violations', {}).get('critical', []):
        print(f\"::error file={violation['file']},line={violation.get('line', 1)}::{violation['message']}\")
"
            ;;
            
        gitlab)
            # Generate GitLab CI output
            python3 -c "
import json
with open('${report_file}') as f:
    data = json.load(f)
    summary = data.get('summary', {})
    
    # Generate JUnit XML for test results
    junit_file = '${report_file%.json}.xml'
    # ... generate JUnit XML ...
"
            ;;
            
        jenkins)
            # Generate Jenkins properties file
            python3 -c "
import json
with open('${report_file}') as f:
    data = json.load(f)
    summary = data.get('summary', {})
    
    with open('hygiene.properties', 'w') as out:
        out.write(f\"hygiene_score={summary.get('hygiene_score', 0)}\\n\")
        out.write(f\"critical_violations={summary.get('critical_violations', 0)}\\n\")
        out.write(f\"high_violations={summary.get('high_violations', 0)}\\n\")
"
            ;;
    esac
}

# Main execution
main() {
    log_info "Starting hygiene runner..."
    
    # Validate environment
    validate_environment
    
    # Detect CI platform
    detect_ci_platform
    
    # Load configuration
    load_config
    
    # Setup Python environment
    setup_python_env
    
    # Get rules to check
    local rules_to_check=$(get_rules_to_check)
    log_info "Rules to check: ${rules_to_check}"
    
    # Run hygiene analysis
    run_hygiene_analysis "${rules_to_check}" "${OUTPUT_FILE}"
    
    # Run agent enforcement if needed
    run_agent_enforcement "${OUTPUT_FILE}"
    
    # Generate CI-specific output
    generate_ci_output "${OUTPUT_FILE}"
    
    # Check results and set exit code
    local exit_code=0
    if [[ -f "${OUTPUT_FILE}" ]]; then
        local critical_violations=$(jq -r '.summary.critical_violations // 0' "${OUTPUT_FILE}")
        if [[ "${critical_violations}" -gt 0 ]]; then
            log_error "Found ${critical_violations} critical violations"
            exit_code=1
        else
            log_info "No critical violations found"
        fi
    else
        log_error "Failed to generate report"
        exit_code=2
    fi
    
    log_info "Hygiene runner completed with exit code: ${exit_code}"
    return ${exit_code}
}

# Parse arguments and run
parse_args "$@"
main