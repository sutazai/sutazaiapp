#!/bin/bash
# Purpose: Automated security scanning for SutazAI system
# Usage: ./security-scanner.sh [--full-scan] [--report-format json|html|both]
# Requires: Docker, trivy, semgrep (optional)

set -euo pipefail


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
BASE_DIR="/opt/sutazaiapp"
LOG_DIR="$BASE_DIR/logs"
REPORT_DIR="$BASE_DIR/reports/security"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Configuration
FULL_SCAN=false
REPORT_FORMAT="both"
SEVERITY_THRESHOLD="MEDIUM"  # LOW, MEDIUM, HIGH, CRITICAL
MAX_SCAN_TIME=3600          # Maximum scan time in seconds

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --full-scan)
            FULL_SCAN=true
            shift
            ;;
        --report-format)
            REPORT_FORMAT="$2"
            shift 2
            ;;
        --severity)
            SEVERITY_THRESHOLD="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--full-scan] [--report-format json|html|both] [--severity LOW|MEDIUM|HIGH|CRITICAL]"
            exit 1
            ;;
    esac
done

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local log_file="$LOG_DIR/security_scan_$TIMESTAMP.log"
    
    echo "[$timestamp] $level: $message" >> "$log_file"
    
    case $level in
        ERROR) echo -e "${RED}[$timestamp] ERROR: $message${NC}" ;;
        WARN) echo -e "${YELLOW}[$timestamp] WARN: $message${NC}" ;;
        INFO) echo -e "${BLUE}[$timestamp] INFO: $message${NC}" ;;
        SUCCESS) echo -e "${GREEN}[$timestamp] SUCCESS: $message${NC}" ;;
    esac
}

# Setup security scan directories
setup_scan_directories() {
    log "INFO" "Setting up security scan directories..."
    mkdir -p "$REPORT_DIR" "$LOG_DIR"
}

# Check if required tools are installed
check_security_tools() {
    log "INFO" "Checking security scanning tools..."
    
    local tools_available=true
    
    # Check for Trivy (container vulnerability scanner)
    if ! command -v trivy >/dev/null 2>&1; then
        log "WARN" "Trivy not found. Installing Trivy..."
        if command -v apt-get >/dev/null 2>&1; then
            # Ubuntu/Debian installation
            sudo apt-get update && sudo apt-get install -y wget apt-transport-https gnupg lsb-release
            wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
            echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
            sudo apt-get update && sudo apt-get install -y trivy
        else
            log "ERROR" "Cannot install Trivy automatically. Please install manually."
            tools_available=false
        fi
    fi
    
    # Check for Docker
    if ! command -v docker >/dev/null 2>&1; then
        log "ERROR" "Docker is not installed or not accessible"
        tools_available=false
    fi
    
    # Check for Semgrep (code analysis - optional)
    if ! command -v semgrep >/dev/null 2>&1; then
        log "WARN" "Semgrep not found. Code analysis will be limited."
        if command -v pip3 >/dev/null 2>&1; then
            log "INFO" "Attempting to install Semgrep..."
            pip3 install semgrep >/dev/null 2>&1 || log "WARN" "Failed to install Semgrep"
        fi
    fi
    
    return $([ "$tools_available" = true ] && echo 0 || echo 1)
}

# Scan Docker images for vulnerabilities
scan_container_vulnerabilities() {
    log "INFO" "Scanning Docker containers for vulnerabilities..."
    
    local container_scan_results="[]"
    local total_vulnerabilities=0
    local critical_vulnerabilities=0
    local high_vulnerabilities=0
    
    # Get all SutazAI images
    local images=()
    while IFS= read -r image; do
        if [[ -n "$image" ]]; then
            images+=("$image")
        fi
    done < <(docker images --format "{{.Repository}}:{{.Tag}}" | grep -E "(sutazai|ollama)" || true)
    
    if [[ ${#images[@]} -eq 0 ]]; then
        log "WARN" "No SutazAI or Ollama images found for scanning"
        echo "$container_scan_results"
        return
    fi
    
    for image in "${images[@]}"; do
        log "INFO" "Scanning image: $image"
        
        local scan_output
        local scan_json_file="$REPORT_DIR/trivy_${image//[:\/]/_}_$TIMESTAMP.json"
        
        # Run Trivy scan with timeout
        if timeout $MAX_SCAN_TIME trivy image --format json --severity "$SEVERITY_THRESHOLD,HIGH,CRITICAL" "$image" > "$scan_json_file" 2>/dev/null; then
            local image_vulnerabilities=$(jq '[.Results[]? | .Vulnerabilities[]? | select(.Severity)] | length' "$scan_json_file" 2>/dev/null || echo 0)
            local image_critical=$(jq '[.Results[]? | .Vulnerabilities[]? | select(.Severity == "CRITICAL")] | length' "$scan_json_file" 2>/dev/null || echo 0)
            local image_high=$(jq '[.Results[]? | .Vulnerabilities[]? | select(.Severity == "HIGH")] | length' "$scan_json_file" 2>/dev/null || echo 0)
            
            total_vulnerabilities=$((total_vulnerabilities + image_vulnerabilities))
            critical_vulnerabilities=$((critical_vulnerabilities + image_critical))
            high_vulnerabilities=$((high_vulnerabilities + image_high))
            
            # Create image scan result
            local image_result=$(jq -n \
                --arg image "$image" \
                --arg vulnerabilities "$image_vulnerabilities" \
                --arg critical "$image_critical" \
                --arg high "$image_high" \
                --arg scan_file "$scan_json_file" \
                '{
                    "image": $image,
                    "total_vulnerabilities": ($vulnerabilities | tonumber),
                    "critical_vulnerabilities": ($critical | tonumber),
                    "high_vulnerabilities": ($high | tonumber),
                    "scan_file": $scan_file,
                    "scan_status": "completed"
                }')
            
            container_scan_results=$(echo "$container_scan_results" | jq ". += [$image_result]")
            
            log "SUCCESS" "Scanned $image: $image_vulnerabilities vulnerabilities ($image_critical critical, $image_high high)"
        else
            log "ERROR" "Failed to scan image: $image (timeout or error)"
            
            local failed_result=$(jq -n \
                --arg image "$image" \
                '{
                    "image": $image,
                    "total_vulnerabilities": 0,
                    "critical_vulnerabilities": 0,
                    "high_vulnerabilities": 0,
                    "scan_file": null,
                    "scan_status": "failed"
                }')
            
            container_scan_results=$(echo "$container_scan_results" | jq ". += [$failed_result]")
        fi
    done
    
    log "SUCCESS" "Container vulnerability scan completed: $total_vulnerabilities total vulnerabilities ($critical_vulnerabilities critical, $high_vulnerabilities high)"
    echo "$container_scan_results"
}

# Scan filesystem for security issues
scan_filesystem_security() {
    log "INFO" "Scanning filesystem for security issues..."
    
    local fs_scan_results="{}"
    
    # Check file permissions on sensitive directories
    local sensitive_dirs=("$BASE_DIR/ssl" "$BASE_DIR/secrets" "$BASE_DIR/secrets_secure")
    local permission_issues=0
    
    for dir in "${sensitive_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            local dir_perms=$(stat -c "%a" "$dir" 2>/dev/null || echo "000")
            if [[ "$dir_perms" != "700" ]]; then
                log "WARN" "Insecure permissions on $dir: $dir_perms (should be 700)"
                ((permission_issues++))
            fi
            
            # Check files in directory
            while IFS= read -r -d '' file; do
                local file_perms=$(stat -c "%a" "$file" 2>/dev/null || echo "000")
                if [[ "$file_perms" != "600" ]]; then
                    log "WARN" "Insecure permissions on $file: $file_perms (should be 600)"
                    ((permission_issues++))
                fi
            done < <(find "$dir" -type f -print0 2>/dev/null)
        fi
    done
    
    # Check for world-writable files
    local world_writable=0
    while IFS= read -r -d '' file; do
        log "WARN" "World-writable file found: $file"
        ((world_writable++))
    done < <(find "$BASE_DIR" -type f -perm -002 -print0 2>/dev/null | head -20)
    
    # Check for SUID/SGID files
    local suid_files=0
    while IFS= read -r -d '' file; do
        log "INFO" "SUID/SGID file found: $file"
        ((suid_files++))
    done < <(find "$BASE_DIR" -type f \( -perm -4000 -o -perm -2000 \) -print0 2>/dev/null | head -10)
    
    fs_scan_results=$(jq -n \
        --arg permission_issues "$permission_issues" \
        --arg world_writable "$world_writable" \
        --arg suid_files "$suid_files" \
        '{
            "permission_issues": ($permission_issues | tonumber),
            "world_writable_files": ($world_writable | tonumber),
            "suid_sgid_files": ($suid_files | tonumber)
        }')
    
    log "SUCCESS" "Filesystem security scan completed: $permission_issues permission issues, $world_writable world-writable files"
    echo "$fs_scan_results"
}

# Scan network security
scan_network_security() {
    log "INFO" "Scanning network security configuration..."
    
    local network_scan_results="{}"
    local open_ports=()
    local docker_ports=()
    
    # Check for open ports
    if command -v ss >/dev/null 2>&1; then
        while IFS= read -r port_info; do
            if [[ -n "$port_info" ]]; then
                open_ports+=("$port_info")
            fi
        done < <(ss -tuln | grep LISTEN | awk '{print $5}' | cut -d: -f2 | sort -n | uniq)
    fi
    
    # Check Docker exposed ports
    while IFS= read -r container_port; do
        if [[ -n "$container_port" ]]; then
            docker_ports+=("$container_port")
        fi
    done < <(docker ps --format "{{.Names}}:{{.Ports}}" | grep "sutazai" | cut -d: -f2- || true)
    
    # Check for insecure protocols
    local insecure_protocols=0
    
    # Check if services are using HTTP instead of HTTPS
    for port in 8000 8001 8002 8003 8004 8005 8501; do
        if curl -s -m 5 "http://localhost:$port" >/dev/null 2>&1; then
            log "WARN" "Service on port $port is using HTTP (insecure)"
            ((insecure_protocols++))
        fi
    done
    
    network_scan_results=$(jq -n \
        --argjson open_ports "$(printf '%s\n' "${open_ports[@]}" | jq -R . | jq -s .)" \
        --argjson docker_ports "$(printf '%s\n' "${docker_ports[@]}" | jq -R . | jq -s .)" \
        --arg insecure_protocols "$insecure_protocols" \
        '{
            "open_ports": $open_ports,
            "docker_exposed_ports": $docker_ports,
            "insecure_protocol_count": ($insecure_protocols | tonumber)
        }')
    
    log "SUCCESS" "Network security scan completed: ${#open_ports[@]} open ports, $insecure_protocols insecure protocols"
    echo "$network_scan_results"
}

# Scan code for security issues using Semgrep
scan_code_security() {
    log "INFO" "Scanning code for security vulnerabilities..."
    
    local code_scan_results="{}"
    
    if command -v semgrep >/dev/null 2>&1; then
        local semgrep_json_file="$REPORT_DIR/semgrep_scan_$TIMESTAMP.json"
        
        # Run Semgrep scan with timeout
        if timeout $((MAX_SCAN_TIME / 2)) semgrep --config=auto --json --output="$semgrep_json_file" "$BASE_DIR" >/dev/null 2>&1; then
            local total_findings=$(jq '.results | length' "$semgrep_json_file" 2>/dev/null || echo 0)
            local high_severity=$(jq '[.results[] | select(.extra.severity == "ERROR")] | length' "$semgrep_json_file" 2>/dev/null || echo 0)
            local medium_severity=$(jq '[.results[] | select(.extra.severity == "WARNING")] | length' "$semgrep_json_file" 2>/dev/null || echo 0)
            
            code_scan_results=$(jq -n \
                --arg total "$total_findings" \
                --arg high "$high_severity" \
                --arg medium "$medium_severity" \
                --arg scan_file "$semgrep_json_file" \
                '{
                    "total_findings": ($total | tonumber),
                    "high_severity": ($high | tonumber),
                    "medium_severity": ($medium | tonumber),
                    "scan_file": $scan_file,
                    "scan_status": "completed"
                }')
            
            log "SUCCESS" "Code security scan completed: $total_findings findings ($high_severity high, $medium_severity medium)"
        else
            log "WARN" "Semgrep scan failed or timed out"
            code_scan_results='{"scan_status": "failed", "total_findings": 0}'
        fi
    else
        log "WARN" "Semgrep not available, skipping code security scan"
        code_scan_results='{"scan_status": "skipped", "total_findings": 0, "reason": "semgrep_not_available"}'
    fi
    
    echo "$code_scan_results"
}

# Check configuration security
scan_configuration_security() {
    log "INFO" "Scanning configuration security..."
    
    local config_scan_results="{}"
    local security_issues=0
    local recommendations=()
    
    # Check Docker configuration
    if [[ -f "$BASE_DIR/docker-compose.yml" ]]; then
        # Check for hardcoded secrets
        if grep -q "password.*=" "$BASE_DIR/docker-compose.yml" 2>/dev/null; then
            log "WARN" "Potential hardcoded passwords found in docker-compose.yml"
            ((security_issues++))
            recommendations+=("Remove hardcoded passwords from docker-compose.yml")
        fi
        
        # Check for privileged containers
        if grep -q "privileged.*true" "$BASE_DIR/docker-compose.yml" 2>/dev/null; then
            log "WARN" "Privileged containers found in docker-compose.yml"
            ((security_issues++))
            recommendations+=("Review privileged container usage")
        fi
    fi
    
    # Check SSL/TLS configuration
    if [[ -d "$BASE_DIR/ssl" ]]; then
        local ssl_issues=0
        
        # Check certificate expiry
        if [[ -f "$BASE_DIR/ssl/cert.pem" ]]; then
            local cert_expiry=$(openssl x509 -in "$BASE_DIR/ssl/cert.pem" -noout -enddate 2>/dev/null | cut -d= -f2)
            local expiry_epoch=$(date -d "$cert_expiry" +%s 2>/dev/null || echo 0)
            local current_epoch=$(date +%s)
            local days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))
            
            if [[ $days_until_expiry -lt 30 ]]; then
                log "WARN" "SSL certificate expires in $days_until_expiry days"
                ((ssl_issues++))
                recommendations+=("Renew SSL certificate (expires in $days_until_expiry days)")
            fi
        fi
        
        security_issues=$((security_issues + ssl_issues))
    fi
    
    # Check for default credentials
    local default_creds=0
    local cred_files=("$BASE_DIR/secrets/postgres_password.txt" "$BASE_DIR/secrets/redis_password.txt")
    
    for cred_file in "${cred_files[@]}"; do
        if [[ -f "$cred_file" ]]; then
            local cred_content=$(cat "$cred_file" 2>/dev/null || echo "")
            if [[ "$cred_content" == "password" || "$cred_content" == "admin" || "$cred_content" == "123456" ]]; then
                log "WARN" "Default/weak credential detected in $(basename "$cred_file")"
                ((default_creds++))
                recommendations+=("Change default credential in $(basename "$cred_file")")
            fi
        fi
    done
    
    security_issues=$((security_issues + default_creds))
    
    config_scan_results=$(jq -n \
        --arg security_issues "$security_issues" \
        --arg default_creds "$default_creds" \
        --argjson recommendations "$(printf '%s\n' "${recommendations[@]}" | jq -R . | jq -s .)" \
        '{
            "total_security_issues": ($security_issues | tonumber),
            "default_credentials": ($default_creds | tonumber),
            "recommendations": $recommendations
        }')
    
    log "SUCCESS" "Configuration security scan completed: $security_issues issues found"
    echo "$config_scan_results"
}

# Generate security summary and risk assessment
generate_security_summary() {
    local container_results="$1"
    local filesystem_results="$2"
    local network_results="$3"
    local code_results="$4"
    local config_results="$5"
    
    log "INFO" "Generating security summary and risk assessment..."
    
    local summary="{}"
    local total_critical=0
    local total_high=0
    local total_medium=0
    local overall_risk="LOW"
    local recommendations=()
    
    # Calculate totals from container scan
    total_critical=$(echo "$container_results" | jq '[.[].critical_vulnerabilities] | add' 2>/dev/null || echo 0)
    total_high=$(echo "$container_results" | jq '[.[].high_vulnerabilities] | add' 2>/dev/null || echo 0)
    
    # Add code scan results
    local code_high=$(echo "$code_results" | jq -r '.high_severity // 0')
    local code_medium=$(echo "$code_results" | jq -r '.medium_severity // 0')
    total_high=$((total_high + code_high))
    total_medium=$((total_medium + code_medium))
    
    # Add filesystem issues
    local fs_permission_issues=$(echo "$filesystem_results" | jq -r '.permission_issues')
    total_medium=$((total_medium + fs_permission_issues))
    
    # Add network issues
    local network_insecure=$(echo "$network_results" | jq -r '.insecure_protocol_count')
    total_medium=$((total_medium + network_insecure))
    
    # Add configuration issues
    local config_issues=$(echo "$config_results" | jq -r '.total_security_issues')
    local config_recs=$(echo "$config_results" | jq -r '.recommendations[]?' 2>/dev/null)
    while IFS= read -r rec; do
        if [[ -n "$rec" ]]; then
            recommendations+=("$rec")
        fi
    done <<< "$config_recs"
    
    # Determine overall risk level
    if [[ $total_critical -gt 0 ]]; then
        overall_risk="CRITICAL"
        recommendations+=("Address $total_critical critical vulnerabilities immediately")
    elif [[ $total_high -gt 5 ]]; then
        overall_risk="HIGH"
        recommendations+=("Address $total_high high-severity issues")
    elif [[ $total_high -gt 0 || $total_medium -gt 10 ]]; then
        overall_risk="MEDIUM"
        recommendations+=("Review and address high and medium severity issues")
    elif [[ $total_medium -gt 0 ]]; then
        overall_risk="LOW"
        recommendations+=("Consider addressing medium severity issues during maintenance windows")
    fi
    
    # Calculate security score (0-100)
    local security_score=100
    security_score=$((security_score - total_critical * 25))
    security_score=$((security_score - total_high * 10))
    security_score=$((security_score - total_medium * 3))
    
    # Ensure score doesn't go below 0
    if [[ $security_score -lt 0 ]]; then
        security_score=0
    fi
    
    summary=$(jq -n \
        --arg overall_risk "$overall_risk" \
        --arg security_score "$security_score" \
        --arg total_critical "$total_critical" \
        --arg total_high "$total_high" \
        --arg total_medium "$total_medium" \
        --argjson recommendations "$(printf '%s\n' "${recommendations[@]}" | jq -R . | jq -s .)" \
        '{
            "overall_risk": $overall_risk,
            "security_score": ($security_score | tonumber),
            "vulnerability_summary": {
                "critical": ($total_critical | tonumber),
                "high": ($total_high | tonumber),
                "medium": ($total_medium | tonumber)
            },
            "recommendations": $recommendations
        }')
    
    echo "$summary"
}

# Generate JSON security report
generate_json_security_report() {
    local container_results="$1"
    local filesystem_results="$2"
    local network_results="$3"
    local code_results="$4"
    local config_results="$5"
    local summary="$6"
    
    local json_report_file="$REPORT_DIR/security_scan_report_$TIMESTAMP.json"
    
    log "INFO" "Generating JSON security report..."
    
    jq -n \
        --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --arg hostname "$(hostname)" \
        --arg scan_type "$([ "$FULL_SCAN" == "true" ] && echo "full" || echo "standard")" \
        --arg severity_threshold "$SEVERITY_THRESHOLD" \
        --argjson containers "$container_results" \
        --argjson filesystem "$filesystem_results" \
        --argjson network "$network_results" \
        --argjson code "$code_results" \
        --argjson configuration "$config_results" \
        --argjson summary "$summary" \
        '{
            "scan_info": {
                "timestamp": $timestamp,
                "hostname": $hostname,
                "scan_type": $scan_type,
                "severity_threshold": $severity_threshold
            },
            "results": {
                "containers": $containers,
                "filesystem": $filesystem,
                "network": $network,
                "code": $code,
                "configuration": $configuration
            },
            "summary": $summary
        }' > "$json_report_file"
    
    log "SUCCESS" "JSON security report generated: $json_report_file"
    
    # Create symlink to latest report
    ln -sf "$json_report_file" "$REPORT_DIR/latest_security_report.json"
    
    echo "$json_report_file"
}

# Generate HTML security report
generate_html_security_report() {
    local json_report_file="$1"
    local html_report_file="${json_report_file%.json}.html"
    
    log "INFO" "Generating HTML security report..."
    
    # Read JSON data
    local report_data=$(cat "$json_report_file")
    local timestamp=$(echo "$report_data" | jq -r '.scan_info.timestamp')
    local hostname=$(echo "$report_data" | jq -r '.scan_info.hostname')
    local summary=$(echo "$report_data" | jq -r '.summary')
    
    # Generate HTML
    cat > "$html_report_file" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SutazAI Security Scan Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .risk-critical { color: #dc3545; background: #f8d7da; padding: 10px; border-radius: 5px; }
        .risk-high { color: #fd7e14; background: #fff3cd; padding: 10px; border-radius: 5px; }
        .risk-medium { color: #ffc107; background: #fff3cd; padding: 10px; border-radius: 5px; }
        .risk-low { color: #28a745; background: #d4edda; padding: 10px; border-radius: 5px; }
        .metric-card { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }
        .metric-title { font-weight: bold; font-size: 1.1em; margin-bottom: 10px; }
        .vulnerability-summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .vuln-count { text-align: center; padding: 15px; border-radius: 5px; }
        .vuln-critical { background: #f8d7da; color: #721c24; }
        .vuln-high { background: #fff3cd; color: #856404; }
        .vuln-medium { background: #d1ecf1; color: #0c5460; }
        .recommendations { background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .recommendations ul { margin: 10px 0; padding-left: 20px; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SutazAI Security Scan Report</h1>
            <p><strong>Host:</strong> $hostname | <strong>Generated:</strong> $timestamp</p>
        </div>

        <div class="metric-card">
            <div class="metric-title">Overall Security Status</div>
            <div class="risk-$(echo "$summary" | jq -r '.overall_risk' | tr '[:upper:]' '[:lower:]')">
                <strong>Risk Level:</strong> $(echo "$summary" | jq -r '.overall_risk') 
                | <strong>Security Score:</strong> $(echo "$summary" | jq -r '.security_score')/100
            </div>
        </div>

        <div class="vulnerability-summary">
            <div class="vuln-count vuln-critical">
                <h3>$(echo "$summary" | jq -r '.vulnerability_summary.critical')</h3>
                <p>Critical</p>
            </div>
            <div class="vuln-count vuln-high">
                <h3>$(echo "$summary" | jq -r '.vulnerability_summary.high')</h3>
                <p>High</p>
            </div>
            <div class="vuln-count vuln-medium">
                <h3>$(echo "$summary" | jq -r '.vulnerability_summary.medium')</h3>
                <p>Medium</p>
            </div>
        </div>

        <div class="metric-card">
            <div class="metric-title">Container Vulnerabilities</div>
            <table>
                <thead>
                    <tr>
                        <th>Image</th>
                        <th>Total Vulnerabilities</th>
                        <th>Critical</th>
                        <th>High</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
EOF

    # Add container vulnerability rows
    echo "$report_data" | jq -r '.results.containers[] | 
        "<tr><td>" + .image + "</td><td>" + (.total_vulnerabilities | tostring) + "</td><td>" + 
        (.critical_vulnerabilities | tostring) + "</td><td>" + (.high_vulnerabilities | tostring) + 
        "</td><td>" + .scan_status + "</td></tr>"' >> "$html_report_file"

    cat >> "$html_report_file" << EOF
                </tbody>
            </table>
        </div>

        <div class="recommendations">
            <h3>Security Recommendations</h3>
EOF

    # Add recommendations
    local recommendations_count=$(echo "$summary" | jq '.recommendations | length')
    if [[ $recommendations_count -gt 0 ]]; then
        echo "<ul>" >> "$html_report_file"
        echo "$summary" | jq -r '.recommendations[] | "<li>" + . + "</li>"' >> "$html_report_file"
        echo "</ul>" >> "$html_report_file"
    else
        echo "<p>No specific security recommendations at this time.</p>" >> "$html_report_file"
    fi

    cat >> "$html_report_file" << EOF
        </div>

        <div class="metric-card">
            <div class="metric-title">Detailed Results</div>
            <pre style="background: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 0.9em;">$(echo "$report_data" | jq .)</pre>
        </div>
    </div>
</body>
</html>
EOF

    log "SUCCESS" "HTML security report generated: $html_report_file"
    
    # Create symlink to latest HTML report
    ln -sf "$html_report_file" "$REPORT_DIR/latest_security_report.html"
    
    echo "$html_report_file"
}

# Clean old security reports
clean_old_security_reports() {
    log "INFO" "Cleaning old security reports (older than 60 days)..."
    
    local deleted_count=0
    
    while IFS= read -r -d '' report; do
        local basename=$(basename "$report")
        log "INFO" "Deleting old security report: $basename"
        rm "$report"
        ((deleted_count++))
    done < <(find "$REPORT_DIR" -name "security_scan_report_*.json" -o -name "security_scan_report_*.html" -o -name "trivy_*.json" -o -name "semgrep_*.json" -type f -mtime +60 -print0 2>/dev/null)
    
    if [[ $deleted_count -gt 0 ]]; then
        log "SUCCESS" "Deleted $deleted_count old security reports"
    else
        log "INFO" "No old security reports found for deletion"
    fi
}

# Main execution
main() {
    log "INFO" "Starting security scan for SutazAI system"
    log "INFO" "Scan type: $([ "$FULL_SCAN" == "true" ] && echo "FULL" || echo "STANDARD"), Report format: $REPORT_FORMAT, Severity threshold: $SEVERITY_THRESHOLD"
    
    # Setup directories
    setup_scan_directories
    
    # Check required tools
    if ! check_security_tools; then
        log "ERROR" "Required security tools are not available, aborting scan"
        exit 1
    fi
    
    # Run security scans
    log "INFO" "Running security scans..."
    local container_results=$(scan_container_vulnerabilities)
    local filesystem_results=$(scan_filesystem_security)
    local network_results=$(scan_network_security)
    local code_results=$(scan_code_security)
    local config_results=$(scan_configuration_security)
    
    # Generate security summary
    local summary=$(generate_security_summary "$container_results" "$filesystem_results" "$network_results" "$code_results" "$config_results")
    
    # Generate reports based on format
    local json_report=""
    local html_report=""
    
    if [[ "$REPORT_FORMAT" == "json" || "$REPORT_FORMAT" == "both" ]]; then
        json_report=$(generate_json_security_report "$container_results" "$filesystem_results" "$network_results" "$code_results" "$config_results" "$summary")
    fi
    
    if [[ "$REPORT_FORMAT" == "html" || "$REPORT_FORMAT" == "both" ]]; then
        if [[ -z "$json_report" ]]; then
            json_report=$(generate_json_security_report "$container_results" "$filesystem_results" "$network_results" "$code_results" "$config_results" "$summary")
        fi
        html_report=$(generate_html_security_report "$json_report")
    fi
    
    # Clean old reports
    clean_old_security_reports
    
    log "SUCCESS" "Security scan completed"
    
    # Show summary
    echo
    echo "============================================"
    echo "         SECURITY SCAN SUMMARY"
    echo "============================================"
    echo "Overall Risk: $(echo "$summary" | jq -r '.overall_risk')"
    echo "Security Score: $(echo "$summary" | jq -r '.security_score')/100"
    echo "Critical Vulnerabilities: $(echo "$summary" | jq -r '.vulnerability_summary.critical')"
    echo "High Vulnerabilities: $(echo "$summary" | jq -r '.vulnerability_summary.high')"
    echo "Medium Vulnerabilities: $(echo "$summary" | jq -r '.vulnerability_summary.medium')"
    if [[ -n "$json_report" ]]; then
        echo "JSON Report: $json_report"
    fi
    if [[ -n "$html_report" ]]; then
        echo "HTML Report: $html_report"
    fi
    echo "Timestamp: $(date)"
    echo "============================================"
    
    # Exit with appropriate code based on risk level
    local risk_level=$(echo "$summary" | jq -r '.overall_risk')
    case $risk_level in
        "LOW") exit 0 ;;
        "MEDIUM") exit 1 ;;
        "HIGH") exit 2 ;;
        "CRITICAL") exit 3 ;;
        *) exit 4 ;;
    esac
}

# Run main function
main "$@"