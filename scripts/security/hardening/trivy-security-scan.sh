#!/bin/bash
# Purpose: Comprehensive container security scan using Trivy
# Usage: ./trivy-security-scan.sh [--output-format json|table] [--severity HIGH,CRITICAL|MEDIUM,HIGH,CRITICAL|ALL]
# Requirements: Trivy installed, Docker running

set -euo pipefail

# Default values

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

OUTPUT_FORMAT="${1:-table}"
SEVERITY="${2:-HIGH,CRITICAL}"
SCAN_DATE=$(date +"%Y%m%d_%H%M%S")
REPORT_DIR="/opt/sutazaiapp/security-reports"
SUMMARY_FILE="${REPORT_DIR}/trivy_scan_summary_${SCAN_DATE}.md"

# Create reports directory
mkdir -p "${REPORT_DIR}"

echo "Starting comprehensive container security scan with Trivy..."
echo "Scan date: $(date)"
echo "Output format: ${OUTPUT_FORMAT}"
echo "Severity levels: ${SEVERITY}"
echo ""

# Initialize summary file
cat > "${SUMMARY_FILE}" << EOF
# Container Security Scan Report
**Date**: $(date)
**Scanner**: Trivy
**Severity Filter**: ${SEVERITY}

## Executive Summary
This report provides a comprehensive security analysis of all SutazAI container images and configurations.

## Scan Results Summary
EOF

# Function to scan container image
scan_image() {
    local image_name="$1"
    local clean_name=$(echo "$image_name" | tr ':/' '_')
    local output_file="${REPORT_DIR}/trivy_${clean_name}_${SCAN_DATE}.${OUTPUT_FORMAT}"
    
    echo "Scanning image: $image_name"
    
    if [[ "$OUTPUT_FORMAT" == "json" ]]; then
        trivy image "$image_name" --severity "$SEVERITY" --format json --quiet > "$output_file" 2>/dev/null || {
            echo "  ❌ Failed to scan $image_name" >&2
            return 1
        }
        
        # Extract vulnerability count from JSON
        local vuln_count=$(jq '[.Results[]?.Vulnerabilities // empty] | length' "$output_file" 2>/dev/null || echo "0")
        echo "  📊 Found $vuln_count vulnerabilities"
        
        # Add to summary
        echo "- **$image_name**: $vuln_count vulnerabilities ([detailed report](./${output_file##*/}))" >> "${SUMMARY_FILE}"
        
    else
        trivy image "$image_name" --severity "$SEVERITY" --format table --quiet > "$output_file" 2>/dev/null || {
            echo "  ❌ Failed to scan $image_name" >&2
            return 1
        }
        
        # Extract vulnerability count from table output
        local vuln_count=$(grep -o "Total: [0-9]*" "$output_file" | head -1 | grep -o "[0-9]*" || echo "0")
        echo "  📊 Found $vuln_count vulnerabilities"
        
        # Add to summary
        echo "- **$image_name**: $vuln_count vulnerabilities ([detailed report](./${output_file##*/}))" >> "${SUMMARY_FILE}"
    fi
}

# Function to scan Dockerfile
scan_dockerfile() {
    local dockerfile_path="$1"
    local clean_name=$(echo "$dockerfile_path" | tr '/' '_')
    local output_file="${REPORT_DIR}/trivy_config_${clean_name}_${SCAN_DATE}.${OUTPUT_FORMAT}"
    
    echo "Scanning Dockerfile: $dockerfile_path"
    
    if [[ "$OUTPUT_FORMAT" == "json" ]]; then
        trivy config "$dockerfile_path" --severity "$SEVERITY" --format json --quiet > "$output_file" 2>/dev/null || {
            echo "  ❌ Failed to scan $dockerfile_path" >&2
            return 1
        }
        
        # Extract misconfigurations count from JSON
        local misconfig_count=$(jq '[.Results[]?.Misconfigurations // empty] | length' "$output_file" 2>/dev/null || echo "0")
        echo "  📊 Found $misconfig_count misconfigurations"
        
    else
        trivy config "$dockerfile_path" --severity "$SEVERITY" --format table --quiet > "$output_file" 2>/dev/null || {
            echo "  ❌ Failed to scan $dockerfile_path" >&2
            return 1
        }
        
        # Extract misconfigurations count from table output
        local misconfig_count=$(grep -c "MEDIUM\|HIGH\|CRITICAL" "$output_file" 2>/dev/null || echo "0")
        echo "  📊 Found $misconfig_count misconfigurations"
    fi
}

# Get all Docker images related to sutazai
echo "## Container Images" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

docker images | grep -E "(sutazai|ollama|postgres|redis|neo4j|chromadb|qdrant|prom|grafana)" | while read -r repo tag image_id created size; do
    if [[ "$repo" != "REPOSITORY" ]]; then
        scan_image "${repo}:${tag}"
    fi
done

# Scan key Dockerfiles
echo "" >> "${SUMMARY_FILE}"
echo "## Dockerfile Configuration Scans" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# Find and scan active Dockerfiles (not in archive)
find . -name "Dockerfile*" -not -path "./archive/*" -not -path "./security_backup*" -type f | head -20 | while read -r dockerfile; do
    scan_dockerfile "$dockerfile"
done

# Scan docker-compose files for security issues
echo "" >> "${SUMMARY_FILE}"
echo "## Docker Compose Configuration Scans" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

find . -name "docker-compose*.yml" -not -path "./archive/*" -not -path "./security_backup*" -type f | head -10 | while read -r compose_file; do
    local clean_name=$(echo "$compose_file" | tr '/' '_')
    local output_file="${REPORT_DIR}/trivy_compose_${clean_name}_${SCAN_DATE}.${OUTPUT_FORMAT}"
    
    echo "Scanning compose file: $compose_file"
    
    trivy config "$compose_file" --severity "$SEVERITY" --format "$OUTPUT_FORMAT" --quiet > "$output_file" 2>/dev/null || {
        echo "  ❌ Failed to scan $compose_file" >&2
        continue
    }
    
    if [[ "$OUTPUT_FORMAT" == "table" ]]; then
        local misconfig_count=$(grep -c "MEDIUM\|HIGH\|CRITICAL" "$output_file" 2>/dev/null || echo "0")
        echo "  📊 Found $misconfig_count misconfigurations"
        echo "- **$compose_file**: $misconfig_count misconfigurations ([detailed report](./${output_file##*/}))" >> "${SUMMARY_FILE}"
    fi
done

# Add recommendations to summary
cat >> "${SUMMARY_FILE}" << EOF

## Security Recommendations

### Immediate Actions Required (Critical/High Severity)
1. **Update Base Images**: Switch to latest LTS versions with security patches
2. **Remove Root Users**: Implement non-root user configurations in all containers
3. **Add Security Contexts**: Configure proper security contexts and capabilities
4. **Update Dependencies**: Upgrade vulnerable packages to fixed versions

### Medium Priority Improvements
1. **Minimize Attack Surface**: Remove unnecessary packages and tools
2. **Implement Health Checks**: Add proper container health monitoring
3. **Network Security**: Configure proper network policies and isolation
4. **Secrets Management**: Implement proper secrets handling

### Container Hardening Best Practices
1. Use distroless or minimal base images
2. Enable read-only root filesystems where possible
3. Drop unnecessary Linux capabilities
4. Implement proper resource limits
5. Use multi-stage builds to reduce image size
6. Enable security scanning in CI/CD pipeline

## Files Generated
All detailed scan results are available in the \`security-reports/\` directory:
- Individual container scan reports
- Dockerfile configuration analysis
- Docker Compose security analysis
- This summary report

**Next Steps**: Review individual reports and implement the recommended security fixes.
EOF

echo ""
echo "✅ Container security scan completed!"
echo "📁 Reports saved to: ${REPORT_DIR}"
echo "📋 Summary report: ${SUMMARY_FILE}"
echo ""
echo "Quick stats:"
echo "- Total scan files generated: $(ls -1 ${REPORT_DIR}/*${SCAN_DATE}* | wc -l)"
echo "- Report directory size: $(du -sh ${REPORT_DIR} | cut -f1)"
echo ""
echo "Review the summary report for prioritized security recommendations."