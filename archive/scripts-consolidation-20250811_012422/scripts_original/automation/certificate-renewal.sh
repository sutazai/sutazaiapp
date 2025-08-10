#!/bin/bash
# Purpose: Automated SSL certificate renewal and management for SutazAI system
# Usage: ./certificate-renewal.sh [--dry-run] [--force-renew]
# Requires: OpenSSL, Docker (if using containerized services)

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
SSL_DIR="$BASE_DIR/ssl"
LOG_DIR="$BASE_DIR/logs"
BACKUP_DIR="$BASE_DIR/backups/certificates"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Configuration
DRY_RUN=false
FORCE_RENEW=false
RENEWAL_THRESHOLD_DAYS=30  # Renew if certificate expires within this many days
CERT_VALIDITY_DAYS=365     # New certificate validity period
KEY_SIZE=2048              # RSA key size

# Certificate configuration
DOMAIN_NAME="${SUTAZAI_DOMAIN:-localhost}"
ORGANIZATION="${SUTAZAI_ORG:-SutazAI}"
ORGANIZATIONAL_UNIT="${SUTAZAI_OU:-IT Department}"
CITY="${SUTAZAI_CITY:-Local}"
STATE="${SUTAZAI_STATE:-Local}"
COUNTRY="${SUTAZAI_COUNTRY:-US}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force-renew)
            FORCE_RENEW=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--force-renew]"
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
    local log_file="$LOG_DIR/certificate_renewal_$TIMESTAMP.log"
    
    echo "[$timestamp] $level: $message" >> "$log_file"
    
    case $level in
        ERROR) echo -e "${RED}[$timestamp] ERROR: $message${NC}" ;;
        WARN) echo -e "${YELLOW}[$timestamp] WARN: $message${NC}" ;;
        INFO) echo -e "${BLUE}[$timestamp] INFO: $message${NC}" ;;
        SUCCESS) echo -e "${GREEN}[$timestamp] SUCCESS: $message${NC}" ;;
    esac
}

# Create necessary directories
setup_directories() {
    log "INFO" "Setting up certificate directories..."
    
    if [[ "$DRY_RUN" == "false" ]]; then
        mkdir -p "$SSL_DIR" "$LOG_DIR" "$BACKUP_DIR"
        
        # Set appropriate permissions for SSL directory
        chmod 700 "$SSL_DIR"
        chmod 700 "$BACKUP_DIR"
    else
        log "INFO" "[DRY RUN] Would create directories: $SSL_DIR, $BACKUP_DIR"
    fi
}

# Check if certificate exists and get its expiration date
check_certificate_expiry() {
    local cert_file="$SSL_DIR/cert.pem"
    
    if [[ ! -f "$cert_file" ]]; then
        log "WARN" "Certificate file not found: $cert_file"
        return 1
    fi
    
    # Get certificate expiration date
    local expiry_date=$(openssl x509 -in "$cert_file" -noout -enddate 2>/dev/null | cut -d= -f2)
    
    if [[ -z "$expiry_date" ]]; then
        log "ERROR" "Cannot read certificate expiration date"
        return 1
    fi
    
    # Convert to epoch time
    local expiry_epoch=$(date -d "$expiry_date" +%s 2>/dev/null || echo 0)
    local current_epoch=$(date +%s)
    local threshold_epoch=$((current_epoch + RENEWAL_THRESHOLD_DAYS * 24 * 3600))
    
    log "INFO" "Certificate expires on: $expiry_date"
    
    # Check if renewal is needed
    if [[ $expiry_epoch -le $threshold_epoch ]]; then
        local days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))
        log "WARN" "Certificate expires in $days_until_expiry days (threshold: $RENEWAL_THRESHOLD_DAYS days)"
        return 0  # Renewal needed
    else
        local days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))
        log "SUCCESS" "Certificate is valid for $days_until_expiry more days"
        return 1  # No renewal needed
    fi
}

# Backup existing certificates
backup_certificates() {
    log "INFO" "Backing up existing certificates..."
    
    local cert_file="$SSL_DIR/cert.pem"
    local key_file="$SSL_DIR/key.pem"
    local backup_subdir="$BACKUP_DIR/backup_$TIMESTAMP"
    
    if [[ ! -f "$cert_file" && ! -f "$key_file" ]]; then
        log "INFO" "No existing certificates to backup"
        return 0
    fi
    
    if [[ "$DRY_RUN" == "false" ]]; then
        mkdir -p "$backup_subdir"
        
        if [[ -f "$cert_file" ]]; then
            cp "$cert_file" "$backup_subdir/"
            log "SUCCESS" "Backed up certificate: $backup_subdir/cert.pem"
        fi
        
        if [[ -f "$key_file" ]]; then
            cp "$key_file" "$backup_subdir/"
            chmod 600 "$backup_subdir/key.pem"
            log "SUCCESS" "Backed up private key: $backup_subdir/key.pem"
        fi
    else
        log "INFO" "[DRY RUN] Would backup certificates to: $backup_subdir"
    fi
}

# Generate new private key
generate_private_key() {
    local key_file="$SSL_DIR/key.pem"
    
    log "INFO" "Generating new private key (${KEY_SIZE} bits)..."
    
    if [[ "$DRY_RUN" == "false" ]]; then
        if openssl genrsa -out "$key_file" $KEY_SIZE >/dev/null 2>&1; then
            chmod 600 "$key_file"
            log "SUCCESS" "Private key generated: $key_file"
        else
            log "ERROR" "Failed to generate private key"
            return 1
        fi
    else
        log "INFO" "[DRY RUN] Would generate private key: $key_file"
    fi
}

# Generate certificate signing request
generate_csr() {
    local key_file="$SSL_DIR/key.pem"
    local csr_file="$SSL_DIR/cert.csr"
    
    log "INFO" "Generating certificate signing request..."
    
    # Create subject string
    local subject="/C=$COUNTRY/ST=$STATE/L=$CITY/O=$ORGANIZATION/OU=$ORGANIZATIONAL_UNIT/CN=$DOMAIN_NAME"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        if openssl req -new -key "$key_file" -out "$csr_file" -subj "$subject" >/dev/null 2>&1; then
            log "SUCCESS" "CSR generated: $csr_file"
        else
            log "ERROR" "Failed to generate CSR"
            return 1
        fi
    else
        log "INFO" "[DRY RUN] Would generate CSR with subject: $subject"
    fi
}

# Generate self-signed certificate
generate_self_signed_certificate() {
    local key_file="$SSL_DIR/key.pem"
    local cert_file="$SSL_DIR/cert.pem"
    local csr_file="$SSL_DIR/cert.csr"
    
    log "INFO" "Generating self-signed certificate (valid for $CERT_VALIDITY_DAYS days)..."
    
    if [[ "$DRY_RUN" == "false" ]]; then
        # Create certificate with SAN extension for localhost and domain
        local config_file="$SSL_DIR/cert.conf"
        cat > "$config_file" << EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = $COUNTRY
ST = $STATE
L = $CITY
O = $ORGANIZATION
OU = $ORGANIZATIONAL_UNIT
CN = $DOMAIN_NAME

[v3_req]
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = $DOMAIN_NAME
DNS.2 = localhost
DNS.3 = *.localhost
IP.1 = 127.0.0.1
IP.2 = ::1
EOF
        
        if openssl req -x509 -new -nodes -key "$key_file" -sha256 -days $CERT_VALIDITY_DAYS -out "$cert_file" -config "$config_file" -extensions v3_req >/dev/null 2>&1; then
            log "SUCCESS" "Self-signed certificate generated: $cert_file"
            
            # Clean up temporary files
            rm -f "$csr_file" "$config_file"
        else
            log "ERROR" "Failed to generate self-signed certificate"
            return 1
        fi
    else
        log "INFO" "[DRY RUN] Would generate self-signed certificate: $cert_file"
    fi
}

# Validate generated certificate
validate_certificate() {
    local cert_file="$SSL_DIR/cert.pem"
    local key_file="$SSL_DIR/key.pem"
    
    log "INFO" "Validating generated certificate..."
    
    if [[ ! -f "$cert_file" ]]; then
        log "ERROR" "Certificate file not found for validation"
        return 1
    fi
    
    # Check certificate format
    if ! openssl x509 -in "$cert_file" -noout -text >/dev/null 2>&1; then
        log "ERROR" "Certificate format is invalid"
        return 1
    fi
    
    # Check if certificate matches private key
    if [[ -f "$key_file" ]]; then
        local cert_modulus=$(openssl x509 -in "$cert_file" -noout -modulus 2>/dev/null | md5sum | cut -d' ' -f1)
        local key_modulus=$(openssl rsa -in "$key_file" -noout -modulus 2>/dev/null | md5sum | cut -d' ' -f1)
        
        if [[ "$cert_modulus" == "$key_modulus" ]]; then
            log "SUCCESS" "Certificate matches private key"
        else
            log "ERROR" "Certificate does not match private key"
            return 1
        fi
    fi
    
    # Display certificate information
    local subject=$(openssl x509 -in "$cert_file" -noout -subject 2>/dev/null | cut -d= -f2-)
    local issuer=$(openssl x509 -in "$cert_file" -noout -issuer 2>/dev/null | cut -d= -f2-)
    local expiry=$(openssl x509 -in "$cert_file" -noout -enddate 2>/dev/null | cut -d= -f2)
    local fingerprint=$(openssl x509 -in "$cert_file" -noout -fingerprint -sha256 2>/dev/null | cut -d= -f2)
    
    log "INFO" "Certificate subject: $subject"
    log "INFO" "Certificate issuer: $issuer"
    log "INFO" "Certificate expires: $expiry"
    log "INFO" "Certificate fingerprint: $fingerprint"
    
    return 0
}

# Update services that use certificates
update_services() {
    log "INFO" "Checking services that need certificate updates..."
    
    local services_updated=0
    
    # Check if nginx is running and reload configuration
    if docker ps --format "{{.Names}}" | grep -q "nginx"; then
        log "INFO" "Reloading nginx configuration..."
        if [[ "$DRY_RUN" == "false" ]]; then
            if docker exec nginx nginx -s reload >/dev/null 2>&1; then
                log "SUCCESS" "Nginx configuration reloaded"
                ((services_updated++))
            else
                log "WARN" "Failed to reload nginx configuration"
            fi
        else
            log "INFO" "[DRY RUN] Would reload nginx configuration"
        fi
    fi
    
    # Check if any SutazAI services need restart
    local sutazai_services=("sutazai-backend" "sutazai-frontend")
    for service in "${sutazai_services[@]}"; do
        if docker ps --format "{{.Names}}" | grep -q "$service"; then
            log "INFO" "Restarting service: $service"
            if [[ "$DRY_RUN" == "false" ]]; then
                if docker restart "$service" >/dev/null 2>&1; then
                    log "SUCCESS" "Service restarted: $service"
                    ((services_updated++))
                else
                    log "WARN" "Failed to restart service: $service"
                fi
            else
                log "INFO" "[DRY RUN] Would restart service: $service"
            fi
        fi
    done
    
    if [[ $services_updated -gt 0 ]]; then
        log "SUCCESS" "Updated $services_updated services with new certificate"
    else
        log "INFO" "No services required certificate updates"
    fi
}

# Clean old certificate backups
clean_old_backups() {
    log "INFO" "Cleaning old certificate backups (older than 90 days)..."
    
    if [[ ! -d "$BACKUP_DIR" ]]; then
        log "INFO" "Backup directory does not exist, skipping cleanup"
        return 0
    fi
    
    local deleted_count=0
    
    while IFS= read -r -d '' backup_dir; do
        local basename=$(basename "$backup_dir")
        
        log "INFO" "Deleting old backup: $basename"
        
        if [[ "$DRY_RUN" == "false" ]]; then
            rm -rf "$backup_dir"
            log "SUCCESS" "Deleted: $basename"
        else
            log "INFO" "[DRY RUN] Would delete: $basename"
        fi
        
        ((deleted_count++))
    done < <(find "$BACKUP_DIR" -name "backup_*" -type d -mtime +90 -print0 2>/dev/null)
    
    if [[ $deleted_count -gt 0 ]]; then
        log "SUCCESS" "Deleted $deleted_count old certificate backups"
    else
        log "INFO" "No old certificate backups found for deletion"
    fi
}

# Test certificate with sample connection
test_certificate() {
    log "INFO" "Testing certificate with sample connection..."
    
    local cert_file="$SSL_DIR/cert.pem"
    
    if [[ ! -f "$cert_file" ]]; then
        log "ERROR" "Certificate file not found for testing"
        return 1
    fi
    
    # Test certificate verification
    if openssl verify -CAfile "$cert_file" "$cert_file" >/dev/null 2>&1; then
        log "SUCCESS" "Certificate verification passed (self-signed)"
    else
        log "INFO" "Certificate verification failed (expected for self-signed certificates)"
    fi
    
    # Test if certificate can be loaded by OpenSSL
    local cert_info=$(openssl x509 -in "$cert_file" -noout -text 2>/dev/null | head -20)
    if [[ -n "$cert_info" ]]; then
        log "SUCCESS" "Certificate can be loaded and parsed correctly"
    else
        log "ERROR" "Certificate cannot be loaded or parsed"
        return 1
    fi
    
    return 0
}

# Generate certificate renewal report
generate_renewal_report() {
    log "INFO" "Generating certificate renewal report..."
    
    local report_file="$LOG_DIR/certificate_renewal_report_$TIMESTAMP.json"
    local cert_file="$SSL_DIR/cert.pem"
    
    # Get certificate information
    local cert_subject=""
    local cert_expiry=""
    local cert_fingerprint=""
    
    if [[ -f "$cert_file" ]]; then
        cert_subject=$(openssl x509 -in "$cert_file" -noout -subject 2>/dev/null | cut -d= -f2- || echo "unknown")
        cert_expiry=$(openssl x509 -in "$cert_file" -noout -enddate 2>/dev/null | cut -d= -f2 || echo "unknown")
        cert_fingerprint=$(openssl x509 -in "$cert_file" -noout -fingerprint -sha256 2>/dev/null | cut -d= -f2 || echo "unknown")
    fi
    
    cat > "$report_file" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "renewal_type": "$([ "$DRY_RUN" == "true" ] && echo "dry_run" || echo "actual")",
    "force_renew": $FORCE_RENEW,
    "certificate": {
        "domain": "$DOMAIN_NAME",
        "subject": "$cert_subject",
        "expiry_date": "$cert_expiry",
        "fingerprint": "$cert_fingerprint",
        "validity_days": $CERT_VALIDITY_DAYS,
        "key_size": $KEY_SIZE
    },
    "renewal_threshold_days": $RENEWAL_THRESHOLD_DAYS,
    "next_check": "$(date -d '+1 day' -u +%Y-%m-%dT%H:%M:%SZ)",
    "next_renewal_check": "$(date -d "+$((RENEWAL_THRESHOLD_DAYS + 1)) days" -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    
    log "SUCCESS" "Certificate renewal report saved to: $report_file"
    
    # Create symlink to latest report
    if [[ "$DRY_RUN" == "false" ]]; then
        ln -sf "$report_file" "$LOG_DIR/latest_certificate_report.json"
    fi
}

# Main execution
main() {
    log "INFO" "Starting certificate renewal process for SutazAI system"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "Running in DRY RUN mode - no changes will be made"
    fi
    
    if [[ "$FORCE_RENEW" == "true" ]]; then
        log "INFO" "Force renewal enabled - will generate new certificate regardless of expiry"
    fi
    
    # Setup directories
    setup_directories
    
    # Check if renewal is needed
    local renewal_needed=false
    
    if [[ "$FORCE_RENEW" == "true" ]]; then
        renewal_needed=true
        log "INFO" "Force renewal requested"
    elif check_certificate_expiry; then
        renewal_needed=true
        log "INFO" "Certificate renewal needed based on expiry check"
    else
        log "INFO" "Certificate renewal not needed"
    fi
    
    if [[ "$renewal_needed" == "true" ]]; then
        log "INFO" "Proceeding with certificate renewal..."
        
        # Backup existing certificates
        backup_certificates
        
        # Generate new certificate
        generate_private_key || exit 1
        generate_csr || exit 1
        generate_self_signed_certificate || exit 1
        
        # Validate new certificate
        validate_certificate || exit 1
        
        # Update services
        update_services
        
        # Test certificate
        test_certificate || log "WARN" "Certificate testing had issues"
        
        log "SUCCESS" "Certificate renewal completed successfully"
    else
        log "INFO" "No certificate renewal performed"
    fi
    
    # Clean old backups
    clean_old_backups
    
    # Generate report
    generate_renewal_report
    
    log "SUCCESS" "Certificate renewal process completed"
    
    # Show summary
    echo
    echo "============================================"
    echo "       CERTIFICATE RENEWAL SUMMARY"
    echo "============================================"
    echo "Mode: $([ "$DRY_RUN" == "true" ] && echo "DRY RUN" || echo "ACTUAL RENEWAL")"
    echo "Domain: $DOMAIN_NAME"
    echo "Renewal performed: $([ "$renewal_needed" == "true" ] && echo "YES" || echo "NO")"
    echo "Force renewal: $([ "$FORCE_RENEW" == "true" ] && echo "YES" || echo "NO")"
    echo "Timestamp: $(date)"
    echo "============================================"
}

# Run main function
main "$@"