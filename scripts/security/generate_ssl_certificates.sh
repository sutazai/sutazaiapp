#!/bin/bash
# ULTRA Security SSL Certificate Generator
# Generates self-signed certificates for development and production-ready CSRs
# Author: ULTRA Security Engineer
# Date: 2025-08-11

set -euo pipefail

# Configuration
CERT_DIR="/opt/sutazaiapp/config/ssl/certs"
DAYS_VALID=365
KEY_SIZE=4096
COUNTRY="US"
STATE="California"
LOCALITY="San Francisco"
ORGANIZATION="SutazAI"
ORG_UNIT="Security"
COMMON_NAME="${1:-sutazai.local}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== ULTRA SSL/TLS Certificate Generator ===${NC}"

# Create certificate directory
mkdir -p "$CERT_DIR"
chmod 700 "$CERT_DIR"

# Function to generate self-signed certificate
generate_self_signed() {
    local domain="$1"
    echo -e "${YELLOW}Generating self-signed certificate for: $domain${NC}"
    
    # Generate private key
    openssl genpkey -algorithm RSA -out "$CERT_DIR/$domain.key" -pkeyopt rsa_keygen_bits:$KEY_SIZE
    chmod 600 "$CERT_DIR/$domain.key"
    
    # Generate certificate signing request
    openssl req -new -key "$CERT_DIR/$domain.key" -out "$CERT_DIR/$domain.csr" \
        -subj "/C=$COUNTRY/ST=$STATE/L=$LOCALITY/O=$ORGANIZATION/OU=$ORG_UNIT/CN=$domain"
    
    # Generate self-signed certificate
    openssl x509 -req -days $DAYS_VALID -in "$CERT_DIR/$domain.csr" \
        -signkey "$CERT_DIR/$domain.key" -out "$CERT_DIR/$domain.crt" \
        -extensions v3_req \
        -extfile <(cat <<EOF
[v3_req]
subjectAltName = @alt_names
[alt_names]
DNS.1 = $domain
DNS.2 = *.$domain
DNS.3 = localhost
IP.1 = 127.0.0.1
IP.2 = ::1
EOF
    )
    
    # Generate combined PEM file
    cat "$CERT_DIR/$domain.crt" "$CERT_DIR/$domain.key" > "$CERT_DIR/$domain.pem"
    chmod 600 "$CERT_DIR/$domain.pem"
    
    echo -e "${GREEN}✓ Self-signed certificate generated${NC}"
}

# Function to generate production CSR
generate_production_csr() {
    local domain="$1"
    echo -e "${YELLOW}Generating production CSR for: $domain${NC}"
    
    # Generate private key
    openssl genpkey -algorithm RSA -out "$CERT_DIR/$domain-prod.key" -pkeyopt rsa_keygen_bits:$KEY_SIZE
    chmod 600 "$CERT_DIR/$domain-prod.key"
    
    # Generate CSR with SAN
    openssl req -new -key "$CERT_DIR/$domain-prod.key" -out "$CERT_DIR/$domain-prod.csr" \
        -config <(cat <<EOF
[req]
default_bits = $KEY_SIZE
prompt = no
default_md = sha256
distinguished_name = dn
req_extensions = v3_req

[dn]
C=$COUNTRY
ST=$STATE
L=$LOCALITY
O=$ORGANIZATION
OU=$ORG_UNIT
CN=$domain

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = $domain
DNS.2 = www.$domain
DNS.3 = api.$domain
DNS.4 = *.api.$domain
EOF
    )
    
    echo -e "${GREEN}✓ Production CSR generated${NC}"
    echo -e "${YELLOW}Submit this CSR to your Certificate Authority:${NC}"
    echo "$CERT_DIR/$domain-prod.csr"
}

# Function to generate Diffie-Hellman parameters
generate_dhparam() {
    echo -e "${YELLOW}Generating Diffie-Hellman parameters (this may take a while)...${NC}"
    openssl dhparam -out "$CERT_DIR/dhparam.pem" 2048
    chmod 600 "$CERT_DIR/dhparam.pem"
    echo -e "${GREEN}✓ DH parameters generated${NC}"
}

# Function to create certificate bundle
create_bundle() {
    local domain="$1"
    echo -e "${YELLOW}Creating certificate bundle...${NC}"
    
    # Create CA bundle (for production, replace with actual CA certificates)
    cat > "$CERT_DIR/ca-bundle.crt" <<EOF
# Add your Certificate Authority certificates here
# For production, this should contain:
# 1. Root CA certificate
# 2. Intermediate CA certificates
EOF
    
    chmod 644 "$CERT_DIR/ca-bundle.crt"
    echo -e "${GREEN}✓ Certificate bundle created${NC}"
}

# Function to generate certificates for all services
generate_service_certificates() {
    echo -e "${YELLOW}Generating certificates for all services...${NC}"
    
    local services=(
        "postgres.sutazai.local"
        "redis.sutazai.local"
        "neo4j.sutazai.local"
        "rabbitmq.sutazai.local"
        "ollama.sutazai.local"
        "backend.sutazai.local"
        "frontend.sutazai.local"
        "grafana.sutazai.local"
        "prometheus.sutazai.local"
    )
    
    for service in "${services[@]}"; do
        generate_self_signed "$service"
    done
    
    echo -e "${GREEN}✓ All service certificates generated${NC}"
}

# Function to verify certificate
verify_certificate() {
    local cert_file="$1"
    echo -e "${YELLOW}Verifying certificate: $cert_file${NC}"
    
    # Check certificate details
    openssl x509 -in "$cert_file" -text -noout | grep -E "(Subject:|Issuer:|Not Before|Not After)"
    
    # Verify certificate chain (for production certificates)
    if [ -f "$CERT_DIR/ca-bundle.crt" ]; then
        openssl verify -CAfile "$CERT_DIR/ca-bundle.crt" "$cert_file" 2>/dev/null || echo "Self-signed certificate (expected for development)"
    fi
    
    echo -e "${GREEN}✓ Certificate verification complete${NC}"
}

# Main execution
main() {
    echo -e "${GREEN}Starting SSL/TLS certificate generation...${NC}"
    echo -e "Domain: ${YELLOW}$COMMON_NAME${NC}"
    echo ""
    
    # Generate certificates
    generate_self_signed "$COMMON_NAME"
    generate_production_csr "$COMMON_NAME"
    generate_dhparam
    create_bundle "$COMMON_NAME"
    
    # Generate service certificates
    if [ "${2:-}" = "--all-services" ]; then
        generate_service_certificates
    fi
    
    # Verify main certificate
    verify_certificate "$CERT_DIR/$COMMON_NAME.crt"
    
    echo ""
    echo -e "${GREEN}=== Certificate Generation Complete ===${NC}"
    echo -e "Certificates location: ${YELLOW}$CERT_DIR${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. For development: Use the self-signed certificates"
    echo "2. For production: Submit the CSR to your Certificate Authority"
    echo "3. Update nginx/apache configuration to use the certificates"
    echo "4. Restart web server to apply SSL/TLS configuration"
    
    # Display certificate fingerprint
    echo ""
    echo -e "${YELLOW}Certificate fingerprint:${NC}"
    openssl x509 -in "$CERT_DIR/$COMMON_NAME.crt" -fingerprint -sha256 -noout
}

# Run main function
main "$@"