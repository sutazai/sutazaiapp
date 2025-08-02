#!/bin/bash

###############################################################################
# Enterprise Security Setup for SutazAI
# Configures Vault, generates secrets, and sets up secure communication
###############################################################################

set -euo pipefail

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Create secure directories
create_secure_dirs() {
    log "Creating secure directories..."
    
    dirs=(
        "$PROJECT_ROOT/secrets"
        "$PROJECT_ROOT/certs"
        "$PROJECT_ROOT/config/vault"
        "$PROJECT_ROOT/config/kong"
        "$PROJECT_ROOT/config/haproxy"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        chmod 700 "$dir"
    done
}

# Generate secure passwords
generate_secrets() {
    log "Generating secure secrets..."
    
    # Generate strong passwords
    generate_password() {
        openssl rand -base64 32 | tr -d "=+/" | cut -c1-25
    }
    
    # Database passwords
    echo "$(generate_password)" > "$PROJECT_ROOT/secrets/postgres_password.txt"
    echo "$(generate_password)" > "$PROJECT_ROOT/secrets/replication_password.txt"
    echo "$(generate_password)" > "$PROJECT_ROOT/secrets/grafana_password.txt"
    echo "$(generate_password)" > "$PROJECT_ROOT/secrets/neo4j_password.txt"
    echo "$(generate_password)" > "$PROJECT_ROOT/secrets/redis_password.txt"
    
    # API keys
    echo "$(openssl rand -hex 32)" > "$PROJECT_ROOT/secrets/api_key.txt"
    echo "$(openssl rand -hex 32)" > "$PROJECT_ROOT/secrets/jwt_secret.txt"
    
    # Vault root token
    echo "$(uuidgen)" > "$PROJECT_ROOT/secrets/vault_token.txt"
    
    # Set secure permissions
    chmod 600 "$PROJECT_ROOT/secrets/"*.txt
    
    log "Secrets generated successfully"
}

# Setup Vault configuration
setup_vault() {
    log "Setting up Vault configuration..."
    
    cat > "$PROJECT_ROOT/config/vault/vault.hcl" << 'EOF'
ui = true

listener "tcp" {
  address       = "0.0.0.0:8200"
  tls_cert_file = "/vault/certs/vault.crt"
  tls_key_file  = "/vault/certs/vault.key"
}

storage "raft" {
  path    = "/vault/data"
  node_id = "node1"
}

seal "awskms" {
  region     = "us-east-1"
  kms_key_id = "REPLACE_WITH_KMS_KEY"
}

telemetry {
  prometheus_retention_time = "30s"
  disable_hostname = true
}

api_addr = "http://127.0.0.1:8200"
cluster_addr = "https://127.0.0.1:8201"
EOF

    # Create Vault policy for SutazAI
    cat > "$PROJECT_ROOT/config/vault/sutazai-policy.hcl" << 'EOF'
# Policy for SutazAI services
path "secret/data/sutazai/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "secret/metadata/sutazai/*" {
  capabilities = ["list"]
}

path "database/creds/sutazai-*" {
  capabilities = ["read"]
}

path "pki/issue/sutazai" {
  capabilities = ["create", "update"]
}

path "auth/token/renew-self" {
  capabilities = ["update"]
}
EOF
}

# Generate SSL certificates
generate_certificates() {
    log "Generating SSL certificates..."
    
    # Create CA key and certificate
    openssl genrsa -out "$PROJECT_ROOT/certs/ca-key.pem" 4096
    
    openssl req -new -x509 -days 3650 -key "$PROJECT_ROOT/certs/ca-key.pem" \
        -out "$PROJECT_ROOT/certs/ca.pem" \
        -subj "/C=US/ST=State/L=City/O=SutazAI/CN=SutazAI-CA"
    
    # Generate server certificate
    openssl genrsa -out "$PROJECT_ROOT/certs/server-key.pem" 4096
    
    openssl req -new -key "$PROJECT_ROOT/certs/server-key.pem" \
        -out "$PROJECT_ROOT/certs/server.csr" \
        -subj "/C=US/ST=State/L=City/O=SutazAI/CN=*.sutazai.local"
    
    # Create extensions file
    cat > "$PROJECT_ROOT/certs/extensions.cnf" << EOF
[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = sutazai.local
DNS.2 = *.sutazai.local
DNS.3 = localhost
IP.1 = 127.0.0.1
IP.2 = 192.168.131.128
EOF
    
    # Sign the certificate
    openssl x509 -req -days 365 -in "$PROJECT_ROOT/certs/server.csr" \
        -CA "$PROJECT_ROOT/certs/ca.pem" -CAkey "$PROJECT_ROOT/certs/ca-key.pem" \
        -out "$PROJECT_ROOT/certs/server.pem" -CAcreateserial \
        -extfile "$PROJECT_ROOT/certs/extensions.cnf" -extensions v3_req
    
    # Generate client certificates for mTLS
    openssl genrsa -out "$PROJECT_ROOT/certs/client-key.pem" 4096
    
    openssl req -new -key "$PROJECT_ROOT/certs/client-key.pem" \
        -out "$PROJECT_ROOT/certs/client.csr" \
        -subj "/C=US/ST=State/L=City/O=SutazAI/CN=sutazai-client"
    
    openssl x509 -req -days 365 -in "$PROJECT_ROOT/certs/client.csr" \
        -CA "$PROJECT_ROOT/certs/ca.pem" -CAkey "$PROJECT_ROOT/certs/ca-key.pem" \
        -out "$PROJECT_ROOT/certs/client.pem" -CAcreateserial
    
    # Set secure permissions
    chmod 600 "$PROJECT_ROOT/certs/"*.pem
    chmod 600 "$PROJECT_ROOT/certs/"*.key
    
    log "SSL certificates generated successfully"
}

# Setup HAProxy configuration
setup_haproxy() {
    log "Setting up HAProxy configuration..."
    
    cat > "$PROJECT_ROOT/config/haproxy/haproxy.cfg" << 'EOF'
global
    maxconn 4096
    log stdout local0
    ssl-default-bind-ciphers ECDHE+AESGCM:ECDHE+AES256:ECDHE+AES128:!PSK:!DHE:!RSA:!DSS:!aNull:!MD5
    ssl-default-bind-options no-sslv3 no-tlsv10 no-tlsv11
    tune.ssl.default-dh-param 2048

defaults
    mode http
    timeout connect 10s
    timeout client 30s
    timeout server 30s
    option httplog
    option dontlognull
    option forwardfor
    option http-server-close

frontend stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 30s
    stats auth admin:${HAPROXY_STATS_PASSWORD}

frontend http_front
    bind *:80
    redirect scheme https code 301 if !{ ssl_fc }

frontend https_front
    bind *:443 ssl crt /certs/server.pem ca-file /certs/ca.pem verify optional
    
    # Security headers
    http-response set-header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload"
    http-response set-header X-Frame-Options "DENY"
    http-response set-header X-Content-Type-Options "nosniff"
    http-response set-header X-XSS-Protection "1; mode=block"
    
    # ACLs
    acl is_api path_beg /api
    acl is_websocket hdr(Upgrade) -i WebSocket
    acl is_monitoring path_beg /metrics /health
    
    # Use backend based on ACL
    use_backend api_backend if is_api
    use_backend websocket_backend if is_websocket
    use_backend monitoring_backend if is_monitoring
    default_backend web_backend

backend web_backend
    balance roundrobin
    option httpchk GET /health
    server web1 sutazai-streamlit:8501 check ssl verify none

backend api_backend
    balance leastconn
    option httpchk GET /health
    server api1 agent-orchestrator:8000 check ssl verify none
    server api2 agent-orchestrator:8000 check ssl verify none
    server api3 agent-orchestrator:8000 check ssl verify none

backend websocket_backend
    balance source
    option http-server-close
    server ws1 agent-orchestrator:8000 check ssl verify none

backend monitoring_backend
    balance roundrobin
    server prometheus prometheus:9090 check
    server grafana grafana:3000 check
EOF
}

# Setup Kong API Gateway
setup_kong() {
    log "Setting up Kong API Gateway configuration..."
    
    # Kong declarative configuration
    cat > "$PROJECT_ROOT/config/kong/kong.yml" << 'EOF'
_format_version: "3.0"

services:
  - name: sutazai-api
    url: http://agent-orchestrator:8000
    routes:
      - name: api-route
        paths:
          - /api
        strip_path: true
    plugins:
      - name: rate-limiting
        config:
          minute: 60
          policy: local
      - name: jwt
        config:
          key_claim_name: kid
          claims_to_verify:
            - exp
      - name: cors
        config:
          origins:
            - "*"
          methods:
            - GET
            - POST
            - PUT
            - DELETE
            - OPTIONS
          headers:
            - Accept
            - Authorization
            - Content-Type
          exposed_headers:
            - X-Auth-Token
          credentials: true
          max_age: 3600

  - name: sutazai-websocket
    url: ws://agent-orchestrator:8000
    routes:
      - name: websocket-route
        paths:
          - /ws
        protocols:
          - ws
          - wss

consumers:
  - username: sutazai-admin
    jwt_secrets:
      - key: ${JWT_SECRET}
        algorithm: HS256

  - username: sutazai-service
    jwt_secrets:
      - key: ${SERVICE_JWT_SECRET}
        algorithm: HS256

plugins:
  - name: prometheus
    config:
      status_code_metrics: true
      latency_metrics: true
      bandwidth_metrics: true
      upstream_health_metrics: true
EOF
}

# Initialize Vault
initialize_vault() {
    log "Initializing Vault..."
    
    # Start Vault in dev mode for initial setup
    docker run -d --name vault-init \
        -p 8200:8200 \
        -e VAULT_DEV_ROOT_TOKEN_ID="$(cat $PROJECT_ROOT/secrets/vault_token.txt)" \
        vault:1.15
    
    sleep 5
    
    export VAULT_ADDR='http://127.0.0.1:8200'
    export VAULT_TOKEN="$(cat $PROJECT_ROOT/secrets/vault_token.txt)"
    
    # Enable required secret engines
    docker exec vault-init vault auth enable jwt
    docker exec vault-init vault secrets enable -path=secret kv-v2
    docker exec vault-init vault secrets enable database
    docker exec vault-init vault secrets enable pki
    
    # Configure database secret engine
    docker exec vault-init vault write database/config/postgresql \
        plugin_name=postgresql-database-plugin \
        allowed_roles="sutazai-*" \
        connection_url="postgresql://{{username}}:{{password}}@postgres-master:5432/sutazai?sslmode=disable" \
        username="vault_admin" \
        password="$(cat $PROJECT_ROOT/secrets/postgres_password.txt)"
    
    # Create database role
    docker exec vault-init vault write database/roles/sutazai-app \
        db_name=postgresql \
        creation_statements="CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}'; \
                           GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO \"{{name}}\";" \
        default_ttl="1h" \
        max_ttl="24h"
    
    # Store secrets in Vault
    docker exec vault-init vault kv put secret/sutazai/database \
        postgres_password="$(cat $PROJECT_ROOT/secrets/postgres_password.txt)" \
        redis_password="$(cat $PROJECT_ROOT/secrets/redis_password.txt)" \
        neo4j_password="$(cat $PROJECT_ROOT/secrets/neo4j_password.txt)"
    
    docker exec vault-init vault kv put secret/sutazai/api \
        jwt_secret="$(cat $PROJECT_ROOT/secrets/jwt_secret.txt)" \
        api_key="$(cat $PROJECT_ROOT/secrets/api_key.txt)"
    
    # Create policy
    docker cp "$PROJECT_ROOT/config/vault/sutazai-policy.hcl" vault-init:/tmp/
    docker exec vault-init vault policy write sutazai /tmp/sutazai-policy.hcl
    
    # Stop initialization container
    docker stop vault-init
    docker rm vault-init
    
    log "Vault initialized successfully"
}

# Create security scanning script
create_security_scan() {
    log "Creating security scanning script..."
    
    cat > "$PROJECT_ROOT/scripts/security_scan.sh" << 'EOF'
#!/bin/bash

# Security scanning script for SutazAI

echo "Running security scan..."

# Scan Docker images for vulnerabilities
echo "Scanning Docker images..."
docker images --format "{{.Repository}}:{{.Tag}}" | while read image; do
    echo "Scanning $image..."
    docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
        aquasec/trivy image --severity HIGH,CRITICAL "$image"
done

# Check for exposed secrets
echo "Checking for exposed secrets..."
docker-compose config | grep -E "(password|secret|key|token)" | grep -v "_FILE"

# Network security check
echo "Checking exposed ports..."
docker ps --format "table {{.Names}}\t{{.Ports}}" | grep -E "0\.0\.0\.0"

# File permission check
echo "Checking file permissions..."
find . -type f -perm 0777 -o -perm 0666 | grep -v ".git"

echo "Security scan complete!"
EOF
    
    chmod +x "$PROJECT_ROOT/scripts/security_scan.sh"
}

# Main setup function
main() {
    log "Starting Enterprise Security Setup for SutazAI..."
    
    cd "$PROJECT_ROOT"
    
    create_secure_dirs
    generate_secrets
    setup_vault
    generate_certificates
    setup_haproxy
    setup_kong
    initialize_vault
    create_security_scan
    
    log "Enterprise security setup completed successfully!"
    
    echo -e "\n${GREEN}Security Setup Complete!${NC}"
    echo -e "${BLUE}Generated Secrets Location:${NC} $PROJECT_ROOT/secrets/"
    echo -e "${BLUE}SSL Certificates Location:${NC} $PROJECT_ROOT/certs/"
    echo -e "${BLUE}Vault Token:${NC} $(cat $PROJECT_ROOT/secrets/vault_token.txt)"
    echo -e "\n${YELLOW}Next Steps:${NC}"
    echo "1. Review and customize security configurations"
    echo "2. Run security scan: ./scripts/security_scan.sh"
    echo "3. Deploy with enterprise configuration: docker-compose -f docker-compose.enterprise.yml up -d"
}

# Run main function
main "$@"