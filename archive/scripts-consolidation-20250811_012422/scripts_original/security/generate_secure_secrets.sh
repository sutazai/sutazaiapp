#!/bin/bash
# ============================================================================
# SutazAI Secure Secrets Generator
# ============================================================================
# This script generates cryptographically secure secrets for all required
# environment variables in the .env.secure file
# ============================================================================

set -euo pipefail

echo "============================================================================"
echo "SutazAI Secure Secrets Generator"
echo "============================================================================"
echo ""

# Check if .env.secure exists
if [ ! -f ".env.secure" ]; then
    echo "ERROR: .env.secure file not found. Please create it first."
    exit 1
fi

# Create a backup of the existing .env.secure
cp .env.secure .env.secure.backup.$(date +%Y%m%d_%H%M%S)

# Function to generate secure random strings
generate_password() {
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-32
}

generate_hex_key() {
    openssl rand -hex 32
}

generate_api_key() {
    echo "sk-$(openssl rand -hex 24)"
}

generate_uuid() {
    uuidgen | tr '[:upper:]' '[:lower:]'
}

echo "Generating secure secrets..."
echo ""

# Database Passwords
POSTGRES_PASSWORD=$(generate_password)
REDIS_PASSWORD=$(generate_password)
NEO4J_PASSWORD=$(generate_password)
RABBITMQ_PASSWORD=$(generate_password)

# Security Keys
SECRET_KEY=$(generate_hex_key)
JWT_SECRET=$(generate_hex_key)
ENCRYPTION_KEY=$(generate_hex_key)
API_ENCRYPTION_KEY=$(generate_hex_key)

# Monitoring Passwords
GRAFANA_PASSWORD=$(generate_password)
PROMETHEUS_PASSWORD=$(generate_password)
LOKI_PASSWORD=$(generate_password)
ALERTMANAGER_PASSWORD=$(generate_password)

# API Keys
CHROMADB_API_KEY=$(generate_api_key)
QDRANT_API_KEY=$(generate_api_key)
FAISS_API_KEY=$(generate_api_key)
TABBY_API_KEY=$(generate_api_key)
PORCUPINE_ACCESS_KEY=$(generate_api_key)

# Authentication
VAULT_TOKEN=$(generate_hex_key)
VAULT_UNSEAL_KEY=$(generate_hex_key)
KEYCLOAK_CLIENT_SECRET=$(generate_uuid)
KEYCLOAK_ADMIN_PASSWORD=$(generate_password)
OAUTH2_CLIENT_SECRET=$(generate_uuid)

# Gateway & Service Mesh
API_GATEWAY_SECRET=$(generate_hex_key)
KONG_ADMIN_PASSWORD=$(generate_password)

# Testing
TEST_PASSWORD=$(generate_password)
TEST_API_KEY=$(generate_api_key)
CONFIG_MANAGER_PASSWORD=$(generate_password)
CONFIG_MANAGER_TOKEN=$(generate_hex_key)

# Backup
BACKUP_ENCRYPTION_KEY=$(generate_hex_key)

# Create temporary file with generated secrets
cat > .env.secure.generated << EOF
# ============================================================================
# SutazAI Production Security Environment Configuration
# Generated: $(date)
# ============================================================================
# WARNING: This file contains sensitive credentials. 
# - Store securely and never commit to version control
# - Use a secrets management system in production
# - Rotate these secrets regularly
# ============================================================================

# System Configuration
SUTAZAI_ENV=production
TZ=UTC
NODE_ENV=production
DEBUG=false

# Database Credentials
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_DB=sutazai
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
DATABASE_URL=postgresql://sutazai:${POSTGRES_PASSWORD}@postgres:5432/sutazai

REDIS_PASSWORD=${REDIS_PASSWORD}
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
REDIS_REQUIREPASS=${REDIS_PASSWORD}

NEO4J_USER=neo4j
NEO4J_PASSWORD=${NEO4J_PASSWORD}
NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}
NEO4J_HOST=neo4j
NEO4J_HTTP_PORT=10002
NEO4J_BOLT_PORT=10003

RABBITMQ_DEFAULT_USER=sutazai
RABBITMQ_DEFAULT_PASS=${RABBITMQ_PASSWORD}
RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672
AMQP_URL=amqp://sutazai:${RABBITMQ_PASSWORD}@rabbitmq:5672

# Security Keys
SECRET_KEY=${SECRET_KEY}
JWT_SECRET=${JWT_SECRET}
JWT_SECRET_KEY=${JWT_SECRET}
JWT_ALGORITHM=HS256
ENCRYPTION_KEY=${ENCRYPTION_KEY}
API_ENCRYPTION_KEY=${API_ENCRYPTION_KEY}

# Monitoring
GRAFANA_USERNAME=admin
GRAFANA_PASSWORD=${GRAFANA_PASSWORD}
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
PROMETHEUS_PASSWORD=${PROMETHEUS_PASSWORD}
LOKI_PASSWORD=${LOKI_PASSWORD}
ALERTMANAGER_PASSWORD=${ALERTMANAGER_PASSWORD}

# AI/ML Services
OLLAMA_HOST=ollama
OLLAMA_PORT=11434
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=tinyllama
DEFAULT_MODEL=tinyllama

CHROMADB_API_KEY=${CHROMADB_API_KEY}
QDRANT_API_KEY=${QDRANT_API_KEY}
FAISS_API_KEY=${FAISS_API_KEY}

# Authentication
VAULT_ADDR=http://vault:8200
VAULT_TOKEN=${VAULT_TOKEN}
VAULT_UNSEAL_KEY=${VAULT_UNSEAL_KEY}
VAULT_MOUNT_PATH=secret

KEYCLOAK_SERVER_URL=http://keycloak:8080
KEYCLOAK_REALM=sutazai
KEYCLOAK_CLIENT_ID=sutazai
KEYCLOAK_CLIENT_SECRET=${KEYCLOAK_CLIENT_SECRET}
KEYCLOAK_ADMIN_USER=admin
KEYCLOAK_ADMIN_PASSWORD=${KEYCLOAK_ADMIN_PASSWORD}

OAUTH2_CLIENT_ID=sutazai
OAUTH2_CLIENT_SECRET=${OAUTH2_CLIENT_SECRET}

# External Integrations
ENABLE_TABBY=false
TABBY_URL=http://tabbyml:8080
TABBY_API_KEY=${TABBY_API_KEY}
PORCUPINE_ACCESS_KEY=${PORCUPINE_ACCESS_KEY}

# API Gateway
API_GATEWAY_SECRET=${API_GATEWAY_SECRET}
KONG_ADMIN_PASSWORD=${KONG_ADMIN_PASSWORD}

# Security Hardening
SECURITY_HARDENED=true
ENABLE_SSL=false
DISABLE_ROOT_ACCESS=true
ENABLE_AUDIT_LOGGING=true
ENABLE_RATE_LIMITING=true
ENABLE_CORS=true
CORS_ORIGINS=["http://localhost:10011"]

# Container Security
DOCKER_USER_ID=1001
DOCKER_GROUP_ID=1001
RUN_AS_NON_ROOT=true

# Testing
TEST_PASSWORD=${TEST_PASSWORD}
TEST_API_KEY=${TEST_API_KEY}
CONFIG_MANAGER_PASSWORD=${CONFIG_MANAGER_PASSWORD}
CONFIG_MANAGER_TOKEN=${CONFIG_MANAGER_TOKEN}

# Feature Flags
SUTAZAI_ENTERPRISE_FEATURES=1
SUTAZAI_ENABLE_KNOWLEDGE_GRAPH=1
SUTAZAI_ENABLE_COGNITIVE=1
ENABLE_FSDP=false

# Backup Configuration
BACKUP_ENABLED=true
BACKUP_RETENTION_DAYS=30
BACKUP_ENCRYPTION_KEY=${BACKUP_ENCRYPTION_KEY}

EOF

echo "✅ Secure secrets generated successfully!"
echo ""
echo "Generated secrets have been saved to: .env.secure.generated"
echo ""
echo "⚠️  IMPORTANT SECURITY STEPS:"
echo "1. Review the generated file: .env.secure.generated"
echo "2. Copy it to .env to use with Docker Compose"
echo "3. Store a backup in a secure location (password manager, vault, etc.)"
echo "4. Delete .env.secure.generated after copying"
echo "5. Never commit .env files to version control"
echo ""
echo "To use the generated secrets:"
echo "  cp .env.secure.generated .env"
echo "  docker-compose up -d"
echo ""
echo "Passwords have been generated using cryptographically secure methods:"
echo "- Database passwords: 32 characters (base64)"
echo "- API keys: 48 character hex with 'sk-' prefix"
echo "- Security keys: 64 character hex strings"
echo "- UUIDs: Standard v4 format"
echo ""
echo "============================================================================"