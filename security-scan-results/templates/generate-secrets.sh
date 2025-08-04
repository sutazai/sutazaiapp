#!/bin/bash
# Generate secure secrets for SutazAI deployment

echo "# Generated SutazAI Secrets - $(date)"
echo "# Store these securely and never commit to version control"
echo ""

echo "# Database passwords"
echo "POSTGRES_PASSWORD=$(openssl rand -base64 32)"
echo "REDIS_PASSWORD=$(openssl rand -base64 32)"
echo "NEO4J_PASSWORD=$(openssl rand -base64 32)"
echo "GRAFANA_PASSWORD=$(openssl rand -base64 32)"
echo ""

echo "# JWT and encryption keys"
echo "JWT_SECRET=$(openssl rand -base64 64)"
echo "SECRET_KEY=$(openssl rand -base64 64)"
echo ""

echo "# Service tokens"
echo "CONFIG_MANAGER_PASSWORD=$(openssl rand -base64 32)"
echo "CONFIG_MANAGER_TOKEN=$(openssl rand -base64 32)"
echo "CHROMADB_API_KEY=$(openssl rand -base64 32)"
