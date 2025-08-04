#!/bin/bash
# Service Mesh Initialization Script
# Registers all services with Consul and configures health checks

set -e

CONSUL_URL="${CONSUL_URL:-http://localhost:10040}"

echo "Waiting for Consul to be ready..."
until curl -s "${CONSUL_URL}/v1/status/leader" | grep -q '"'; do
    sleep 2
done

echo "Registering services with Consul..."

# Register Ollama service
curl -X PUT "${CONSUL_URL}/v1/agent/service/register" -d '{
  "ID": "ollama",
  "Name": "ollama",
  "Tags": ["ai", "llm", "primary"],
  "Address": "ollama",
  "Port": 11434,
  "Check": {
    "HTTP": "http://ollama:11434/api/tags",
    "Interval": "10s",
    "Timeout": "5s"
  }
}'

# Register AI Runtime service
curl -X PUT "${CONSUL_URL}/v1/agent/service/register" -d '{
  "ID": "ai-runtime",
  "Name": "ai-runtime",
  "Tags": ["ai", "runtime", "shared"],
  "Address": "ai-runtime",
  "Port": 8000,
  "Check": {
    "HTTP": "http://ai-runtime:8000/health",
    "Interval": "10s",
    "Timeout": "5s"
  }
}'

# Register PostgreSQL
curl -X PUT "${CONSUL_URL}/v1/agent/service/register" -d '{
  "ID": "postgres",
  "Name": "postgres",
  "Tags": ["database", "sql"],
  "Address": "postgres",
  "Port": 5432,
  "Check": {
    "TCP": "postgres:5432",
    "Interval": "10s",
    "Timeout": "5s"
  }
}'

# Register Redis
curl -X PUT "${CONSUL_URL}/v1/agent/service/register" -d '{
  "ID": "redis",
  "Name": "redis",
  "Tags": ["cache", "memory"],
  "Address": "redis",
  "Port": 6379,
  "Check": {
    "TCP": "redis:6379",
    "Interval": "10s",
    "Timeout": "5s"
  }
}'

# Register RabbitMQ
curl -X PUT "${CONSUL_URL}/v1/agent/service/register" -d '{
  "ID": "rabbitmq",
  "Name": "rabbitmq",
  "Tags": ["queue", "messaging"],
  "Address": "rabbitmq",
  "Port": 5672,
  "Check": {
    "TCP": "rabbitmq:5672",
    "Interval": "10s",
    "Timeout": "5s"
  }
}'

# Register ChromaDB
curl -X PUT "${CONSUL_URL}/v1/agent/service/register" -d '{
  "ID": "chromadb",
  "Name": "chromadb",
  "Tags": ["vector", "database"],
  "Address": "chromadb",
  "Port": 8000,
  "Check": {
    "HTTP": "http://chromadb:8000/api/v1/heartbeat",
    "Interval": "10s",
    "Timeout": "5s"
  }
}'

# Register Qdrant
curl -X PUT "${CONSUL_URL}/v1/agent/service/register" -d '{
  "ID": "qdrant",
  "Name": "qdrant",
  "Tags": ["vector", "database"],
  "Address": "qdrant",
  "Port": 6333,
  "Check": {
    "HTTP": "http://qdrant:6333/",
    "Interval": "10s",
    "Timeout": "5s"
  }
}'

# Register Monitoring Services
curl -X PUT "${CONSUL_URL}/v1/agent/service/register" -d '{
  "ID": "prometheus",
  "Name": "prometheus",
  "Tags": ["monitoring", "metrics"],
  "Address": "prometheus",
  "Port": 9090,
  "Check": {
    "HTTP": "http://prometheus:9090/-/healthy",
    "Interval": "10s",
    "Timeout": "5s"
  }
}'

curl -X PUT "${CONSUL_URL}/v1/agent/service/register" -d '{
  "ID": "grafana",
  "Name": "grafana",
  "Tags": ["monitoring", "dashboard"],
  "Address": "grafana",
  "Port": 3000,
  "Check": {
    "HTTP": "http://grafana:3000/api/health",
    "Interval": "10s",
    "Timeout": "5s"
  }
}'

echo "All services registered with Consul successfully!"

# Create key-value configurations
echo "Setting service configurations..."

# Ollama configuration
curl -X PUT "${CONSUL_URL}/v1/kv/config/ollama/models" -d 'tinyllama,qwen2.5-coder:7b,nomic-embed-text'
curl -X PUT "${CONSUL_URL}/v1/kv/config/ollama/max_parallel" -d '2'
curl -X PUT "${CONSUL_URL}/v1/kv/config/ollama/cpu_only" -d 'true'

# AI Runtime configuration
curl -X PUT "${CONSUL_URL}/v1/kv/config/ai-runtime/cache_ttl" -d '3600'
curl -X PUT "${CONSUL_URL}/v1/kv/config/ai-runtime/max_models" -d '3'
curl -X PUT "${CONSUL_URL}/v1/kv/config/ai-runtime/memory_limit_gb" -d '4'

echo "Service mesh initialization complete!"