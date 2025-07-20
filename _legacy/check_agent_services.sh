#!/bin/bash

# Quick check of all running services
echo "=== SutazAI Running Services ==="
echo
echo "Core Services:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "postgres|redis|ollama|chroma|qdrant" | sort
echo
echo "AI Agents & Services:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep sutazai | grep -v -E "postgres|redis|ollama|chroma|qdrant" | sort
echo
echo "Total Containers: $(docker ps | grep sutazai | wc -l)"
echo
echo "Backend Status:"
curl -s http://localhost:8000/health | jq . 2>/dev/null || echo "Backend not responding"
echo
echo "Available Models:"
docker exec sutazai-ollama ollama list 2>/dev/null | tail -n +2 || echo "Could not list models"