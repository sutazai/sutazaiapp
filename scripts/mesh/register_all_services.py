#!/usr/bin/env python3
"""
Register All Services with Consul
Purpose: Register all running services with Consul for service discovery
Created: 2025-08-18 UTC
"""
import requests
import json
import subprocess
import time

def register_service(service_name, service_id, address, port, tags=None):
    """Register a service with Consul"""
    if tags is None:
        tags = []
    
    service_def = {
        "ID": service_id,
        "Name": service_name,
        "Tags": tags,
        "Address": address,
        "Port": port,
        "Check": {
            "HTTP": f"http://{address}:{port}/health",
            "Interval": "10s",
            "Timeout": "5s",
            "DeregisterCriticalServiceAfter": "1m"
        }
    }
    
    try:
        response = requests.put(
            "http://localhost:10006/v1/agent/service/register",
            json=service_def,
            timeout=5
        )
        if response.status_code == 200:
            print(f"  ✓ Registered {service_name} ({service_id})")
            return True
        else:
            print(f"  ✗ Failed to register {service_name}: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Error registering {service_name}: {e}")
        return False

def main():
    print("="*50)
    print("REGISTERING ALL SERVICES WITH CONSUL")
    print("="*50)
    print()
    
    # Services to register
    services = [
        # Core Services
        ("backend-api", "backend-api-10010", "localhost", 10010, ["api", "backend", "fastapi", "mesh-enabled"]),
        ("frontend", "frontend-10011", "localhost", 10011, ["ui", "streamlit", "frontend"]),
        
        # Database Services
        ("postgresql", "postgresql-10000", "localhost", 10000, ["database", "postgres", "primary"]),
        ("redis", "redis-10001", "localhost", 10001, ["cache", "redis", "session"]),
        ("neo4j", "neo4j-10003", "localhost", 10003, ["database", "graph", "neo4j"]),
        
        # AI Services
        ("ollama", "ollama-10104", "localhost", 10104, ["ai", "llm", "ollama"]),
        ("chromadb", "chromadb-10100", "localhost", 10100, ["ai", "vectordb", "chromadb"]),
        ("qdrant", "qdrant-10101", "localhost", 10101, ["ai", "vectordb", "qdrant"]),
        
        # Message Queue
        ("rabbitmq", "rabbitmq-10008", "localhost", 10008, ["queue", "rabbitmq", "messaging"]),
        
        # Monitoring Services
        ("prometheus", "prometheus-10200", "localhost", 10200, ["monitoring", "metrics", "prometheus"]),
        ("grafana", "grafana-10201", "localhost", 10201, ["monitoring", "visualization", "grafana"]),
        ("loki", "loki-10202", "localhost", 10202, ["monitoring", "logging", "loki"]),
        ("jaeger", "jaeger-10210", "localhost", 10210, ["monitoring", "tracing", "jaeger"]),
        ("alertmanager", "alertmanager-10203", "localhost", 10203, ["monitoring", "alerting"]),
        
        # Infrastructure Services
        ("consul", "consul-10006", "localhost", 10006, ["discovery", "consul", "mesh"]),
        ("kong", "kong-10005", "localhost", 10005, ["gateway", "api-gateway", "kong"]),
        
        # Special Services
        ("mcp-orchestrator", "mcp-orchestrator-12375", "localhost", 12375, ["orchestration", "dind", "mcp"]),
        ("unified-memory", "unified-memory-3009", "localhost", 3009, ["memory", "unified", "mcp"]),
        ("unified-dev", "unified-dev-4001", "localhost", 4001, ["development", "unified", "mcp"]),
    ]
    
    # Register MCP services (through DinD)
    mcp_services = [
        ("mcp-claude-flow", 3001),
        ("mcp-ruv-swarm", 3002),
        ("mcp-files", 3003),
        ("mcp-context7", 3004),
        ("mcp-http-fetch", 3005),
        ("mcp-ddg", 3006),
        ("mcp-sequentialthinking", 3007),
        ("mcp-nx-mcp", 3008),
        ("mcp-extended-memory", 3009),
        ("mcp-mcp-ssh", 3010),
        ("mcp-ultimatecoder", 3011),
        ("mcp-playwright-mcp", 3012),
        ("mcp-memory-bank-mcp", 3013),
        ("mcp-knowledge-graph-mcp", 3014),
        ("mcp-compass-mcp", 3015),
        ("mcp-github", 3016),
        ("mcp-http", 3017),
        ("mcp-language-server", 3018),
        ("mcp-claude-task-runner", 3019),
    ]
    
    # Register core services
    print("Registering Core Services:")
    registered = 0
    failed = 0
    
    for service_name, service_id, address, port, tags in services:
        if register_service(service_name, service_id, address, port, tags):
            registered += 1
        else:
            failed += 1
    
    # Register MCP services
    print("\nRegistering MCP Services (via DinD):")
    for service_name, port in mcp_services:
        service_id = f"{service_name}-{port}"
        tags = ["mcp", service_name.replace("mcp-", ""), "containerized", "dind"]
        if register_service(service_name, service_id, "localhost", port, tags):
            registered += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*50)
    print("REGISTRATION SUMMARY")
    print("="*50)
    print(f"✓ Successfully registered: {registered} services")
    print(f"✗ Failed to register: {failed} services")
    
    # Check total services in Consul
    try:
        response = requests.get("http://localhost:10006/v1/agent/services")
        if response.status_code == 200:
            services = response.json()
            print(f"\nTotal services in Consul: {len(services)}")
            
            # List all service names
            print("\nRegistered Services:")
            for service_id, service_info in services.items():
                print(f"  - {service_info['Service']} ({service_id})")
    except Exception as e:
        print(f"\nError checking Consul services: {e}")

if __name__ == "__main__":
    main()