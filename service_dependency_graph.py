#!/usr/bin/env python3
"""Generate service dependency graph visualization"""

import json
from collections import defaultdict

# Load the analysis results
analysis_results = {
    "critical_services": [
        ["ollama", 24],
        ["postgres", 14],
        ["redis", 12],
        ["backend", 7],
        ["neo4j", 5],
        ["qdrant", 5]
    ],
    "service_dependencies": {
        "backend": ["chromadb", "neo4j", "redis", "qdrant", "postgres", "ollama"],
        "frontend": ["backend"],
        "mcp-server": ["chromadb", "backend", "redis", "qdrant", "neo4j", "postgres", "ollama"],
        "ai-metrics-exporter": ["backend", "ollama", "postgres", "redis"],
        "context-framework": ["ollama", "chromadb", "qdrant", "neo4j"],
        "llamaindex": ["neo4j", "chromadb", "qdrant", "ollama"],
        # AI Agents
        "autogpt": ["backend", "ollama"],
        "crewai": ["backend", "ollama"],
        "letta": ["ollama", "redis", "postgres", "backend"],
        "agentgpt": ["redis", "postgres", "ollama"],
        "dify": ["ollama", "postgres", "redis"],
        "langflow": ["postgres", "redis"],
        "finrobot": ["redis", "postgres", "ollama"],
        "agentzero": ["ollama", "postgres", "redis"],
        "code-improver": ["ollama", "postgres", "redis"],
        # Other services with dependencies
        "service-hub": ["postgres", "redis"],
        "health-monitor": ["postgres"],
        "grafana": ["prometheus", "loki"],
        "promtail": ["loki"]
    }
}

def generate_mermaid_graph():
    """Generate Mermaid graph syntax for service dependencies"""
    
    mermaid = ["graph TD"]
    
    # Define node styles based on service category
    styles = {
        "databases": "fill:#FF6B6B,stroke:#C92A2A,stroke-width:3px,color:#fff",
        "core": "fill:#4ECDC4,stroke:#216969,stroke-width:3px,color:#fff",
        "vector_stores": "fill:#45B7D1,stroke:#1864AB,stroke-width:2px,color:#fff",
        "ai_agents": "fill:#96CEB4,stroke:#51A877,stroke-width:2px,color:#fff",
        "monitoring": "fill:#DDA0DD,stroke:#8B008B,stroke-width:2px,color:#fff",
        "infrastructure": "fill:#FFD93D,stroke:#F59F00,stroke-width:2px,color:#fff"
    }
    
    # Service categories
    categories = {
        "postgres": "databases",
        "redis": "databases",
        "neo4j": "databases",
        "backend": "core",
        "frontend": "core",
        "ollama": "core",
        "chromadb": "vector_stores",
        "qdrant": "vector_stores",
        "prometheus": "monitoring",
        "grafana": "monitoring",
        "loki": "monitoring",
        "mcp-server": "infrastructure",
        "service-hub": "infrastructure",
        "ai-metrics-exporter": "monitoring"
    }
    
    # Add nodes with labels
    for service, deps in analysis_results["service_dependencies"].items():
        # Default category for uncategorized services
        if service not in categories:
            if "gpt" in service or "ai" in service or "agent" in service:
                categories[service] = "ai_agents"
            else:
                categories[service] = "infrastructure"
    
    # Add critical services with special notation
    critical_services = {svc[0]: svc[1] for svc in analysis_results["critical_services"]}
    
    # Add edges
    edge_count = defaultdict(lambda: defaultdict(int))
    for service, deps in analysis_results["service_dependencies"].items():
        for dep in deps:
            edge_count[dep][service] += 1
            mermaid.append(f"    {service} --> {dep}")
    
    # Add node definitions with counts for critical services
    mermaid.append("")
    mermaid.append("    %% Node definitions")
    for service in set(sum(analysis_results["service_dependencies"].values(), []) + list(analysis_results["service_dependencies"].keys())):
        if service in critical_services:
            mermaid.append(f"    {service}[\"{service}<br/>({critical_services[service]} deps)\"]")
        else:
            mermaid.append(f"    {service}[\"{service}\"]")
    
    # Apply styles
    mermaid.append("")
    mermaid.append("    %% Apply styles")
    for service, category in categories.items():
        if category in styles:
            mermaid.append(f"    style {service} {styles[category]}")
    
    return "\n".join(mermaid)

def generate_summary_report():
    """Generate a summary report of the service architecture"""
    
    report = []
    report.append("# SutazAI Service Architecture Analysis")
    report.append("")
    report.append("## Executive Summary")
    report.append("")
    report.append("The SutazAI system consists of 46+ interconnected services organized into the following categories:")
    report.append("- **Core Services**: Backend API, Frontend UI, and Ollama LLM service")
    report.append("- **Data Layer**: PostgreSQL (primary DB), Redis (cache/messaging), Neo4j (graph DB)")
    report.append("- **Vector Stores**: ChromaDB, Qdrant, and FAISS for embeddings")
    report.append("- **AI Agents**: 19 specialized agents for various tasks")
    report.append("- **Monitoring**: Comprehensive Prometheus/Grafana stack")
    report.append("- **Infrastructure**: Service orchestration and management tools")
    report.append("")
    report.append("## Critical Services (by dependency count)")
    report.append("")
    report.append("| Service | Dependencies | Role |")
    report.append("|---------|--------------|------|")
    for service, count in analysis_results["critical_services"]:
        role = {
            "ollama": "LLM inference engine - central to all AI operations",
            "postgres": "Primary database - stores all persistent data",
            "redis": "Cache & message broker - enables async communication",
            "backend": "API gateway - coordinates all service interactions",
            "neo4j": "Graph database - stores relationships and knowledge graphs",
            "qdrant": "Vector database - handles embeddings and similarity search"
        }.get(service, "Supporting service")
        report.append(f"| {service} | {count} | {role} |")
    
    report.append("")
    report.append("## Communication Patterns")
    report.append("")
    report.append("### 1. Synchronous HTTP (REST APIs)")
    report.append("- Frontend → Backend")
    report.append("- AI Agents → Backend")
    report.append("- Monitoring services → Backend")
    report.append("")
    report.append("### 2. Asynchronous Messaging (Redis Pub/Sub)")
    report.append("- 12 services use Redis for async communication")
    report.append("- Enables event-driven architecture")
    report.append("- Supports task queuing and result caching")
    report.append("")
    report.append("### 3. Database Connections")
    report.append("- 14 services connect directly to PostgreSQL")
    report.append("- 5 services use Neo4j for graph operations")
    report.append("- Multiple services share vector stores")
    report.append("")
    report.append("### 4. LLM Service Integration")
    report.append("- 24 services depend on Ollama")
    report.append("- Centralized LLM inference reduces resource usage")
    report.append("- Supports model switching without service changes")
    
    return "\n".join(report)

# Generate outputs
print("=== Mermaid Graph ===")
print(generate_mermaid_graph())
print("\n\n=== Summary Report ===")
print(generate_summary_report())