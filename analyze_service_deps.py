#\!/usr/bin/env python3
"""Analyze SutazAI service dependencies and communication patterns"""

import os
import json
import yaml
import re
from collections import defaultdict
from pathlib import Path

# Service dependency mapping
service_deps = defaultdict(set)
service_configs = {}
api_contracts = defaultdict(list)

# Parse docker-compose.yml for service dependencies
docker_compose_path = Path("/opt/sutazaiapp/docker-compose.yml")
if docker_compose_path.exists():
    with open(docker_compose_path, 'r') as f:
        compose_data = yaml.safe_load(f)
        
    services = compose_data.get('services', {})
    for service_name, service_config in services.items():
        # Extract dependencies
        deps = service_config.get('depends_on', {})
        if isinstance(deps, list):
            service_deps[service_name].update(deps)
        elif isinstance(deps, dict):
            service_deps[service_name].update(deps.keys())
        
        # Extract environment variables that indicate connections
        env = service_config.get('environment', {})
        if isinstance(env, list):
            env_dict = {}
            for item in env:
                if '=' in item:
                    k, v = item.split('=', 1)
                    env_dict[k] = v
            env = env_dict
        
        # Analyze connection patterns
        for key, value in env.items():
            if isinstance(value, str):
                # Database connections
                if 'DATABASE_URL' in key or 'postgres' in value.lower():
                    service_deps[service_name].add('postgres')
                # Redis connections
                if 'REDIS_URL' in key or 'redis:' in value:
                    service_deps[service_name].add('redis')
                # Ollama connections
                if 'OLLAMA' in key or 'ollama:' in value:
                    service_deps[service_name].add('ollama')
                # Neo4j connections
                if 'NEO4J' in key or 'neo4j:' in value:
                    service_deps[service_name].add('neo4j')
                # Vector DB connections
                if 'CHROMADB' in key or 'chromadb:' in value:
                    service_deps[service_name].add('chromadb')
                if 'QDRANT' in key or 'qdrant:' in value:
                    service_deps[service_name].add('qdrant')
                # Backend API connections
                if 'BACKEND_URL' in key or 'backend:8000' in value:
                    service_deps[service_name].add('backend')
        
        # Store service config for later analysis
        service_configs[service_name] = {
            'ports': service_config.get('ports', []),
            'volumes': service_config.get('volumes', []),
            'networks': service_config.get('networks', []),
            'restart': service_config.get('restart', ''),
            'container_name': service_config.get('container_name', ''),
            'image': service_config.get('image', ''),
            'build': service_config.get('build', {}),
        }

# Categorize services
service_categories = {
    'databases': ['postgres', 'redis', 'neo4j'],
    'vector_stores': ['chromadb', 'qdrant', 'faiss'],
    'core': ['backend', 'frontend', 'ollama'],
    'ai_agents': ['autogpt', 'crewai', 'letta', 'aider', 'gpt-engineer', 'agentgpt', 
                  'privategpt', 'langflow', 'flowise', 'shellgpt', 'pentestgpt', 
                  'finrobot', 'opendevin', 'documind', 'browser-use', 'skyvern',
                  'dify', 'agentzero', 'autogen'],
    'ml_frameworks': ['pytorch', 'tensorflow', 'jax'],
    'monitoring': ['prometheus', 'grafana', 'loki', 'promtail', 'alertmanager', 
                   'node-exporter', 'cadvisor', 'externalSystem, thirdPartyService-exporter', 'postgres-exporter',
                   'redis-exporter', 'ai-metrics-exporter'],
    'infrastructure': ['n8n', 'health-monitor', 'context-framework', 'service-hub',
                       'code-improver', 'awesome-code-ai', 'fsdp', 'mcp-server'],
}

# Create reverse mapping
service_to_category = {}
for category, services in service_categories.items():
    for service in services:
        service_to_category[service] = category

# Output results
results = {
    'total_services': len(services),
    'service_dependencies': {k: list(v) for k, v in service_deps.items()},
    'categories': service_categories,
    'critical_services': [],
    'dependency_chains': {},
    'communication_patterns': {}
}

# Identify critical services (depended on by many others)
dependency_count = defaultdict(int)
for service, deps in service_deps.items():
    for dep in deps:
        dependency_count[dep] += 1

results['critical_services'] = sorted(
    [(service, count) for service, count in dependency_count.items() if count >= 5],
    key=lambda x: x[1],
    reverse=True
)

# Analyze communication patterns
communication_patterns = {
    'sync_http': [],
    'async_messaging': [],
    'database_connections': [],
    'cache_connections': [],
    'llm_connections': []
}

for service, deps in service_deps.items():
    if 'backend' in deps:
        communication_patterns['sync_http'].append(f"{service} -> backend")
    if 'redis' in deps:
        communication_patterns['async_messaging'].append(f"{service} -> redis")
        communication_patterns['cache_connections'].append(f"{service} -> redis")
    if 'postgres' in deps:
        communication_patterns['database_connections'].append(f"{service} -> postgres")
    if 'ollama' in deps:
        communication_patterns['llm_connections'].append(f"{service} -> ollama")

results['communication_patterns'] = communication_patterns

# Output results
print(json.dumps(results, indent=2))
