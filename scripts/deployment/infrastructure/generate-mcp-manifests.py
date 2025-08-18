#!/usr/bin/env python3
"""
Generate MCP Service Manifests for DinD Deployment
Creates all missing MCP service manifests based on the service registry
"""
import os
import yaml
from pathlib import Path

# Define all MCP services with their specific configurations
MCP_SERVICES = {
    'http-fetch': {
        'image': 'node:18-alpine',
        'command': ['npx', 'http-fetch-mcp', 'server', '--port', '3005'],
        'port': 3005,
        'tier': 'network',
        'protocol': 'http',
        'memory': '256M'
    },
    'ddg': {
        'image': 'node:18-alpine',
        'command': ['npx', 'ddg-mcp', 'server', '--port', '3006'],
        'port': 3006,
        'tier': 'search',
        'protocol': 'http',
        'memory': '256M'
    },
    'sequentialthinking': {
        'image': 'python:3.11-slim',
        'command': ['python', '-m', 'sequentialthinking', '--port', '3007'],
        'port': 3007,
        'tier': 'ai',
        'protocol': 'http',
        'memory': '512M'
    },
    'nx-mcp': {
        'image': 'node:18-alpine',
        'command': ['npx', '@nx/mcp-server', '--port', '3008'],
        'port': 3008,
        'tier': 'development',
        'protocol': 'http',
        'memory': '512M'
    },
    'extended-memory': {
        'image': 'python:3.11-slim',
        'command': ['python', '-m', 'extended_memory.server', '--port', '3009'],
        'port': 3009,
        'tier': 'storage',
        'protocol': 'http',
        'memory': '1024M'
    },
    'mcp-ssh': {
        'image': 'alpine:latest',
        'command': ['sh', '-c', 'apk add openssh && /usr/sbin/sshd -D -p 3010'],
        'port': 3010,
        'tier': 'network',
        'protocol': 'ssh',
        'memory': '128M'
    },
    'ultimatecoder': {
        'image': 'python:3.11-slim',
        'command': ['python', '-m', 'ultimatecoder.server', '--port', '3011'],
        'port': 3011,
        'tier': 'ai',
        'protocol': 'http',
        'memory': '1024M'
    },
    'playwright-mcp': {
        'image': 'mcr.microsoft.com/playwright:v1.40.0-focal',
        'command': ['node', '/app/server.js', '--port', '3012'],
        'port': 3012,
        'tier': 'testing',
        'protocol': 'http',
        'memory': '1024M'
    },
    'memory-bank-mcp': {
        'image': 'python:3.11-slim',
        'command': ['python', '-m', 'memory_bank.server', '--port', '3013'],
        'port': 3013,
        'tier': 'storage',
        'protocol': 'http',
        'memory': '512M'
    },
    'knowledge-graph-mcp': {
        'image': 'python:3.11-slim',
        'command': ['python', '-m', 'knowledge_graph.server', '--port', '3014'],
        'port': 3014,
        'tier': 'knowledge',
        'protocol': 'http',
        'memory': '512M'
    },
    'compass-mcp': {
        'image': 'node:18-alpine',
        'command': ['npx', 'compass-mcp', 'server', '--port', '3015'],
        'port': 3015,
        'tier': 'navigation',
        'protocol': 'http',
        'memory': '256M'
    },
    'github': {
        'image': 'node:18-alpine',
        'command': ['npx', '@modelcontextprotocol/github', '--port', '3016'],
        'port': 3016,
        'tier': 'vcs',
        'protocol': 'http',
        'memory': '256M'
    },
    'http': {
        'image': 'node:18-alpine',
        'command': ['npx', 'http-mcp', 'server', '--port', '3017'],
        'port': 3017,
        'tier': 'network',
        'protocol': 'http',
        'memory': '256M'
    },
    'language-server': {
        'image': 'node:18-alpine',
        'command': ['npx', 'language-server-mcp', '--port', '3018'],
        'port': 3018,
        'tier': 'development',
        'protocol': 'lsp',
        'memory': '512M'
    }
}

def generate_manifest(service_name, config):
    """Generate a MCP service manifest"""
    manifest = {
        'apiVersion': 'mcp/v1',
        'kind': 'MCPContainer',
        'metadata': {
            'name': f'mcp-{service_name}',
            'labels': {
                'mcp.service': service_name,
                'mcp.tier': config['tier']
            }
        },
        'spec': {
            'image': config['image'],
            'container_name': f'mcp-{service_name}',
            'command': config['command'],
            'environment': {
                'MCP_PORT': str(config['port']),
                'MESH_ENABLED': 'true',
                'SERVICE_NAME': service_name
            },
            'ports': {
                str(13000 + config['port'] - 3000): str(config['port'])
            },
            'volumes': {
                'mcp_shared_data': '/mcp-shared',
                '/opt/sutazaiapp': '/opt/sutazaiapp:ro'
            },
            'healthcheck': {
                'test': ['CMD', 'wget', '-q', '--spider', f"http://localhost:{config['port']}/health"],
                'interval': '30s',
                'timeout': '10s',
                'retries': 3,
                'start_period': '45s'
            },
            'restart_policy': 'unless-stopped',
            'resources': {
                'limits': {
                    'cpus': '1.0' if config['memory'] == '1024M' else '0.5',
                    'memory': config['memory']
                },
                'reservations': {
                    'cpus': '0.2' if config['memory'] == '1024M' else '0.1',
                    'memory': '128M' if config['memory'] == '1024M' else '64M'
                }
            },
            'networks': ['sutazai-dind-internal'],
            'labels': {
                'mcp.managed': 'true',
                'mcp.protocol': config['protocol'],
                'mcp.integration': service_name.replace('-', '_'),
                'mcp.service': 'true'
            }
        }
    }
    
    # Add environment variables based on service type
    if 'python' in config['image']:
        manifest['spec']['environment']['PYTHONUNBUFFERED'] = '1'
    elif 'node' in config['image']:
        manifest['spec']['environment']['NODE_ENV'] = 'production'
    
    return manifest

def main():
    """Generate all MCP service manifests"""
    manifest_dir = Path('/opt/sutazaiapp/docker/dind/orchestrator/mcp-manifests')
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    generated = []
    skipped = []
    
    for service_name, config in MCP_SERVICES.items():
        manifest_file = manifest_dir / f'{service_name}-mcp.yml'
        
        # Skip if manifest already exists
        if manifest_file.exists():
            skipped.append(service_name)
            continue
        
        # Generate manifest
        manifest = generate_manifest(service_name, config)
        
        # Write manifest file
        with open(manifest_file, 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
        
        generated.append(service_name)
        print(f"✅ Generated manifest for {service_name}")
    
    print(f"\n📊 Summary:")
    print(f"  - Generated: {len(generated)} manifests")
    print(f"  - Skipped (existing): {len(skipped)} manifests")
    print(f"  - Total manifests: {len(generated) + len(skipped)}")
    
    if generated:
        print(f"\n📝 New manifests created:")
        for service in generated:
            print(f"  - {service}-mcp.yml")

if __name__ == '__main__':
    main()