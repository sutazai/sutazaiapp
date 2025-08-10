#!/usr/bin/env python3
"""
SutazAI Chaos Engineering - Docker Integration
Integrates chaos engineering with existing Docker Compose infrastructure
"""

import os
import json
import yaml
import logging
import docker

class DockerIntegration:
    """Integrates chaos engineering with Docker Compose"""
    
    def __init__(self, compose_file: str = "/opt/sutazaiapp/docker-compose.yml"):
        self.compose_file = compose_file
        self.chaos_compose_file = "/opt/sutazaiapp/chaos/docker-compose.chaos.yml"
        self.docker_client = docker.from_env()
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("chaos_docker_integration")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def create_chaos_compose_extension(self):
        """Create a Docker Compose extension for chaos services"""
        chaos_compose = {
            'version': '3.8',
            'services': {
                'chaos-engine': {
                    'build': {
                        'context': '/opt/sutazaiapp/chaos',
                        'dockerfile': 'Dockerfile'
                    },
                    'container_name': 'sutazai-chaos-engine',
                    'restart': 'unless-stopped',
                    'volumes': [
                        '/var/run/docker.sock:/var/run/docker.sock:ro',
                        '/opt/sutazaiapp/chaos:/app/chaos',
                        '/opt/sutazaiapp/logs:/app/logs'
                    ],
                    'environment': [
                        'CHAOS_CONFIG_PATH=/app/chaos/config/chaos-config.yaml',
                        'DOCKER_COMPOSE_FILE=/opt/sutazaiapp/docker-compose.yml',
                        'LOG_LEVEL=INFO'
                    ],
                    'ports': ['8200:8080'],
                    'depends_on': {
                        'prometheus': {'condition': 'service_healthy'},
                        'grafana': {'condition': 'service_healthy'}
                    },
                    'networks': ['sutazai-network'],
                    'healthcheck': {
                        'test': ['CMD', 'python3', '-c', 'import requests; requests.get("http://localhost:8080/health")'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3,
                        'start_period': '40s'
                    },
                    'labels': {
                        'chaos.sutazai.com/service': 'chaos-engine',
                        'chaos.sutazai.com/protected': 'true'
                    }
                },
                
                'chaos-monkey': {
                    'build': {
                        'context': '/opt/sutazaiapp/chaos',
                        'dockerfile': 'Dockerfile'
                    },
                    'container_name': 'sutazai-chaos-monkey',
                    'restart': 'unless-stopped',
                    'volumes': [
                        '/var/run/docker.sock:/var/run/docker.sock:ro',
                        '/opt/sutazaiapp/chaos:/app/chaos',
                        '/opt/sutazaiapp/logs:/app/logs'
                    ],
                    'environment': [
                        'CHAOS_CONFIG_PATH=/app/chaos/config/chaos-config.yaml',
                        'CHAOS_MODE=safe',
                        'LOG_LEVEL=INFO'
                    ],
                    'command': ['python3', '/app/chaos/scripts/chaos-monkey.py', '--daemon'],
                    'depends_on': {
                        'chaos-engine': {'condition': 'service_healthy'}
                    },
                    'networks': ['sutazai-network'],
                    'healthcheck': {
                        'test': ['CMD', 'python3', '-c', 'import json; print(json.load(open("/app/chaos/chaos_monkey_state.json"))["mode"])'],
                        'interval': '60s',
                        'timeout': '10s',
                        'retries': 3,
                        'start_period': '60s'
                    },
                    'labels': {
                        'chaos.sutazai.com/service': 'chaos-monkey',
                        'chaos.sutazai.com/protected': 'true'
                    }
                },
                
                'chaos-dashboard': {
                    'build': {
                        'context': '/opt/sutazaiapp/chaos/dashboard',
                        'dockerfile': 'Dockerfile'
                    },
                    'container_name': 'sutazai-chaos-dashboard',
                    'restart': 'unless-stopped',
                    'volumes': [
                        '/opt/sutazaiapp/chaos/reports:/app/reports:ro'
                    ],
                    'environment': [
                        'CHAOS_API_URL=http://chaos-engine:8080',
                        'REPORTS_DIR=/app/reports'
                    ],
                    'ports': ['8201:8080'],
                    'depends_on': ['chaos-engine'],
                    'networks': ['sutazai-network'],
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8080/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3
                    }
                }
            },
            
            'networks': {
                'sutazai-network': {
                    'external': True
                }
            }
        }
        
        # Save chaos compose file
        with open(self.chaos_compose_file, 'w') as f:
            yaml.dump(chaos_compose, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Chaos Docker Compose extension created: {self.chaos_compose_file}")
    
    def add_chaos_labels_to_services(self):
        """Add chaos engineering labels to existing services"""
        try:
            with open(self.compose_file, 'r') as f:
                compose_data = yaml.safe_load(f)
            
            services = compose_data.get('services', {})
            
            # Define service categories for chaos targeting
            service_categories = {
                'critical': ['postgres', 'redis', 'prometheus', 'grafana'],
                'core': ['backend', 'frontend', 'ollama'],
                'vector': ['chromadb', 'qdrant', 'neo4j'],
                'agents': ['autogpt', 'crewai', 'letta', 'aider'],
                'tools': ['langflow', 'flowise', 'dify', 'n8n'],
                'monitoring': ['health-monitor', 'promtail', 'loki']
            }
            
            # Add chaos labels to services
            for service_name, service_config in services.items():
                if not isinstance(service_config, dict):
                    continue
                
                if 'labels' not in service_config:
                    service_config['labels'] = {}
                
                # Determine service category
                category = 'misc'
                for cat, service_list in service_categories.items():
                    if any(s in service_name for s in service_list):
                        category = cat
                        break
                
                # Add chaos labels
                chaos_labels = {
                    'chaos.sutazai.com/target': 'true',
                    'chaos.sutazai.com/category': category,
                    'chaos.sutazai.com/protected': 'true' if category == 'critical' else 'false',
                    'chaos.sutazai.com/safe_mode_only': 'true' if category in ['critical', 'monitoring'] else 'false'
                }
                
                service_config['labels'].update(chaos_labels)
            
            # Backup original file
            backup_file = f"{self.compose_file}.backup_chaos"
            os.rename(self.compose_file, backup_file)
            
            # Write updated compose file
            with open(self.compose_file, 'w') as f:
                yaml.dump(compose_data, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Added chaos labels to services. Backup: {backup_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to add chaos labels: {e}")
            raise
    
    def create_chaos_dockerfile(self):
        """Create Dockerfile for chaos services"""
        dockerfile_content = """# SutazAI Chaos Engineering Docker Image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    wget \\
    jq \\
    stress-ng \\
    iproute2 \\
    tcpdump \\
    net-tools \\
    procps \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create app directory
WORKDIR /app

# Copy chaos framework
COPY . /app/chaos/

# Set Python path
ENV PYTHONPATH="/app/chaos:$PYTHONPATH"

# Create non-root user
RUN useradd -m -s /bin/bash chaos && \\
    chown -R chaos:chaos /app

USER chaos

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \\
    CMD python3 -c "import sys; sys.exit(0)"

# Default command
CMD ["python3", "/app/chaos/scripts/chaos-engine.py", "--help"]
"""
        
        dockerfile_path = "/opt/sutazaiapp/chaos/Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # Create requirements.txt
        requirements_content = """docker>=6.0.0
pyyaml>=6.0
requests>=2.28.0
schedule>=1.2.0
prometheus-client>=0.15.0
psutil>=5.9.0
networkx>=3.0
numpy>=1.24.0
aiofiles>=22.1.0
asyncio-mqtt>=0.11.0
"""
        
        requirements_path = "/opt/sutazaiapp/chaos/requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write(requirements_content)
        
        self.logger.info(f"Created Dockerfile and requirements.txt for chaos services")
    
    def get_service_dependencies(self) -> Dict[str, List[str]]:
        """Extract service dependencies from Docker Compose"""
        try:
            with open(self.compose_file, 'r') as f:
                compose_data = yaml.safe_load(f)
            
            dependencies = {}
            services = compose_data.get('services', {})
            
            for service_name, service_config in services.items():
                deps = []
                
                # Extract depends_on
                depends_on = service_config.get('depends_on', [])
                if isinstance(depends_on, list):
                    deps.extend(depends_on)
                elif isinstance(depends_on, dict):
                    deps.extend(depends_on.keys())
                
                # Extract environment variable dependencies (simplified)
                env_vars = service_config.get('environment', [])
                if isinstance(env_vars, list):
                    for env_var in env_vars:
                        if isinstance(env_var, str) and 'localhost' in env_var:
                            # Parse service references
                            pass
                
                dependencies[service_name] = deps
            
            return dependencies
            
        except Exception as e:
            self.logger.error(f"Failed to extract dependencies: {e}")
            return {}
    
    def validate_chaos_integration(self) -> Dict[str, Any]:
        """Validate chaos engineering integration"""
        results = {
            'docker_integration': False,
            'compose_labels': False,
            'chaos_services': False,
            'service_dependencies': False,
            'errors': []
        }
        
        try:
            # Check if chaos compose file exists
            if os.path.exists(self.chaos_compose_file):
                results['chaos_services'] = True
            else:
                results['errors'].append("Chaos compose file not found")
            
            # Check if main compose has chaos labels
            with open(self.compose_file, 'r') as f:
                compose_data = yaml.safe_load(f)
            
            chaos_labeled_services = 0
            services = compose_data.get('services', {})
            
            for service_name, service_config in services.items():
                labels = service_config.get('labels', {})
                if any(label.startswith('chaos.sutazai.com/') for label in labels):
                    chaos_labeled_services += 1
            
            if chaos_labeled_services > 0:
                results['compose_labels'] = True
                results['labeled_services'] = chaos_labeled_services
            else:
                results['errors'].append("No chaos labels found in compose file")
            
            # Check service dependencies
            dependencies = self.get_service_dependencies()
            if dependencies:
                results['service_dependencies'] = True
                results['dependency_count'] = len(dependencies)
            
            # Overall integration status
            results['docker_integration'] = all([
                results['chaos_services'],
                results['compose_labels'],
                results['service_dependencies']
            ])
            
        except Exception as e:
            results['errors'].append(f"Validation error: {e}")
        
        return results
    
    def deploy_chaos_services(self):
        """Deploy chaos engineering services"""
        try:
            # Create necessary files
            self.create_chaos_dockerfile()
            self.create_chaos_compose_extension()
            
            # Deploy using docker-compose
            import subprocess
            
            cmd = [
                'docker-compose',
                '-f', self.chaos_compose_file,
                'up', '-d'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='/opt/sutazaiapp')
            
            if result.returncode == 0:
                self.logger.info("Chaos services deployed successfully")
                return True
            else:
                self.logger.error(f"Failed to deploy chaos services: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error deploying chaos services: {e}")
            return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SutazAI Chaos Engineering Docker Integration")
    parser.add_argument("--compose-file", default="/opt/sutazaiapp/docker-compose.yml",
                       help="Path to Docker Compose file")
    parser.add_argument("--add-labels", action="store_true",
                       help="Add chaos labels to existing services")
    parser.add_argument("--create-extension", action="store_true",
                       help="Create chaos compose extension")
    parser.add_argument("--deploy", action="store_true",
                       help="Deploy chaos services")
    parser.add_argument("--validate", action="store_true",
                       help="Validate chaos integration")
    
    args = parser.parse_args()
    
    integration = DockerIntegration(args.compose_file)
    
    if args.add_labels:
        integration.add_chaos_labels_to_services()
    
    if args.create_extension:
        integration.create_chaos_compose_extension()
    
    if args.deploy:
        success = integration.deploy_chaos_services()
        print(f"Deployment {'successful' if success else 'failed'}")
    
    if args.validate:
        results = integration.validate_chaos_integration()
        print(json.dumps(results, indent=2))
    
    if not any([args.add_labels, args.create_extension, args.deploy, args.validate]):
        print("Use --help for available options")