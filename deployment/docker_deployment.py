"""
SutazAI Docker Deployment System
Enterprise-grade containerized deployment for the AGI/ASI system

This module provides comprehensive Docker-based deployment capabilities
including multi-stage builds, health checks, and production optimizations.
"""

import os
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import docker
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentEnvironment(Enum):
    """Deployment environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class DockerConfig:
    """Docker configuration settings"""
    image_name: str = "sutazai/agi-system"
    image_tag: str = "latest"
    base_image: str = "python:3.11-slim"
    expose_port: int = 8000
    health_check_interval: int = 30
    health_check_timeout: int = 10
    health_check_retries: int = 3
    memory_limit: str = "2g"
    cpu_limit: str = "1"
    restart_policy: str = "unless-stopped"

class DockerDeploymentManager:
    """Manages Docker-based deployment of the SutazAI system"""
    
    def __init__(self, config: DockerConfig = None):
        self.config = config or DockerConfig()
        self.docker_client = docker.from_env()
        self.base_dir = Path("/opt/sutazaiapp")
        self.deployment_dir = self.base_dir / "deployment"
        self.deployment_dir.mkdir(exist_ok=True)
        
        logger.info("Docker Deployment Manager initialized")
    
    def create_dockerfile(self, environment: DeploymentEnvironment) -> str:
        """Create optimized Dockerfile for the environment"""
        
        dockerfile_content = f"""
# Multi-stage Docker build for SutazAI AGI System
# Stage 1: Build stage
FROM {self.config.base_image} as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    make \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models /app/config

# Stage 2: Runtime stage
FROM {self.config.base_image} as runtime

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash sutazai

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /app /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/logs /app/models /app/config && \\
    chown -R sutazai:sutazai /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV SUTAZAI_ENV={environment.value}
ENV SUTAZAI_CONFIG_PATH=/app/config/settings.json

# Expose port
EXPOSE {self.config.expose_port}

# Health check
HEALTHCHECK --interval={self.config.health_check_interval}s \\
            --timeout={self.config.health_check_timeout}s \\
            --retries={self.config.health_check_retries} \\
            CMD curl -f http://localhost:{self.config.expose_port}/health || exit 1

# Switch to non-root user
USER sutazai

# Start application
CMD ["python", "-m", "uvicorn", "api.agi_api:get_api_app", "--host", "0.0.0.0", "--port", "{self.config.expose_port}"]
"""
        
        dockerfile_path = self.deployment_dir / f"Dockerfile.{environment.value}"
        dockerfile_path.write_text(dockerfile_content.strip())
        
        logger.info(f"Created Dockerfile for {environment.value} environment")
        return str(dockerfile_path)
    
    def create_docker_compose(self, environment: DeploymentEnvironment) -> str:
        """Create Docker Compose configuration"""
        
        compose_config = {
            "version": "3.8",
            "services": {
                "sutazai-agi": {
                    "build": {
                        "context": str(self.base_dir),
                        "dockerfile": f"deployment/Dockerfile.{environment.value}"
                    },
                    "image": f"{self.config.image_name}:{environment.value}",
                    "container_name": f"sutazai-agi-{environment.value}",
                    "ports": [f"{self.config.expose_port}:{self.config.expose_port}"],
                    "environment": [
                        f"SUTAZAI_ENV={environment.value}",
                        "SUTAZAI_CONFIG_PATH=/app/config/settings.json",
                        "PYTHONPATH=/app",
                        "PYTHONUNBUFFERED=1"
                    ],
                    "volumes": [
                        f"{self.base_dir}/data:/app/data",
                        f"{self.base_dir}/logs:/app/logs",
                        f"{self.base_dir}/models:/app/models",
                        f"{self.base_dir}/config:/app/config"
                    ],
                    "restart": self.config.restart_policy,
                    "deploy": {
                        "resources": {
                            "limits": {
                                "memory": self.config.memory_limit,
                                "cpus": self.config.cpu_limit
                            }
                        }
                    },
                    "depends_on": ["redis", "postgres"],
                    "networks": ["sutazai-network"]
                },
                "redis": {
                    "image": "redis:7-alpine",
                    "container_name": f"sutazai-redis-{environment.value}",
                    "ports": ["6379:6379"],
                    "volumes": [f"{self.base_dir}/data/redis:/data"],
                    "restart": self.config.restart_policy,
                    "networks": ["sutazai-network"]
                },
                "postgres": {
                    "image": "postgres:15-alpine",
                    "container_name": f"sutazai-postgres-{environment.value}",
                    "ports": ["5432:5432"],
                    "environment": [
                        "POSTGRES_DB=sutazai",
                        "POSTGRES_USER=sutazai",
                        "POSTGRES_PASSWORD=secure_password_123"
                    ],
                    "volumes": [
                        f"{self.base_dir}/data/postgres:/var/lib/postgresql/data",
                        f"{self.base_dir}/deployment/init.sql:/docker-entrypoint-initdb.d/init.sql"
                    ],
                    "restart": self.config.restart_policy,
                    "networks": ["sutazai-network"]
                },
                "nginx": {
                    "image": "nginx:alpine",
                    "container_name": f"sutazai-nginx-{environment.value}",
                    "ports": ["80:80", "443:443"],
                    "volumes": [
                        f"{self.base_dir}/deployment/nginx.conf:/etc/nginx/nginx.conf",
                        f"{self.base_dir}/deployment/ssl:/etc/ssl/certs"
                    ],
                    "depends_on": ["sutazai-agi"],
                    "restart": self.config.restart_policy,
                    "networks": ["sutazai-network"]
                }
            },
            "networks": {
                "sutazai-network": {
                    "driver": "bridge"
                }
            },
            "volumes": {
                "sutazai-data": {},
                "sutazai-logs": {},
                "sutazai-models": {}
            }
        }
        
        # Add monitoring services for production
        if environment == DeploymentEnvironment.PRODUCTION:
            compose_config["services"]["prometheus"] = {
                "image": "prom/prometheus:latest",
                "container_name": f"sutazai-prometheus-{environment.value}",
                "ports": ["9090:9090"],
                "volumes": [
                    f"{self.base_dir}/deployment/prometheus.yml:/etc/prometheus/prometheus.yml"
                ],
                "restart": self.config.restart_policy,
                "networks": ["sutazai-network"]
            }
            
            compose_config["services"]["grafana"] = {
                "image": "grafana/grafana:latest",
                "container_name": f"sutazai-grafana-{environment.value}",
                "ports": ["3000:3000"],
                "environment": [
                    "GF_SECURITY_ADMIN_PASSWORD=admin"
                ],
                "volumes": [
                    f"{self.base_dir}/data/grafana:/var/lib/grafana"
                ],
                "restart": self.config.restart_policy,
                "networks": ["sutazai-network"]
            }
        
        compose_path = self.deployment_dir / f"docker-compose.{environment.value}.yml"
        with open(compose_path, 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False)
        
        logger.info(f"Created Docker Compose for {environment.value} environment")
        return str(compose_path)
    
    def create_nginx_config(self) -> str:
        """Create Nginx configuration for load balancing"""
        
        nginx_config = """
events {
    worker_connections 1024;
}

http {
    upstream sutazai_backend {
        server sutazai-agi:8000;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        location / {
            proxy_pass http://sutazai_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /health {
            proxy_pass http://sutazai_backend/health;
            access_log off;
        }
    }
}
"""
        
        nginx_path = self.deployment_dir / "nginx.conf"
        nginx_path.write_text(nginx_config.strip())
        
        logger.info("Created Nginx configuration")
        return str(nginx_path)
    
    def create_prometheus_config(self) -> str:
        """Create Prometheus monitoring configuration"""
        
        prometheus_config = {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s"
            },
            "scrape_configs": [
                {
                    "job_name": "sutazai-agi",
                    "static_configs": [
                        {
                            "targets": ["sutazai-agi:8000"]
                        }
                    ],
                    "scrape_interval": "5s",
                    "metrics_path": "/metrics"
                }
            ]
        }
        
        prometheus_path = self.deployment_dir / "prometheus.yml"
        with open(prometheus_path, 'w') as f:
            yaml.dump(prometheus_config, f, default_flow_style=False)
        
        logger.info("Created Prometheus configuration")
        return str(prometheus_path)
    
    def create_database_init(self) -> str:
        """Create database initialization script"""
        
        init_sql = """
-- SutazAI Database Initialization Script
CREATE DATABASE IF NOT EXISTS sutazai;
USE sutazai;

-- Create tables for AGI system
CREATE TABLE IF NOT EXISTS agi_tasks (
    id VARCHAR(32) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    priority INTEGER NOT NULL,
    status VARCHAR(50) NOT NULL,
    data JSON,
    result JSON,
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS neural_nodes (
    id VARCHAR(32) PRIMARY KEY,
    node_type VARCHAR(50) NOT NULL,
    position JSON,
    threshold FLOAT,
    activity FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS neural_links (
    id VARCHAR(32) PRIMARY KEY,
    source_id VARCHAR(32) NOT NULL,
    target_id VARCHAR(32) NOT NULL,
    weight FLOAT NOT NULL,
    link_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES neural_nodes(id),
    FOREIGN KEY (target_id) REFERENCES neural_nodes(id)
);

CREATE TABLE IF NOT EXISTS system_metrics (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cpu_usage FLOAT,
    memory_usage FLOAT,
    neural_activity FLOAT,
    system_health VARCHAR(50),
    tasks_completed INTEGER,
    tasks_failed INTEGER
);

CREATE TABLE IF NOT EXISTS knowledge_graph (
    id VARCHAR(32) PRIMARY KEY,
    entity_type VARCHAR(100) NOT NULL,
    entity_data JSON,
    relationships JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_agi_tasks_status ON agi_tasks(status);
CREATE INDEX idx_agi_tasks_priority ON agi_tasks(priority);
CREATE INDEX idx_neural_nodes_type ON neural_nodes(node_type);
CREATE INDEX idx_neural_links_source ON neural_links(source_id);
CREATE INDEX idx_neural_links_target ON neural_links(target_id);
CREATE INDEX idx_system_metrics_timestamp ON system_metrics(timestamp);
CREATE INDEX idx_knowledge_graph_type ON knowledge_graph(entity_type);

-- Insert default data
INSERT INTO neural_nodes (id, node_type, position, threshold) VALUES 
('input_0', 'input', '{"x": 0, "y": 0}', 0.5),
('hidden_0', 'processing', '{"x": 1, "y": 0}', 0.6),
('output_0', 'output', '{"x": 2, "y": 0}', 0.7);

INSERT INTO neural_links (id, source_id, target_id, weight, link_type) VALUES 
('link_0', 'input_0', 'hidden_0', 0.5, 'excitatory'),
('link_1', 'hidden_0', 'output_0', 0.7, 'excitatory');
"""
        
        init_path = self.deployment_dir / "init.sql"
        init_path.write_text(init_sql.strip())
        
        logger.info("Created database initialization script")
        return str(init_path)
    
    def build_image(self, environment: DeploymentEnvironment) -> str:
        """Build Docker image for the specified environment"""
        try:
            dockerfile_path = self.create_dockerfile(environment)
            
            logger.info(f"Building Docker image for {environment.value}")
            
            # Build image
            image_tag = f"{self.config.image_name}:{environment.value}"
            
            build_command = [
                "docker", "build",
                "-f", dockerfile_path,
                "-t", image_tag,
                str(self.base_dir)
            ]
            
            result = subprocess.run(build_command, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Docker build failed: {result.stderr}")
            
            logger.info(f"Successfully built image: {image_tag}")
            return image_tag
            
        except Exception as e:
            logger.error(f"Failed to build Docker image: {e}")
            raise
    
    def deploy(self, environment: DeploymentEnvironment, build_image: bool = True) -> Dict[str, Any]:
        """Deploy the SutazAI system using Docker Compose"""
        try:
            logger.info(f"Starting deployment for {environment.value} environment")
            
            # Create all configuration files
            dockerfile = self.create_dockerfile(environment)
            compose_file = self.create_docker_compose(environment)
            nginx_config = self.create_nginx_config()
            prometheus_config = self.create_prometheus_config()
            db_init = self.create_database_init()
            
            # Build image if requested
            if build_image:
                image_tag = self.build_image(environment)
            
            # Deploy using Docker Compose
            deploy_command = [
                "docker-compose",
                "-f", compose_file,
                "up", "-d"
            ]
            
            result = subprocess.run(deploy_command, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Docker Compose deployment failed: {result.stderr}")
            
            # Wait for services to be ready
            self._wait_for_services(environment)
            
            logger.info(f"Successfully deployed {environment.value} environment")
            
            return {
                "status": "success",
                "environment": environment.value,
                "image_tag": f"{self.config.image_name}:{environment.value}",
                "compose_file": compose_file,
                "services": self._get_service_status(environment)
            }
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise
    
    def _wait_for_services(self, environment: DeploymentEnvironment, timeout: int = 300):
        """Wait for all services to be ready"""
        logger.info("Waiting for services to be ready...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check if main application is responding
                result = subprocess.run(
                    ["curl", "-f", f"http://localhost:{self.config.expose_port}/health"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    logger.info("Services are ready")
                    return
                
            except Exception:
                pass
            
            time.sleep(5)
        
        raise Exception(f"Services not ready after {timeout} seconds")
    
    def _get_service_status(self, environment: DeploymentEnvironment) -> Dict[str, str]:
        """Get status of all services"""
        try:
            result = subprocess.run(
                ["docker-compose", "-f", f"deployment/docker-compose.{environment.value}.yml", "ps"],
                capture_output=True,
                text=True,
                cwd=str(self.base_dir)
            )
            
            # Parse service status (simplified)
            services = {}
            for line in result.stdout.split('\n')[2:]:  # Skip header
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 3:
                        service_name = parts[0]
                        status = parts[2] if len(parts) > 2 else "unknown"
                        services[service_name] = status
            
            return services
            
        except Exception as e:
            logger.error(f"Failed to get service status: {e}")
            return {}
    
    def stop(self, environment: DeploymentEnvironment) -> Dict[str, Any]:
        """Stop the deployed services"""
        try:
            logger.info(f"Stopping {environment.value} environment")
            
            compose_file = self.deployment_dir / f"docker-compose.{environment.value}.yml"
            
            stop_command = [
                "docker-compose",
                "-f", str(compose_file),
                "down"
            ]
            
            result = subprocess.run(stop_command, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Failed to stop services: {result.stderr}")
            
            logger.info(f"Successfully stopped {environment.value} environment")
            
            return {
                "status": "success",
                "environment": environment.value,
                "message": "Services stopped successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to stop services: {e}")
            raise
    
    def get_logs(self, environment: DeploymentEnvironment, service: str = None) -> str:
        """Get logs from deployed services"""
        try:
            compose_file = self.deployment_dir / f"docker-compose.{environment.value}.yml"
            
            log_command = [
                "docker-compose",
                "-f", str(compose_file),
                "logs"
            ]
            
            if service:
                log_command.append(service)
            
            result = subprocess.run(log_command, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Failed to get logs: {result.stderr}")
            
            return result.stdout
            
        except Exception as e:
            logger.error(f"Failed to get logs: {e}")
            raise

def create_deployment_manager(config: DockerConfig = None) -> DockerDeploymentManager:
    """Create a new deployment manager instance"""
    return DockerDeploymentManager(config)

if __name__ == "__main__":
    # Example usage
    deployment_manager = create_deployment_manager()
    
    # Deploy development environment
    result = deployment_manager.deploy(DeploymentEnvironment.DEVELOPMENT)
    print(f"Deployment result: {json.dumps(result, indent=2)}")
    
    # Get service status
    status = deployment_manager._get_service_status(DeploymentEnvironment.DEVELOPMENT)
    print(f"Service status: {json.dumps(status, indent=2)}")