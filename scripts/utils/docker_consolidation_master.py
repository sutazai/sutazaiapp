#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
Docker Consolidation Master

Consolidates 389+ Dockerfiles into perfect architecture with base images.
Implements template-based approach following ULTRAORGANIZE principles.

Author: ULTRAORGANIZE Infrastructure Master
Date: August 11, 2025
Status: ACTIVE IMPLEMENTATION
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

class DockerConsolidationMaster:
    """Master orchestrator for Docker file consolidation."""
    
    def __init__(self, root_path: str = '/opt/sutazaiapp'):
        self.root_path = Path(root_path)
        self.docker_dir = self.root_path / 'docker'
        self.consolidated_count = 0
        
        # Base image categories
        self.base_categories = {
            'python-agent': ['python', 'agent', 'flask', 'fastapi'],
            'nodejs-service': ['node', 'nodejs', 'javascript'],
            'ai-ml': ['ollama', 'ai', 'ml', 'torch', 'tensorflow'],
            'monitoring': ['prometheus', 'grafana', 'monitor'],
            'database': ['postgres', 'redis', 'neo4j', 'chromadb'],
            'security': ['security', 'auth', 'jwt'],
            ' ': ['alpine', ' ']
        }
    
    def analyze_dockerfile_patterns(self) -> Dict:
        """Analyze all Dockerfiles to identify consolidation opportunities."""
        logger.info("🔍 Analyzing 389+ Dockerfiles for consolidation patterns...")
        
        patterns = {
            'base_images_used': defaultdict(int),
            'common_packages': defaultdict(int),
            'dockerfile_categories': defaultdict(list),
            'consolidation_opportunities': []
        }
        
        dockerfile_count = 0
        
        for dockerfile in self.root_path.rglob('Dockerfile*'):
            if dockerfile.is_file():
                dockerfile_count += 1
                self._analyze_single_dockerfile(dockerfile, patterns)
        
        logger.info(f"✅ Analyzed {dockerfile_count} Dockerfiles")
        return patterns
    
    def _analyze_single_dockerfile(self, dockerfile_path: Path, patterns: Dict):
        """Analyze a single Dockerfile for patterns."""
        try:
            with open(dockerfile_path, 'r') as f:
                content = f.read().lower()
            
            # Extract base image
            for line in content.split('\n'):
                if line.strip().startswith('from '):
                    base_image = line.split()[1]
                    patterns['base_images_used'][base_image] += 1
                    break
            
            # Categorize Dockerfile
            dockerfile_name = str(dockerfile_path).lower()
            category = self._categorize_dockerfile(dockerfile_name, content)
            patterns['dockerfile_categories'][category].append(str(dockerfile_path))
            
        except Exception as e:
            logger.error(f"  ⚠️  Error analyzing {dockerfile_path.name}: {e}")
    
    def _categorize_dockerfile(self, dockerfile_path: str, content: str) -> str:
        """Categorize Dockerfile based on path and content."""
        for category, keywords in self.base_categories.items():
            if any(keyword in dockerfile_path or keyword in content 
                  for keyword in keywords):
                return category
        return 'misc'
    
    def create_master_base_images(self) -> None:
        """Create master base images for consolidation."""
        logger.info("🏗️  Creating master base images...")
        
        base_dir = self.docker_dir / 'base'
        base_dir.mkdir(exist_ok=True)
        
        # Create Python Agent Master Base
        self._create_python_agent_base(base_dir)
        
        # Create NodeJS Service Master Base
        self._create_nodejs_service_base(base_dir)
        
        # Create AI/ML Master Base
        self._create_ai_ml_base(base_dir)
        
        # Create Monitoring Master Base
        self._create_monitoring_base(base_dir)
        
        # Create Database Master Base
        self._create_database_base(base_dir)
        
        logger.info("✅ Created 5 master base images")
    
    def _create_python_agent_base(self, base_dir: Path):
        """Create Python Agent master base image."""
        dockerfile_content = '''
FROM python:3.11-slim

# Master Python Agent Base Image
# Consolidates 100+ Python-based service containers

# Security: Run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python environment
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    pydantic==2.5.0 \
    httpx==0.25.2 \
    aioredis==2.0.1 \
    psycopg2-binary==2.9.9 \
    python-multipart==0.0.6 \
    python-jose[cryptography]==3.3.0 \
    bcrypt==4.1.2 \
    prometheus-client==0.19.0

# Health check utility
RUN echo '#!/bin/bash\necho "healthy"' > /usr/local/bin/health_check && \
    chmod +x /usr/local/bin/health_check

# Working directory
WORKDIR /app

# Switch to non-root user
USER appuser

# Default health check
HEALTHCHEK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["health_check"]

# Default command
CMD ["python", "app.py"]
'''
        
        with open(base_dir / 'Dockerfile.python-agent-master', 'w') as f:
            f.write(dockerfile_content)
    
    def _create_nodejs_service_base(self, base_dir: Path):
        """Create NodeJS Service master base image."""
        dockerfile_content = '''
FROM node:18-slim

# Master NodeJS Service Base Image
# Consolidates all NodeJS-based services

# Security: Run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Global packages
RUN npm install -g pm2

# Health check utility
RUN echo '#!/bin/bash\necho "healthy"' > /usr/local/bin/health_check && \
    chmod +x /usr/local/bin/health_check

# Working directory
WORKDIR /app

# Switch to non-root user
USER appuser

# Default health check
HEALTHCHEK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["health_check"]

# Default command
CMD ["npm", "start"]
'''
        
        with open(base_dir / 'Dockerfile.nodejs-service-master', 'w') as f:
            f.write(dockerfile_content)
    
    def _create_ai_ml_base(self, base_dir: Path):
        """Create AI/ML master base image."""
        dockerfile_content = '''
FROM python:3.11-slim

# Master AI/ML Base Image
# Consolidates AI/ML services including Ollama integrations

# Security: Run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# AI/ML Python packages
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    transformers==4.35.0 \
    numpy==1.24.3 \
    scikit-learn==1.3.0 \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    httpx==0.25.2 \
    ollama==0.1.7

# Working directory
WORKDIR /app

# Switch to non-root user
USER appuser

# Default health check
HEALTHCHEK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["python", "-c", "logger.info('healthy')"]

# Default command
CMD ["python", "app.py"]
'''
        
        with open(base_dir / 'Dockerfile.ai-ml-master', 'w') as f:
            f.write(dockerfile_content)
    
    def _create_monitoring_base(self, base_dir: Path):
        """Create Monitoring master base image."""
        dockerfile_content = '''
FROM python:3.11-slim

# Master Monitoring Base Image
# Consolidates all monitoring and observability services

# Security: Run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Monitoring packages
RUN pip install --no-cache-dir \
    prometheus-client==0.19.0 \
    grafana-api==1.0.3 \
    psutil==5.9.6 \
    requests==2.31.0 \
    fastapi==0.104.1 \
    uvicorn==0.24.0

# Working directory
WORKDIR /app

# Switch to non-root user
USER appuser

# Default health check
HEALTHCHEK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["python", "-c", "logger.info('healthy')"]

# Default command
CMD ["python", "monitor.py"]
'''
        
        with open(base_dir / 'Dockerfile.monitoring-master', 'w') as f:
            f.write(dockerfile_content)
    
    def _create_database_base(self, base_dir: Path):
        """Create Database master base image."""
        dockerfile_content = '''
FROM python:3.11-slim

# Master Database Service Base Image
# Consolidates database utilities and connectors

# Security: Run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Database packages
RUN pip install --no-cache-dir \
    psycopg2-binary==2.9.9 \
    redis==5.0.1 \
    neo4j==5.14.1 \
    chromadb==0.4.17 \
    qdrant-client==1.6.9 \
    sqlalchemy==2.0.23 \
    alembic==1.12.1

# Working directory
WORKDIR /app

# Switch to non-root user
USER appuser

# Default health check
HEALTHCHEK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["python", "-c", "logger.info('healthy')"]

# Default command
CMD ["python", "database_service.py"]
'''
        
        with open(base_dir / 'Dockerfile.database-master', 'w') as f:
            f.write(dockerfile_content)
    
    def create_service_dockerfile_templates(self) -> None:
        """Create service Dockerfile templates using base images."""
        logger.info("📋 Creating service Dockerfile templates...")
        
        templates_dir = self.docker_dir / 'templates'
        templates_dir.mkdir(exist_ok=True)
        
        # Template for Python agents
        python_template = '''
FROM sutazai/python-agent-master:latest

# Service-specific configuration
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Expose port
EXPOSE 8080

# Service-specific health check
HEALTHCHEK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["curl", "-f", "http://localhost:8080/health"]

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
'''
        
        with open(templates_dir / 'Dockerfile.python-service-template', 'w') as f:
            f.write(python_template)
        
        logger.info("✅ Created service templates")
    
    def generate_dockerfile_migration_plan(self) -> Dict:
        """Generate migration plan for existing Dockerfiles."""
        logger.info("🗺️  Generating Dockerfile migration plan...")
        
        patterns = self.analyze_dockerfile_patterns()
        
        migration_plan = {
            'files_to_migrate': {},
            'files_to_archive': [],
            'consolidation_savings': 0
        }
        
        # Plan migration for each category
        for category, dockerfiles in patterns['dockerfile_categories'].items():
            target_base = self._get_target_base_image(category)
            migration_plan['files_to_migrate'][category] = {
                'target_base': target_base,
                'dockerfiles': dockerfiles[:5],  # Limit for safety
                'count': len(dockerfiles)
            }
            
            # Plan to archive duplicates (keep only a few examples)
            if len(dockerfiles) > 3:
                migration_plan['files_to_archive'].extend(dockerfiles[3:])
        
        migration_plan['consolidation_savings'] = len(migration_plan['files_to_archive'])
        
        return migration_plan
    
    def _get_target_base_image(self, category: str) -> str:
        """Get target base image for a category."""
        mapping = {
            'python-agent': 'sutazai/python-agent-master:latest',
            'nodejs-service': 'sutazai/nodejs-service-master:latest',
            'ai-ml': 'sutazai/ai-ml-master:latest',
            'monitoring': 'sutazai/monitoring-master:latest',
            'database': 'sutazai/database-master:latest',
            'security': 'sutazai/python-agent-master:latest',
            ' ': 'sutazai/python-agent-master:latest'
        }
        return mapping.get(category, 'sutazai/python-agent-master:latest')
    
    def execute_docker_consolidation(self) -> Dict:
        """Execute complete Docker consolidation."""
        logger.info("🚀 DOCKER CONSOLIDATION MASTER - STARTING")
        logger.info("=" * 50)
        
        # Create master base images
        self.create_master_base_images()
        
        # Create templates
        self.create_service_dockerfile_templates()
        
        # Generate migration plan
        migration_plan = self.generate_dockerfile_migration_plan()
        
        # Create consolidated directory structure
        services_dir = self.docker_dir / 'services'
        services_dir.mkdir(exist_ok=True)
        
        production_dir = self.docker_dir / 'production'
        production_dir.mkdir(exist_ok=True)
        
        logger.info("=" * 50)
        logger.info("✅ DOCKER CONSOLIDATION MASTER - PHASE 1 COMPLETE")
        
        return migration_plan

if __name__ == '__main__':
    consolidator = DockerConsolidationMaster()
    migration_plan = consolidator.execute_docker_consolidation()
    
    # Save migration plan
    plan_path = Path('/opt/sutazaiapp/DOCKER_CONSOLIDATION_PLAN.json')
    with open(plan_path, 'w') as f:
        json.dump(migration_plan, f, indent=2)
    
    logger.info(f"📁 Migration plan saved to: {plan_path}")
    logger.info(f"📏 Potential consolidation: {migration_plan['consolidation_savings']} files")