#!/usr/bin/env python3
"""
SutazAI Container Security Fixer
Fixes container security issues by adding non-root users to Dockerfiles
"""

import os
from pathlib import Path


def fix_dockerfile_security(dockerfile_path):
    """Fix a single Dockerfile to run as non-root user"""
    
    try:
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # Skip if already has non-root user setup
        if 'USER appuser' in content or 'USER 1001' in content:
            print(f"✅ {dockerfile_path} already secured")
            return True
            
        # Check if it has USER root
        has_root_user = 'USER root' in content
        
        # Security fix template
        security_fix = """
# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \\
    && mkdir -p /app && chown -R appuser:appuser /app
USER appuser
"""

        # Remove USER root if present
        if has_root_user:
            content = content.replace('USER root', '# Removed USER root for security')
        
        # Add security fix before the last line (usually CMD or ENTRYPOINT)
        lines = content.split('\n')
        
        # Find the last CMD, ENTRYPOINT, or WORKDIR line
        insert_index = len(lines) - 1
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith(('CMD', 'ENTRYPOINT', 'EXPOSE')):
                insert_index = i
                break
        
        # Insert security fix
        lines.insert(insert_index, security_fix)
        
        # Write back the fixed content
        with open(dockerfile_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"🔒 Fixed security in {dockerfile_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing {dockerfile_path}: {e}")
        return False


def main():
    """Fix all Dockerfiles in the project"""
    
    project_root = Path(__file__).parent.parent
    dockerfiles = []
    
    # Find all Dockerfiles
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file == 'Dockerfile' or file.startswith('Dockerfile.'):
                dockerfiles.append(Path(root) / file)
    
    print(f"🔍 Found {len(dockerfiles)} Dockerfile(s) to check")
    
    fixed_count = 0
    for dockerfile in dockerfiles:
        if fix_dockerfile_security(dockerfile):
            fixed_count += 1
    
    print(f"\n🎉 Security fixes applied to {fixed_count}/{len(dockerfiles)} Dockerfiles")
    
    # Create docker-compose security override
    create_docker_compose_security_override(project_root)


def create_docker_compose_security_override(project_root):
    """Create docker-compose override for security hardening"""
    
    override_content = """# Docker Compose Security Override
# This file applies security hardening to all services
version: '3.8'

services:
  # Apply security settings to all database services
  postgres:
    environment:
      - POSTGRES_HOST_AUTH_METHOD=md5  # Require password authentication
    user: "999:999"  # Run as postgres user, not root
    
  redis:
    user: "999:999"  # Run as redis user, not root
    command: ["redis-server", "--requirepass", "${REDIS_PASSWORD}"]
    
  neo4j:
    user: "7474:7474"  # Run as neo4j user, not root
    
  # Security settings for application services  
  backend:
    user: "1001:1001"
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
      
  frontend:
    user: "1001:1001" 
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
    cap_drop:
      - ALL
      
  # Apply to monitoring services
  grafana:
    user: "472:472"  # Grafana user
    
  prometheus:
    user: "65534:65534"  # Nobody user
    
  # Agent security hardening
  ai-agent-orchestrator:
    user: "1001:1001"
    cap_drop:
      - ALL
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=50m
"""

    override_path = project_root / 'docker-compose.security.yml'
    
    with open(override_path, 'w') as f:
        f.write(override_content)
    
    print(f"🛡️  Created security override: {override_path}")
    print("💡 Use with: docker-compose -f docker-compose.yml -f docker-compose.security.yml up")


if __name__ == '__main__':
    main()