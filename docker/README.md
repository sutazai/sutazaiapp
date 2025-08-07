# SutazAI Docker Architecture

This directory contains the Docker configuration for the SutazAI automation system according to Docker Excellence standards.

## Directory Structure

```
/docker
├── README.md                    # This file - Docker architecture overview
├── base/                        # Base images for reuse
│   ├── python-base.Dockerfile   # Common Python base with optimizations
│   ├── agent-base.Dockerfile    # Base for all AI agents
│   └── security-base.Dockerfile # Hardened base for security tools
├── services/                    # Service-specific Dockerfiles
│   ├── frontend/                # Frontend service Docker files
│   ├── backend/                 # Backend service Docker files
│   ├── agents/                  # Individual agent Dockerfiles
│   ├── monitoring/              # Monitoring stack Dockerfiles
│   └── infrastructure/          # Core infrastructure Dockerfiles
└── compose/                     # Docker Compose configurations
    ├── docker-compose.yml       # Production configuration
    ├── docker-compose.dev.yml   # Development overrides
    ├── docker-compose.test.yml  # Testing environment
    └── docker-compose.agents.yml # Agent-specific services
```

## Docker Excellence Standards

### 1. Multi-Stage Builds
All Dockerfiles use multi-stage builds to minimize image size and improve security.

### 2. Non-Root Users
All containers run as non-privileged users for security.

### 3. Health Checks
Every service includes proper health checks for reliability.

### 4. Security Hardening
- Regular security updates
- Minimal attack surface
- Proper secret management
- Network isolation

### 5. Optimization
- Layer caching optimization
- Minimal base images
- Efficient dependency management

## Usage

### Production Deployment
```bash
cd /opt/sutazaiapp
docker-compose -f docker/compose/docker-compose.yml up -d
```

### Development Environment
```bash
cd /opt/sutazaiapp
docker-compose -f docker/compose/docker-compose.yml -f docker/compose/docker-compose.dev.yml up -d
```

### Testing Environment
```bash
cd /opt/sutazaiapp
docker-compose -f docker/compose/docker-compose.test.yml up -d
```

### Agent Services Only
```bash
cd /opt/sutazaiapp
docker-compose -f docker/compose/docker-compose.agents.yml up -d
```

## Best Practices

1. **Image Naming**: Follow semantic versioning for image tags
2. **Resource Limits**: All services have appropriate CPU/memory limits
3. **Health Monitoring**: Comprehensive health check coverage
4. **Security**: Regular vulnerability scanning and updates
5. **Documentation**: All Dockerfiles are well-documented

## Architecture Principles

- **Microservices**: Each component is containerized independently
- **Scalability**: Horizontal scaling support for all services
- **Reliability**: Fault tolerance and automatic recovery
- **Security**: Defense in depth approach
- **Performance**: Optimized for CPU-only environments
- **Maintainability**: Clear structure and documentation