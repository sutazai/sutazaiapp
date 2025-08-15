# SutazAI Docker Architecture

This directory contains the Docker configuration for the SutazAI automation system according to Docker Excellence standards.

## Consolidated Structure (Updated 2025-08-15)

Docker Compose files have been reorganized from 19 files into a logical hierarchy:

### Core Files
- `docker-compose.yml` - Main production configuration (36KB)
- `docker-compose.base.yml` - Base service definitions (5.5KB)
- `docker-compose.override.yml` - Local development overrides

### Environment-Specific
- `docker-compose.minimal.yml` - Minimal setup for testing
- `docker-compose.standard.yml` - Standard deployment
- `docker-compose.optimized.yml` - Optimized production deployment

### Feature-Specific
- `docker-compose.mcp.yml` - MCP server configuration
- `docker-compose.mcp-monitoring.yml` - MCP monitoring stack
- `docker-compose.security-monitoring.yml` - Security monitoring stack

### Performance & Security
- `docker-compose.performance.yml` - Performance optimizations
- `docker-compose.ultra-performance.yml` - Maximum performance settings
- `docker-compose.secure.yml` - Security-hardened configuration
- `docker-compose.secure.hardware-optimizer.yml` - Secure hardware optimizer

### Deployment Strategies
- `docker-compose.blue-green.yml` - Blue-green deployment setup

### External Integrations
- `docker-compose.skyvern.yml` - Skyvern integration
- `docker-compose.documind.override.yml` - Documind override

## Docker Excellence Standards

### 1. Multi-Stage Builds
All Dockerfiles use multi-stage builds to minimize image size and improve security.

### 2. Non-Root Users
All containers run as non-privileged users for security.

### 3. Health Checks
Every service includes proper health checks for reliability.

### 4. Security Hardening
- Regular security updates
-   attack surface
- Proper secret management
- Network isolation

### 5. Optimization
- Layer caching optimization
-   base images
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