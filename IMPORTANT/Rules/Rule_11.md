Rule 11: Docker Excellence
Docker Standards:
All Docker diagrams must include the complete directory structure
Reference these specific diagrams in /opt/sutazaiapp/IMPORTANT/diagrams:

Dockerdiagram-core.md - Core container architecture
Dockerdiagram-self-coding.md - Self-coding service containers
Dockerdiagram-training.md - Training environment containers
Dockerdiagram.md - Main Docker architecture overview
PortRegistry.md - Port allocation and service registry

Every container architecture must follow the structure shown in /opt/sutazaiapp/IMPORTANT/diagrams
All Docker configurations go in /docker/ but reference the diagrams for architecture decisions
Port assignments must follow PortRegistry.md specifications
✅ Required Practices:
Configuration Standards:

All Docker configurations centralized in /docker/ directory only
Reference architecture diagrams in /opt/sutazaiapp/IMPORTANT/diagrams before any container changes
Multi-stage Dockerfiles with development and production variants
Non-root user execution with proper USER directives (never run as root)
Pinned base image versions (never use latest tags)
Comprehensive HEALTHCHECK instructions for all services
Proper .dockerignore files to optimize build context
Docker Compose files for each environment (dev/staging/prod)

Security & Scanning:

Container vulnerability scanning in CI/CD pipeline
Secrets managed externally (never in images or ENV vars)
Security validation before any production deployment
Read-only root filesystem where applicable
Capability dropping with minimal required permissions

Resource & Performance:

Resource limits and requests defined for all containers
Memory limits: Explicit values (e.g., 512m)
CPU limits: Decimal notation (e.g., cpus: '0.5')
Structured logging with proper log levels and formats
Log rotation configured to prevent disk exhaustion

Orchestration & Integration:

Container orchestration with proper service mesh integration
Service discovery through Consul or equivalent
Network isolation using Docker networks per environment
Inter-service communication via service mesh
Health checks integrated with orchestration platform

🚫 Forbidden Practices:
Configuration Violations:

Creating Docker files outside /docker/ directory
Using latest or unpinned image tags in any environment
Running containers as root user without explicit security review
Storing secrets, credentials, or sensitive data in container images

Build & Deployment Violations:

Building images without vulnerability scanning and security validation
Creating monolithic containers that violate single responsibility principle
Using development configurations or debugging tools in production images
Deploying without comprehensive security scanning

Operational Violations:

Implementing containers without proper health checks and monitoring
Creating containers without proper resource limits and quotas
Using containers that don't handle graceful shutdown (SIGTERM)
Ignoring service mesh integration requirements

Cross-Rule References:

See Rule 1: All Docker configurations must be real and working
See Rule 4: Investigate existing Docker configurations before creating new ones
See Rule 12: Docker setup integrated with universal deployment script
See Rule 19: Document all Docker changes in CHANGELOG.md

Validation Criteria:

All containers pass security scans with zero high-severity vulnerabilities
Docker configurations follow established patterns in /docker/ directory
Architecture decisions align with diagrams in /opt/sutazaiapp/IMPORTANT/diagrams
Containers start reliably and handle graceful shutdown
Resource usage is optimized and within defined limits
All services have functional health checks and monitoring
Documentation is current and matches actual container behavior
Service mesh integration verified and functional
Container orchestration properly configured
Security scanning integrated into CI/CD pipeline

Implementation Checklist:
bash# Pre-deployment validation
□ Dockerfile in /docker/ directory
□ Multi-stage build implemented
□ Base image version pinned
□ USER directive sets non-root (1000:1000)
□ HEALTHCHECK defined with timings
□ .dockerignore optimized
□ Resource limits defined
□ Security scan passed
□ Secrets externalized
□ Graceful shutdown tested
□ Service mesh configured
□ Port matches PortRegistry.md
□ Architecture matches diagrams
□ CHANGELOG.md updated


*Last Updated: 2025-08-30 00:00:00 UTC - For the infrastructure based in /opt/sutazaiapp/