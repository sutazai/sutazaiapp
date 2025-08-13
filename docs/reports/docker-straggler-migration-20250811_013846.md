# Docker Stragglers Migration Report

Generated: Mon Aug 11 01:38:46 CEST 2025
Author: DOCKER-MASTER-001

## Migration Categories

### Python Services to Migrate (2 files)
- /opt/sutazaiapp/docker/base/Dockerfile.python-agent- 
- /opt/sutazaiapp/docker/skyvern/Dockerfile

### Node.js Services to Migrate (0 files)
- 

### Alpine Services to Migrate (3 files)
- /opt/sutazaiapp/docker/hygiene-reporter/Dockerfile
- /opt/sutazaiapp/docker/nginx/Dockerfile
- /opt/sutazaiapp/docker/hygiene-dashboard/Dockerfile

### GPU Services to Migrate (4 files)
- /opt/sutazaiapp/docker/base/Dockerfile.gpu-python-base
- /opt/sutazaiapp/docker/tensorflow/Dockerfile
- /opt/sutazaiapp/docker/tabbyml/Dockerfile
- /opt/sutazaiapp/docker/pytorch/Dockerfile

### Infrastructure Services (Skipped - 6 files)
- /opt/sutazaiapp/docker/chromadb-secure/Dockerfile
- /opt/sutazaiapp/docker/qdrant-secure/Dockerfile
- /opt/sutazaiapp/docker/redis-secure/Dockerfile
- /opt/sutazaiapp/docker/neo4j-secure/Dockerfile
- /opt/sutazaiapp/docker/postgres-secure/Dockerfile
- /opt/sutazaiapp/docker/rabbitmq-secure/Dockerfile

## Migration Strategy
1. Python services -> sutazai-python-agent-master:latest
2. Node.js services -> sutazai-nodejs-agent-master:latest  
3. Alpine services -> sutazai-python-alpine-optimized:latest
4. GPU services -> sutazai-ai-ml-gpu:latest or sutazai-ai-ml-cpu:latest
5. Infrastructure services -> Keep existing (specialized databases)

