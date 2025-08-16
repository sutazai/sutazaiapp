# 🔧 DOCKER CLEANUP ACTION PLAN
**Execution Date**: 2025-08-16  
**Estimated Time**: 2-3 hours  
**Risk Level**: Medium (requires testing after each phase)

## 📋 PRE-CLEANUP CHECKLIST
- [ ] Backup current Docker configurations
- [ ] Document which containers are currently running
- [ ] Verify primary docker-compose.yml is in root directory
- [ ] Ensure you can rebuild and restart services

## PHASE 1: BACKUP CURRENT STATE (15 minutes)

```bash
# Create backup directory with timestamp
BACKUP_DIR="/opt/sutazaiapp/backups/docker_cleanup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup all Docker configurations
cp -r /opt/sutazaiapp/docker "$BACKUP_DIR/docker_original"
cp /opt/sutazaiapp/docker-compose.yml "$BACKUP_DIR/"
cp /opt/sutazaiapp/Makefile "$BACKUP_DIR/"

# Document current running state
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" > "$BACKUP_DIR/running_containers.txt"
docker images | grep sutazai > "$BACKUP_DIR/current_images.txt"

echo "Backup created at: $BACKUP_DIR"
```

## PHASE 2: IDENTIFY AND ARCHIVE UNUSED COMPOSE FILES (30 minutes)

### Files to KEEP:
```bash
/opt/sutazaiapp/docker-compose.yml              # PRIMARY - Currently active
/opt/sutazaiapp/docker/docker-compose.override.yml  # Development overrides
/opt/sutazaiapp/docker/docker-compose.mcp.yml       # MCP services (referenced in deploy.sh)
```

### Files to ARCHIVE:
```bash
# Create archive directory
mkdir -p /opt/sutazaiapp/docker/archive/compose-files

# Move unused compose files
cd /opt/sutazaiapp/docker
mv docker-compose.base.yml archive/compose-files/
mv docker-compose.blue-green.yml archive/compose-files/
mv docker-compose.mcp-legacy.yml archive/compose-files/
mv docker-compose.mcp-monitoring.yml archive/compose-files/
mv docker-compose.memory-optimized.yml archive/compose-files/
mv docker-compose.minimal.yml archive/compose-files/
mv docker-compose.optimized.yml archive/compose-files/
mv docker-compose.override-legacy.yml archive/compose-files/
mv docker-compose.performance.yml archive/compose-files/
mv docker-compose.public-images.override.yml archive/compose-files/
mv docker-compose.secure-legacy.yml archive/compose-files/
mv docker-compose.secure.hardware-optimizer.yml archive/compose-files/
mv docker-compose.secure.yml archive/compose-files/
mv docker-compose.security-monitoring.yml archive/compose-files/
mv docker-compose.standard.yml archive/compose-files/
mv docker-compose.ultra-performance.yml archive/compose-files/
mv docker-compose.yml archive/compose-files/docker-compose.duplicate.yml  # Duplicate of root
```

## PHASE 3: CONSOLIDATE DOCKERFILES (45 minutes)

### Step 1: Identify Active Dockerfiles
```bash
# These are the Dockerfiles actually being used based on running containers:
/opt/sutazaiapp/docker/backend/Dockerfile
/opt/sutazaiapp/docker/frontend/Dockerfile
/opt/sutazaiapp/docker/agents/ultra-system-architect/Dockerfile
/opt/sutazaiapp/docker/faiss/Dockerfile.standalone
```

### Step 2: Archive Duplicate/Unused Dockerfiles
```bash
# Create archive for duplicate Dockerfiles
mkdir -p /opt/sutazaiapp/docker/archive/dockerfiles

# Archive duplicate agent Dockerfiles
cd /opt/sutazaiapp/docker/agents/ai_agent_orchestrator
mv Dockerfile.optimized ../../archive/dockerfiles/ai_agent_orchestrator.optimized.Dockerfile
mv Dockerfile.secure ../../archive/dockerfiles/ai_agent_orchestrator.secure.Dockerfile

cd /opt/sutazaiapp/docker/agents/hardware-resource-optimizer
mv Dockerfile.optimized ../../archive/dockerfiles/hardware-resource-optimizer.optimized.Dockerfile
mv Dockerfile.standalone ../../archive/dockerfiles/hardware-resource-optimizer.standalone.Dockerfile

# Archive unused base Dockerfiles (keep only essential ones)
cd /opt/sutazaiapp/docker/base
mkdir -p ../archive/dockerfiles/base
mv Dockerfile.*-secure ../archive/dockerfiles/base/  # Archive all secure variants
```

### Step 3: Standardize Remaining Dockerfiles
```bash
# Ensure each service has ONE Dockerfile with multi-stage builds
# Update Dockerfiles to use multi-stage pattern:
# - Stage 1: base
# - Stage 2: development
# - Stage 3: production
```

## PHASE 4: FIX NAMING CONSISTENCY (30 minutes)

### Current State:
- docker-compose.yml uses: `sutazaiapp-*:v1.0.0`
- Makefile builds: `sutazai-*:latest`
- Build scripts create: various names

### Standardization:
```bash
# Decision: Use sutazaiapp-* for all services

# Update Makefile
sed -i 's/sutazai-backend/sutazaiapp-backend/g' /opt/sutazaiapp/Makefile
sed -i 's/sutazai-frontend/sutazaiapp-frontend/g' /opt/sutazaiapp/Makefile

# Update build scripts
find /opt/sutazaiapp/docker/scripts -name "*.sh" -exec sed -i 's/sutazai-/sutazaiapp-/g' {} \;

# Standardize version tags
# Use :latest for development
# Use :v1.0.0 for production (increment as needed)
```

## PHASE 5: REORGANIZE DOCKER DIRECTORY (30 minutes)

### Target Structure:
```bash
/opt/sutazaiapp/docker/
├── README.md                       # Explains the structure
├── docker-compose.override.yml     # Dev overrides
├── docker-compose.mcp.yml         # MCP services
├── .env.example                    # Environment template
├── services/
│   ├── backend/
│   │   └── Dockerfile
│   ├── frontend/
│   │   └── Dockerfile
│   ├── faiss/
│   │   └── Dockerfile
│   └── agents/
│       ├── ultra-system-architect/
│       │   └── Dockerfile
│       └── [active agents only]/
├── base/
│   └── python/
│       └── Dockerfile              # Single base image
├── scripts/
│   ├── build.sh                   # Unified build script
│   └── cleanup.sh                 # Cleanup script
└── archive/                       # All old/unused files
```

### Execute Reorganization:
```bash
cd /opt/sutazaiapp/docker

# Create new structure
mkdir -p services/backend services/frontend services/faiss services/agents
mkdir -p base/python scripts

# Move active Dockerfiles to new locations
mv backend/Dockerfile services/backend/
mv frontend/Dockerfile services/frontend/
mv faiss/Dockerfile.standalone services/faiss/Dockerfile
mv agents/ultra-system-architect services/agents/

# Move scripts
mv scripts/build_all_images.sh scripts/build.sh
```

## PHASE 6: UPDATE REFERENCES (30 minutes)

### Update docker-compose.yml:
```yaml
# Change build contexts to new structure
services:
  backend:
    build:
      context: .
      dockerfile: docker/services/backend/Dockerfile
    image: sutazaiapp-backend:${VERSION:-latest}

  frontend:
    build:
      context: .
      dockerfile: docker/services/frontend/Dockerfile
    image: sutazaiapp-frontend:${VERSION:-latest}
```

### Update Makefile:
```makefile
# Update Docker build commands
docker-build:
	docker build -t sutazaiapp-backend:latest -f docker/services/backend/Dockerfile .
	docker build -t sutazaiapp-frontend:latest -f docker/services/frontend/Dockerfile .
```

## PHASE 7: TESTING (30 minutes)

```bash
# 1. Stop current services
docker compose down

# 2. Rebuild with new structure
make docker-build

# 3. Start services
docker compose up -d

# 4. Verify all services are running
docker ps
make status

# 5. Test basic functionality
curl http://localhost:10010/health  # Backend
curl http://localhost:10011         # Frontend
```

## PHASE 8: DOCUMENTATION (15 minutes)

### Create /opt/sutazaiapp/docker/README.md:
```markdown
# Docker Configuration Structure

## Active Files
- `docker-compose.override.yml` - Local development overrides
- `docker-compose.mcp.yml` - MCP server configurations
- `services/` - All service Dockerfiles

## Naming Convention
- Images: `sutazaiapp-{service}:{version}`
- Containers: `sutazai-{service}`
- Networks: `sutazai-network`

## Build Commands
- `make docker-build` - Build all images
- `docker/scripts/build.sh` - Build specific images

## Archived Files
Old and unused configurations are in `archive/` for reference.
```

## PHASE 9: CLEANUP (15 minutes)

```bash
# Remove dangling images
docker image prune -f

# Clean build cache
docker builder prune -f

# Remove unused volumes
docker volume prune -f

# Final disk usage check
docker system df
```

## 🎯 SUCCESS VERIFICATION

### Checklist:
- [ ] Only 3 docker-compose files remain in active use
- [ ] Each service has ONE Dockerfile
- [ ] Image naming is consistent (sutazaiapp-*)
- [ ] All services start successfully
- [ ] Documentation reflects reality
- [ ] Archive contains all old files for reference
- [ ] No duplicate configurations exist

### Test Commands:
```bash
# Verify services are running
docker ps | grep sutazai

# Check image naming
docker images | grep sutazaiapp

# Verify build works
make docker-build

# Check compose files
ls -la /opt/sutazaiapp/docker/*.yml
```

## 🚫 ROLLBACK PLAN

If anything goes wrong:
```bash
# Restore from backup
BACKUP_DIR="/opt/sutazaiapp/backups/docker_cleanup_[timestamp]"
cp -r "$BACKUP_DIR/docker_original" /opt/sutazaiapp/docker
cp "$BACKUP_DIR/docker-compose.yml" /opt/sutazaiapp/
cp "$BACKUP_DIR/Makefile" /opt/sutazaiapp/

# Restart services
docker compose up -d
```

## 📝 NOTES

1. **MCP Services**: docker-compose.mcp.yml is referenced in deploy.sh, keep it
2. **Blue-Green**: Referenced but not implemented, can be archived
3. **Version Tags**: Standardize on semantic versioning (v1.0.0, v1.0.1, etc.)
4. **Base Images**: Consolidate to single python base with build args for variants

---
**Execute this plan step-by-step, testing after each phase to ensure system stability.**