# Hygiene Monitoring System - Port Mapping Strategy

## Current Port Conflicts Analysis

### Identified Conflicts
1. **PostgreSQL Database**: 
   - Current hygiene-monitor config: `5433:5432`
   - Main system PostgreSQL already using: `5432:5432`
   - **Status**: ✅ NO CONFLICT (using different host port)

2. **Nginx Reverse Proxy**:
   - Current hygiene-monitor config: `8082:80`
   - System nginx already using: `80:80`
   - **Status**: ✅ NO CONFLICT (using different host port)

3. **Frontend Dashboard**:
   - Current hygiene-monitor config: `3002:3000`
   - Main system using: `3000:3000` (sutazai-grafana), `3001:3001` (sutazai-grafana)
   - **Status**: ✅ NO CONFLICT (using different host port)

4. **Backend API**:
   - Current hygiene-monitor config: `8081:8080`
   - Main system using: `8000:8000`, `8001:8000`, `8003:8003`
   - **Status**: ✅ NO CONFLICT (using different host port)

5. **Rule Control API**:
   - Current hygiene-monitor config: `8101:8100`
   - **Status**: ✅ NO CONFLICT (port available)

### Currently Occupied Ports
```
Port 22     - SSH (system)
Port 53     - DNS (systemd-resolved)
Port 80     - Nginx (system)
Port 3000   - Docker proxy (sutazai frontend)
Port 3001   - Docker proxy (sutazai grafana)
Port 3002   - Docker proxy (hygiene-dashboard) ✅
Port 5432   - Docker proxy (sutazai-postgres)
Port 6333   - Docker proxy (sutazai-qdrant)
Port 6334   - Docker proxy (sutazai-qdrant)
Port 7474   - Docker proxy (sutazai-neo4j)
Port 7687   - Docker proxy (sutazai-neo4j)
Port 8000   - Docker proxy (sutazai-backend)
Port 8001   - Docker proxy (sutazai-chromadb)
Port 8003   - Docker proxy (sutazai-storage)
Port 8081   - Docker proxy (hygiene-backend) ✅
Port 8082   - Docker proxy (hygiene-nginx) ✅
Port 8101   - Docker proxy (rule-control-api) ✅
Port 8116   - Docker proxy (hardware-resource-optimizer)
Port 9090   - Docker proxy (sutazai-prometheus)
Port 11434  - Docker proxy (sutazai-ollama)
```

### Available Port Ranges
- **3010-3050**: Available for frontend services
- **5400-5450**: Available for database services
- **8200-8250**: Available for API/backend services

## Recommended Port Mapping Strategy

### Current Configuration Assessment
The current hygiene monitoring port configuration is **ALREADY CONFLICT-FREE**:

```yaml
services:
  postgres:
    ports:
      - "5433:5432"  # ✅ No conflict - main postgres uses 5432

  hygiene-backend:
    ports:
      - "8081:8080"  # ✅ No conflict - unique port

  rule-control-api:
    ports:
      - "8101:8100"  # ✅ No conflict - unique port

  hygiene-dashboard:
    ports:
      - "3002:3000"  # ✅ No conflict - unique port

  nginx:
    ports:
      - "8082:80"    # ✅ No conflict - unique port
```

### Issues Identified

1. **Missing PostgreSQL Container**: The hygiene-postgres container is not running
   - The main postgres container (sutazai-postgres) is running instead
   - Hygiene services are probably trying to connect to non-existent hygiene-postgres

2. **Redis Service**: Missing from current docker-compose but referenced in backend config
   - Need to add redis service or configure to use existing sutazai-redis

### Optimized Port Allocation Strategy

#### Option 1: Keep Current Ports (Recommended)
All current ports are available and conflict-free. The issue is not port conflicts but missing services.

```yaml
# Current ports - NO CHANGES NEEDED
hygiene-postgres: 5433:5432    # ✅ Available
hygiene-backend: 8081:8080     # ✅ Available  
rule-control-api: 8101:8100    # ✅ Available
hygiene-dashboard: 3002:3000   # ✅ Available
hygiene-nginx: 8082:80         # ✅ Available
```

#### Option 2: Alternative Port Range (If needed)
If you prefer a dedicated port range for hygiene services:

```yaml
# Alternative port allocation in 5400+ and 8200+ ranges
hygiene-postgres: 5400:5432
hygiene-backend: 8200:8080
rule-control-api: 8201:8100
hygiene-dashboard: 3010:3000
hygiene-nginx: 8202:80
hygiene-redis: 5401:6379
```

## Service Dependencies Fix Required

### 1. PostgreSQL Service
The hygiene-monitoring services are configured to connect to `postgres:5432` but this conflicts with the main system postgres. Options:

**Option A**: Use separate hygiene postgres (current config)
- Ensure hygiene-postgres container starts
- Use port 5433 externally, 5432 internally within hygiene network

**Option B**: Share main postgres instance
- Create hygiene database in main postgres
- Update connection strings to use sutazai-postgres
- Remove postgres service from hygiene compose

### 2. Redis Service
Add Redis service to hygiene-monitor.yml or configure to use existing sutazai-redis.

## Network Isolation Strategy

The hygiene monitoring system uses its own network (`172.21.0.0/16`) which is good for isolation from the main system network (`172.20.0.0/16`).

```yaml
networks:
  hygiene-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.21.0.0/16  # ✅ Different from main system
```

## Health Check Status

Current service health:
- ✅ hygiene-backend: healthy
- ✅ rule-control-api: healthy  
- ✅ hygiene-dashboard: healthy
- ❌ hygiene-nginx: unhealthy
- ❌ hygiene-postgres: not running

## Action Items

1. **Fix PostgreSQL**: Ensure hygiene-postgres container starts
2. **Fix Redis**: Add redis service or configure to use existing
3. **Debug Nginx**: Check why hygiene-nginx is unhealthy
4. **Test Connectivity**: Verify all services can communicate

## Port Reservation Policy

Reserve the following port ranges for future hygiene monitoring expansions:
- **5400-5410**: Database services
- **8200-8210**: API/Backend services  
- **3010-3020**: Frontend services
- **6400-6410**: Cache/Queue services

This ensures predictable port allocation and prevents future conflicts.