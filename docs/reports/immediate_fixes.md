# Immediate Fixes for Critical System Failures
## Emergency Response Actions

**Generated:** 2025-08-16 14:57:00 UTC  
**Priority:** URGENT - Production System Issues  
**Estimated Fix Time:** 2-4 hours

---

## Critical Issue #1: Consul API Compatibility Fix ⚠️ **URGENT**

### Problem
```bash
ERROR - Failed to register service mcp-nx-mcp-localhost-11111: 
Consul.Agent.Service.register() got an unexpected keyword argument 'meta'
```

### Root Cause
The MCP service registration code is using the deprecated 'meta' parameter which is no longer supported in the current Consul API version.

### Immediate Fix Required
**File:** `/opt/sutazaiapp/backend/app/mesh/service_mesh.py`

**Action:** Remove or replace the 'meta' parameter in all `Service.register()` calls

**Current Code Pattern to Fix:**
```python
# BROKEN: Using deprecated 'meta' parameter
consul_client.agent.service.register(
    name=service_name,
    service_id=service_id,
    address=address,
    port=port,
    meta=metadata  # THIS PARAMETER CAUSES THE ERROR
)
```

**Fixed Code:**
```python
# FIXED: Use 'tags' instead of 'meta' for service metadata
consul_client.agent.service.register(
    name=service_name,
    service_id=service_id,
    address=address,
    port=port,
    tags=[f"{k}:{v}" for k, v in metadata.items()] if metadata else []
)
```

### Implementation Steps
1. Locate all `Service.register()` calls in `/opt/sutazaiapp/backend/app/mesh/`
2. Replace `meta=` parameter with `tags=` parameter using key:value format
3. Test service registration with `docker restart sutazai-backend`
4. Verify MCP services register successfully

---

## Critical Issue #2: Missing API Endpoints ⚠️ **HIGH PRIORITY**

### Problem
```bash
❌ Agent List (/agents): FAILED  
❌ Model List (/models): FAILED
❌ Simple Chat (/simple-chat): FAILED
```

### Root Cause
The API routes are defined but not properly mounted or the endpoints are not returning the expected responses.

### Analysis
The code shows routes are defined:
- `/api/v1/agents` exists in main.py line 645
- Routes are properly defined but may have routing issues

### Immediate Fix Required

**File:** `/opt/sutazaiapp/backend/app/main.py`

**Issue:** Routes exist but may not be mounted at root level. Test shows:
- `GET /health` works (returns healthy)
- `GET /agents` fails (404 Not Found)

**Fix:** Add route mounting for backward compatibility

**Add to main.py after line 645:**
```python
# Add root-level routes for backward compatibility
@app.get("/agents", response_model=List[AgentResponse])
async def get_agents_root():
    """Root-level agents endpoint for compatibility"""
    return await get_agents()

@app.get("/models")
async def get_models_root():
    """Root-level models endpoint for compatibility"""
    ollama_service = await get_ollama_service()
    models = await ollama_service.list_models()
    return {"models": models}

@app.post("/simple-chat")
async def simple_chat_root(request: ChatRequest):
    """Root-level chat endpoint for compatibility"""
    ollama_service = await get_ollama_service()
    response = await ollama_service.generate(
        model=request.model,
        prompt=request.message
    )
    return {"response": response.get("response", "")}
```

---

## Critical Issue #3: PostgreSQL Authentication Failure ⚠️ **HIGH PRIORITY**

### Problem
```bash
FATAL: password authentication failed for user "sutazai"
```

### Root Cause
Database credentials mismatch or password configuration drift.

### Immediate Investigation Required

**Check current credentials:**
```bash
docker exec sutazai-postgres psql -U postgres -c "\du"
```

**Verify environment variables:**
```bash
docker exec sutazai-backend env | grep -i postgres
```

### Immediate Fix Options

**Option 1: Reset Database Password**
```bash
docker exec sutazai-postgres psql -U postgres -c "ALTER USER sutazai PASSWORD 'sutazai123';"
```

**Option 2: Check Environment Variables**
```bash
# Verify these match in docker-compose.yml and backend config
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=sutazai123
POSTGRES_DB=sutazai
```

**Option 3: Restart Database Connection Pool**
```bash
docker restart sutazai-backend
```

---

## Critical Issue #4: MCP Service Startup Failures ⚠️ **MEDIUM PRIORITY**

### Problem
8 out of 17 MCP services failing to start:
- postgres, files, http, ddg, github, extended-memory, puppeteer-mcp (no longer in use), playwright-mcp

### Root Cause
Service registration failures cascade to prevent MCP startup.

### Dependencies
This issue will be resolved after fixing the Consul API compatibility issue above.

### Verification Steps
After fixing Consul registration:
```bash
docker logs sutazai-backend --tail=20 | grep -i mcp
```

Expected result: All 17 MCP services should start successfully.

---

## Validation Checklist

### After Implementing Fixes

**1. Consul Registration Fix**
- [ ] No more "unexpected keyword argument 'meta'" errors
- [ ] MCP services register with Consul successfully
- [ ] Service mesh shows registered services

**2. API Endpoints Fix**
- [ ] `curl http://localhost:10010/agents` returns JSON response
- [ ] `curl http://localhost:10010/models` returns models list
- [ ] `curl -X POST http://localhost:10010/simple-chat -d '{"message":"test"}'` works

**3. Database Authentication Fix**
- [ ] No more "password authentication failed" errors
- [ ] Backend connects to database successfully
- [ ] Database queries work from backend

**4. Overall System Health**
- [ ] `/opt/sutazaiapp/scripts/monitoring/live_logs.sh --test` shows all endpoints OK
- [ ] All 17 MCP services start successfully
- [ ] No critical errors in application logs

---

## Implementation Priority

**Execute in this order:**
1. **Consul API Fix** (30 minutes) - Unblocks MCP services
2. **API Endpoints Fix** (45 minutes) - Restores API functionality  
3. **Database Authentication** (15 minutes) - Resolves connection issues
4. **Validation Testing** (30 minutes) - Confirms all fixes work

**Total Estimated Time:** 2 hours

---

## Rollback Plan

If fixes cause additional issues:

**1. Backup Current State**
```bash
cp /opt/sutazaiapp/backend/app/main.py /opt/sutazaiapp/backend/app/main.py.backup.$(date +%Y%m%d_%H%M%S)
cp /opt/sutazaiapp/backend/app/mesh/service_mesh.py /opt/sutazaiapp/backend/app/mesh/service_mesh.py.backup.$(date +%Y%m%d_%H%M%S)
```

**2. Rollback Commands**
```bash
docker restart sutazai-backend
docker restart sutazai-postgres
```

**3. Verify Rollback**
```bash
curl http://localhost:10010/health
```

---

## Next Steps After Fixes

1. **Monitor system for 24 hours** to ensure stability
2. **Update monitoring alerts** to catch these issues earlier
3. **Add integration tests** to prevent regression
4. **Document lessons learned** for future reference

**Report Status:** Ready for immediate implementation
**Approval Required:** System Administrator
**Risk Level:** Low (fixes address root causes identified through debugging)