# Ollama Connectivity Fix Report

## Date: 2025-08-29

## Issues Identified and Fixed

### 1. Health Check Failure
**Problem**: Ollama container showed as "unhealthy" because the health check used `curl` which isn't available in the ollama image.

**Solution**: Updated health check to use `ollama list` command which is native to the container.

```yaml
healthcheck:
  test: ["CMD-SHELL", "ollama list >/dev/null 2>&1 || exit 1"]
```

### 2. Port Configuration Mismatch
**Problem**: Services were trying to connect to wrong ports:
- External port: 11435
- Internal port: 11434
- Some configs used 11434 externally (incorrect)

**Solution**: Standardized configuration:
- External access: `localhost:11435`
- Internal Docker network access: `sutazai-ollama:11434`

### 3. Backend Configuration Issue
**Problem**: Backend was using `host.docker.internal` instead of container name.

**Solution**: Updated `/opt/sutazaiapp/docker-compose-backend.yml`:
```yaml
OLLAMA_HOST: sutazai-ollama  # Changed from host.docker.internal
OLLAMA_PORT: 11434
```

## Current Configuration

### Network Configuration
- Container Name: `sutazai-ollama`
- Network: `sutazaiapp_sutazai-network`
- IP Address: Dynamic (use container name for reliability)
- Port Mapping: `11435:11434` (external:internal)

### Service Connectivity

| Service | Connection String | Status |
|---------|------------------|--------|
| From Host | `http://localhost:11435` | ✓ Working |
| From Backend | `http://sutazai-ollama:11434` | ✓ Working |
| From MCP Bridge | `http://sutazai-ollama:11434` | ✓ Working |
| From Agents | `http://sutazai-ollama:11434` | ✓ Working |

### Available Models
- `tinyllama:latest` (637 MB) - Lightweight model for testing

## Files Modified

1. `/opt/sutazaiapp/docker-compose-backend.yml`
   - Changed OLLAMA_HOST from `host.docker.internal` to `sutazai-ollama`

2. `/opt/sutazaiapp/docker-compose-ollama-fix.yml` (Created)
   - Fixed health check configuration
   - Proper network aliases
   - Resource limits

## Testing Results

All connectivity tests passed:
- ✓ Host can reach Ollama on port 11435
- ✓ Backend can reach Ollama via container name
- ✓ MCP Bridge can reach Ollama
- ✓ Health check status: healthy
- ✓ Agents can connect to Ollama

## Recommendations

1. **Consolidate Docker Compose Files**: Multiple compose files define Ollama service:
   - `docker-compose-network-fix.yml`
   - `docker-compose-local-llm.yml`
   - `docker-compose-ollama-fix.yml` (new)
   
   Consider consolidating to avoid conflicts.

2. **Add More Models**: Currently only `tinyllama` is available. Consider adding:
   - `llama2:7b` for better performance
   - `codellama:7b` for code-specific tasks
   
3. **Monitor Resource Usage**: Ollama is limited to 2GB RAM and 2 CPUs. Monitor if this is sufficient for your workload.

4. **Update Agent Configurations**: Ensure all agent compose files use consistent Ollama configuration:
   ```yaml
   environment:
     - OLLAMA_BASE_URL=http://sutazai-ollama:11434
   ```

## Scripts Created

1. `/opt/sutazaiapp/fix-ollama-connectivity.sh`
   - Automated fix application script
   - Can be re-run if issues recur

2. `/opt/sutazaiapp/test-ollama-connectivity.sh`
   - Connectivity testing script
   - Use for troubleshooting

## How to Use

### Check Ollama Status
```bash
docker ps | grep ollama
```

### Test Connectivity
```bash
bash /opt/sutazaiapp/test-ollama-connectivity.sh
```

### Restart Ollama if Needed
```bash
cd /opt/sutazaiapp
docker compose -f docker-compose-ollama-fix.yml restart ollama
```

### Pull Additional Models
```bash
docker exec sutazai-ollama ollama pull llama2:7b
```

## Conclusion

All Ollama connectivity issues have been resolved. The service is now:
- Running with healthy status
- Accessible from all services via Docker network
- Configured with consistent port mapping
- Has a lightweight model (tinyllama) loaded for testing

The fix ensures reliable communication between Ollama and all dependent services in the SutazAI platform.