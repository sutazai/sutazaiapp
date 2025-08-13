# Hardware Resource Optimizer Agent

## Purpose
On-demand hardware resource optimization tool that acts like a "janitor" - comes in when called, cleans up resources, and provides results. No continuous monitoring or interference with existing services.

## Design Philosophy
- **On-demand only**: Runs optimizations when API endpoints are called
- **Clean and exit**: Performs specific optimization tasks and returns results
- **No continuous monitoring**: Does not run background processes or loops
- **Single responsibility**: Focuses only on hardware resource optimization

## API Endpoints

### Health & Status
- `GET /health` - Agent health check and system status
- `GET /status` - Current system resource status

### Optimization Tasks
- `POST /optimize/memory` - Optimize memory usage (garbage collection, cache clearing)
- `POST /optimize/cpu` - Optimize CPU scheduling (adjust process priorities)
- `POST /optimize/disk` - Clean up disk space (remove temp files, old logs)
- `POST /optimize/docker` - Clean up Docker resources (unused containers, images)
- `POST /optimize/all` - Run all optimization tasks

## Hardware Optimization Tasks

### Memory Optimization
- Python garbage collection
- System cache clearing (page cache, dentries, inodes) if memory usage > 85%
- Reports memory freed

### CPU Optimization
- Identifies high CPU processes (>25% usage)
- Adjusts process nice values to lower priority
- Skips system critical processes
- Reports processes adjusted

### Disk Optimization
- Removes temp files older than 7 days from /tmp and /var/tmp
- Removes old log files (>30 days) if disk usage > 90%
- Reports space freed

### Docker Optimization
- Removes stopped containers
- Removes dangling images
- Prunes unused networks
- Prunes build cache
- Reports containers and images removed

## Usage Examples

### Check system status
```bash
curl http://localhost:8080/status
```

### Run memory optimization
```bash
curl -X POST http://localhost:8080/optimize/memory
```

### Run all optimizations
```bash
curl -X POST http://localhost:8080/optimize/all
```

## Response Format
All optimization endpoints return:
```json
{
  "status": "success",
  "optimization_type": "memory|cpu|disk|docker|all",
  "actions_taken": ["List of actions performed"],
  "timestamp": 1234567890.123,
  "additional_metrics": "..."
}
```

## Docker Configuration

### Build
```bash
docker build -t hardware-resource-optimizer .
```

### Run
```bash
docker run -d \
  --name hardware-optimizer \
  --privileged \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -p 8080:8080 \
  hardware-resource-optimizer
```

### Required Permissions
- `--privileged` - Required for system-level optimizations
- Docker socket access - Required for Docker resource cleanup

## Environment Variables
- `PORT` - API server port (default: 8080)
- `LOG_LEVEL` - Logging level (default: INFO)
- `AGENT_TYPE` - Agent type identifier

## Architecture Notes

### Simple and Clean
- No complex state management
- No continuous monitoring loops
- No background tasks or threads
- FastAPI for simple REST endpoints

### Security Considerations
- Requires privileged access for system optimizations
- Only performs safe cleanup operations
- No destructive operations on running services
- Logs all actions taken

### Resource Requirements
-   CPU usage when idle
- Low memory footprint
- Only active during optimization requests

## Integration
This agent is designed to be called by:
- System monitoring tools
- Orchestration scripts
- Manual operations
- Scheduled maintenance tasks

It does NOT run continuously or interfere with normal system operations.