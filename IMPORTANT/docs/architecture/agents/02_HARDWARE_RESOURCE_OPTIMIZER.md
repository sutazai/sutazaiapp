# Hardware Resource Optimizer - Technical Specification

**Service Name:** Hardware Resource Optimizer  
**Container:** `sutazai-hardware-resource-optimizer`  
**Port:** 8002 (maps to internal 8080)  
**Version:** 4.0.0  
**Status:** Most Complete Implementation (60% functional)

## Overview

The Hardware Resource Optimizer is the most functionally complete agent in the system. It provides real system monitoring and optimization capabilities, including Docker container management, storage analysis, and system resource cleanup. Unlike other agents, this service actually performs meaningful operations.

## Technical Architecture

### Technology Stack
- **Framework:** FastAPI with custom BaseAgent class
- **System Integration:** psutil for metrics, Docker SDK
- **Storage Analysis:** hashlib for duplicate detection, gzip for compression
- **Process Management:** subprocess for system commands
- **Container Management:** Docker Python SDK

### File Structure
```
/app/
├── app.py                 # Main application with optimization logic
├── requirements.txt       # Dependencies
├── shared/
│   └── agent_base.py     # Base agent class
└── agent_data/           # Local data directory
```

### Unique Implementation Notes
- Only agent with actual system interaction capabilities
- Includes safety mechanisms to prevent system damage
- Has the most comprehensive endpoint implementation

## API Endpoints

### Health & Monitoring

#### GET /health
Returns health status with real system metrics.

**Response Example:**
```json
{
  "status": "healthy",
  "agent": "hardware-resource-optimizer",
  "description": "On-demand hardware resource optimization and cleanup tool",
  "docker_available": true,
  "system_status": {
    "cpu_percent": 2.5,
    "memory_percent": 20.0,
    "disk_percent": 22.13,
    "memory_available_gb": 23.51,
    "disk_free_gb": 732.76,
    "timestamp": 1754527428.629
  },
  "timestamp": 1754527428.629
}
```

#### GET /status
Detailed system resource status.

**Response Example:**
```json
{
  "agent": "hardware-resource-optimizer",
  "system_metrics": {
    "cpu": {
      "percent": 2.5,
      "cores": 8,
      "frequency_mhz": 2400
    },
    "memory": {
      "total_gb": 32.0,
      "available_gb": 23.51,
      "percent": 20.0,
      "swap_percent": 0.0
    },
    "disk": {
      "total_gb": 1000.0,
      "free_gb": 732.76,
      "percent": 22.13
    },
    "processes": {
      "total": 287,
      "running": 2,
      "sleeping": 285
    }
  },
  "docker_status": "available",
  "optimization_available": true
}
```

### Memory Optimization

#### POST /optimize/memory
Performs memory optimization operations.

**Request Body:** None required

**Response Example:**
```json
{
  "status": "completed",
  "optimizations": {
    "python_gc": "collected",
    "cleared_objects": 1547,
    "memory_freed_mb": 12.5,
    "system_cache": "attempted",
    "duration_seconds": 0.8
  },
  "memory_after": {
    "available_gb": 23.8,
    "percent": 19.5
  }
}
```

**What Actually Happens:**
- Runs Python garbage collection (`gc.collect()`)
- Attempts to clear system buffers (requires root)
- Syncs filesystem buffers

### CPU Optimization

#### POST /optimize/cpu
Optimizes CPU usage by adjusting process priorities.

**Request Body (Optional):**
```json
{
  "target_processes": ["python", "node"],
  "nice_level": 10
}
```

**Response Example:**
```json
{
  "status": "completed",
  "adjustments": {
    "processes_adjusted": 5,
    "average_nice_change": 5,
    "cpu_governors": "performance"
  },
  "cpu_after": {
    "percent": 2.1,
    "load_average": [0.5, 0.6, 0.7]
  }
}
```

**What Actually Happens:**
- Adjusts process nice values for CPU-intensive processes
- Attempts to set CPU governor (requires root)
- Identifies and reports high CPU consumers

### Disk Optimization

#### POST /optimize/disk
Cleans up disk space by removing temporary files.

**Request Body (Optional):**
```json
{
  "clean_docker": true,
  "clean_temp": true,
  "clean_logs": false
}
```

**Response Example:**
```json
{
  "status": "completed",
  "cleaned": {
    "temp_files": 847,
    "space_freed_mb": 234.5,
    "docker_images": 0,
    "docker_containers": 0
  },
  "disk_after": {
    "free_gb": 733.0,
    "percent": 21.9
  }
}
```

**What Actually Happens:**
- Removes files in `/tmp` older than 7 days
- Cleans user temp directories
- Optionally cleans Docker resources

### Docker Optimization

#### POST /optimize/docker
Manages Docker resources - containers, images, volumes.

**Request Body (Optional):**
```json
{
  "remove_stopped": true,
  "remove_dangling_images": true,
  "prune_volumes": false,
  "prune_networks": false
}
```

**Response Example:**
```json
{
  "status": "completed",
  "docker_cleanup": {
    "containers_removed": 3,
    "images_removed": 7,
    "space_reclaimed_mb": 1245.8,
    "volumes_pruned": 0,
    "networks_pruned": 0
  }
}
```

**What Actually Happens:**
- Removes stopped containers
- Removes dangling images
- Optionally prunes unused volumes and networks
- Reports space reclaimed

### Storage Analysis

#### GET /analyze/storage
Comprehensive storage analysis of the system.

**Query Parameters:**
- `path`: Directory to analyze (default: /)
- `min_size_mb`: Minimum file size to report (default: 100)

**Response Example:**
```json
{
  "analysis": {
    "total_files": 145623,
    "total_size_gb": 267.4,
    "largest_directories": [
      {
        "path": "/var/lib/docker",
        "size_gb": 45.2
      },
      {
        "path": "/home/user/Downloads",
        "size_gb": 23.1
      }
    ],
    "file_type_distribution": {
      ".log": {
        "count": 3421,
        "size_gb": 8.7
      },
      ".json": {
        "count": 9823,
        "size_gb": 2.1
      }
    },
    "optimization_opportunities": {
      "duplicate_files": 823,
      "potential_space_gb": 4.5,
      "old_logs": 234,
      "large_temp_files": 12
    }
  }
}
```

#### GET /analyze/storage/duplicates
Finds duplicate files using MD5 hashing.

**Query Parameters:**
- `path`: Directory to scan (default: /tmp)
- `min_size_bytes`: Minimum file size (default: 1024)

**Response Example:**
```json
{
  "duplicates": [
    {
      "hash": "5d41402abc4b2a76b9719d911017c592",
      "size_bytes": 1048576,
      "files": [
        "/tmp/file1.dat",
        "/var/tmp/copy_of_file1.dat"
      ]
    }
  ],
  "summary": {
    "total_duplicates": 23,
    "space_wasted_mb": 156.7,
    "scan_duration_seconds": 4.2
  }
}
```

#### GET /analyze/storage/large-files
Identifies large files consuming disk space.

**Query Parameters:**
- `path`: Directory to scan (default: /)
- `limit`: Number of files to return (default: 50)
- `min_size_mb`: Minimum size in MB (default: 100)

**Response Example:**
```json
{
  "large_files": [
    {
      "path": "/var/log/application.log",
      "size_mb": 2341.5,
      "modified": "2025-08-06T15:23:41",
      "accessed": "2025-08-07T00:01:15"
    }
  ],
  "summary": {
    "total_large_files": 47,
    "total_size_gb": 89.3
  }
}
```

### Storage Optimization

#### POST /optimize/storage
Comprehensive storage optimization combining multiple strategies.

**Request Body:**
```json
{
  "remove_duplicates": true,
  "compress_old_files": true,
  "clean_cache": true,
  "archive_logs": true
}
```

**Response Example:**
```json
{
  "status": "completed",
  "optimizations": {
    "duplicates_removed": 45,
    "files_compressed": 234,
    "cache_cleared_mb": 567.8,
    "logs_archived": 89,
    "total_space_freed_gb": 4.7
  },
  "duration_seconds": 23.4
}
```

#### POST /optimize/storage/compress
Compresses specified files or directories.

**Request Body:**
```json
{
  "paths": ["/var/log/old", "/tmp/data"],
  "compression_level": 6,
  "keep_originals": false
}
```

**Response Example:**
```json
{
  "status": "completed",
  "compression_results": {
    "files_compressed": 156,
    "original_size_mb": 1234.5,
    "compressed_size_mb": 234.1,
    "compression_ratio": 0.81,
    "space_saved_mb": 1000.4
  }
}
```

## Safety Mechanisms

### Protected Paths
The optimizer includes safety mechanisms to prevent system damage:

```python
protected_paths = {
    '/etc',      # System configuration
    '/boot',     # Boot files
    '/usr',      # System binaries
    '/bin',      # Essential commands
    '/sbin',     # System binaries
    '/lib',      # System libraries
    '/proc',     # Process information
    '/sys',      # System information
    '/dev'       # Device files
}
```

### User Protection Patterns
```python
user_protected_patterns = {
    '/home/*/Documents',
    '/home/*/Desktop',
    '/home/*/Pictures'
}
```

### Operation Safeguards
1. **Dry Run Mode:** Can simulate operations without executing
2. **Size Limits:** Won't process files beyond certain sizes
3. **Age Checks:** Only removes files older than threshold
4. **Confirmation:** Critical operations require confirmation
5. **Backup:** Creates safety copies in `/tmp/hardware_optimizer_safety`

## Implementation Details

### System Metrics Collection
```python
def _get_system_status(self) -> Dict[str, Any]:
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "disk_percent": (disk.used / disk.total) * 100,
        "memory_available_gb": memory.available / (1024**3),
        "disk_free_gb": disk.free / (1024**3)
    }
```

### Docker Management
```python
def _init_docker_client(self):
    try:
        client = docker.from_env()
        client.ping()  # Test connectivity
        return client
    except Exception as e:
        self.logger.warning(f"Docker unavailable: {e}")
        return None
```

### Duplicate Detection Algorithm
```python
def find_duplicates(self, directory: str) -> Dict[str, List[str]]:
    hash_map = defaultdict(list)
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_hash = self._calculate_md5(filepath)
            hash_map[file_hash].append(filepath)
    
    # Return only duplicates (more than one file with same hash)
    return {k: v for k, v in hash_map.items() if len(v) > 1}
```

## Performance Characteristics

### Resource Usage
- **Memory:** 100-200MB during operation
- **CPU:** Spikes during analysis (10-30%)
- **Disk I/O:** Heavy during storage analysis
- **Network:**   (Docker API calls only)

### Operation Timing
| Operation | Typical Duration | Resource Impact |
|-----------|-----------------|-----------------|
| Memory optimization | < 1 second | Low |
| CPU optimization | < 2 seconds | Low |
| Disk cleanup | 5-60 seconds | Medium |
| Docker cleanup | 10-120 seconds | High |
| Storage analysis | 30-300 seconds | High |
| Duplicate detection | 60-600 seconds | Very High |

### Scalability Considerations
- **File System Size:** Performance degrades with millions of files
- **Hash Calculation:** MD5 is fast but still I/O bound
- **Docker Operations:** Limited by Docker service performance
- **Memory Analysis:** Limited by available system RAM

## Configuration

### Environment Variables
```bash
PORT=8080                          # Internal service port
DOCKER_HOST=unix:///var/run/docker.sock  # Docker socket
OPTIMIZATION_SAFE_MODE=true       # Enable safety checks
MAX_FILE_SIZE_GB=10               # Max file size to process
TEMP_CLEANUP_DAYS=7               # Age of temp files to remove
```

### Safety Configuration
```python
# Configuration constants
SAFE_TEMP_LOCATION = '/tmp/hardware_optimizer_safety'
MAX_COMPRESSION_SIZE_GB = 10
MIN_FREE_SPACE_GB = 10
DUPLICATE_BATCH_SIZE = 1000
```

## Error Handling

### Graceful Degradation
- Continues without Docker if unavailable
- Skips protected paths silently
- Logs but doesn't fail on permission errors
- Returns partial results on timeout

### Error Response Format
```json
{
  "status": "partial",
  "message": "Operation completed with errors",
  "errors": [
    {
      "operation": "docker_cleanup",
      "error": "Docker service not responding"
    }
  ],
  "completed_operations": ["memory", "disk"]
}
```

## Testing

### Integration Test Suite
Located in `/app/tests/` with comprehensive coverage:

1. **Endpoint Tests** (`test_endpoints.py`)
2. **Performance Tests** (`performance_stress_tests.py`)
3. **Safety Tests** (`test_dry_run_safety.py`)
4. **Storage Tests** (`test_storage_analysis.py`)
5. **Docker Tests** (`test_docker_operations.py`)

### Manual Testing Commands
```bash
# Test health endpoint
curl http://localhost:8002/health

# Analyze storage
curl "http://localhost:8002/analyze/storage?path=/tmp"

# Find duplicates (be patient)
curl "http://localhost:8002/analyze/storage/duplicates?path=/tmp"

# Safe optimization
curl -X POST http://localhost:8002/optimize/all

# Docker cleanup (requires Docker)
curl -X POST http://localhost:8002/optimize/docker
```

## Known Issues & Limitations

### Current Limitations
1. **Root Required:** Many system optimizations need root privileges
2. **No GPU Support:** GPU optimization not implemented
3. **Large File Systems:** Duplicate detection slow on large volumes
4. **No Scheduling:** Only on-demand, no automatic optimization
5. **No Clustering:** Single node only, no distributed operation

### Known Bugs
1. **Memory Metrics:** May report incorrect values in containers
2. **Docker Cleanup:** Sometimes fails to remove volumes in use
3. **Compression:** May timeout on very large files
4. **Path Traversal:** Some symbolic links not handled correctly

## Security Considerations

### Current Security
- Path validation to prevent directory traversal
- Protected system paths cannot be modified
- Size limits prevent resource exhaustion
- No remote code execution

### Security Gaps
- **No Authentication:** All endpoints are public
- **No Rate Limiting:** Could be used for DoS
- **No Audit Logging:** Operations not logged for compliance
- **Privilege Escalation:** Some operations attempt sudo

## Development Roadmap

### Completed Features
- ✅ System metrics collection
- ✅ Docker container management
- ✅ Storage analysis
- ✅ Duplicate file detection
- ✅ Basic optimization operations
- ✅ Safety mechanisms

### In Progress
- ⚠️ Compression optimization
- ⚠️ Advanced memory management
- ⚠️ Process optimization

### Planned Features
- ⬜ GPU resource management
- ⬜ Network optimization
- ⬜ Database optimization
- ⬜ Scheduled optimization
- ⬜ Machine learning for predictive optimization
- ⬜ Cluster-wide optimization
- ⬜ Web UI dashboard

## Best Practices for Usage

### Production Deployment
1. **Run with Limited Privileges:** Don't run as root unless necessary
2. **Configure Limits:** Set appropriate resource limits
3. **Monitor Operations:** Watch for long-running operations
4. **Test First:** Use dry-run mode before actual optimization
5. **Backup Critical Data:** Before running storage optimization

### Optimization Strategy
1. **Start Small:** Test on `/tmp` before system-wide
2. **Monitor Impact:** Watch system metrics during operation
3. **Schedule Wisely:** Run during low-usage periods
4. **Incremental Approach:** Don't run all optimizations at once
5. **Verify Results:** Check system stability after optimization

## Conclusion

The Hardware Resource Optimizer is the **most functional agent** in the SutazAI system, providing real value through system monitoring and optimization capabilities. While not fully complete, it demonstrates proper implementation patterns and actually performs useful operations.

**Strengths:**
- Real system integration
- Working Docker management
- Comprehensive storage analysis
- Safety mechanisms
- Good error handling

**Weaknesses:**
- Limited by system privileges
- No scheduling capability
- Missing GPU support
- Performance issues on large systems

**Production Readiness:** Suitable for development/staging environments with supervision. Not recommended for production without authentication and rate limiting.