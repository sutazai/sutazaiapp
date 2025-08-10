# Hardware Optimization API Usage Guide

This document describes how to use the Hardware Optimization API endpoints integrated with the SutazAI backend system.

## Overview

The Hardware API provides comprehensive access to system monitoring, optimization, and resource management capabilities through the hardware-resource-optimizer service. All endpoints are authenticated and require appropriate permissions.

## Base URL

All hardware endpoints are prefixed with `/api/v1/hardware`

## Authentication

All endpoints require authentication using JWT Bearer tokens:

```bash
Authorization: Bearer <your_jwt_token>
```

## Permission System

The hardware API uses a role-based permission system with the following permissions:

- `hardware:monitor` - View system metrics and status
- `hardware:optimize` - Run system optimizations
- `hardware:process_control` - Control system processes
- `hardware:configure` - Configure monitoring settings
- `hardware:benchmark` - Run system benchmarks

Admin users have all permissions automatically.

## Available Endpoints

### 1. Health Check

Get the status of the hardware optimization service.

```bash
GET /api/v1/hardware/health
```

**Response:**
```json
{
  "status": "healthy",
  "agent": "hardware-resource-optimizer",
  "timestamp": "2025-08-09T12:00:00Z",
  "version": "1.0.0",
  "uptime_seconds": 3600.5,
  "last_optimization": "2025-08-09T11:30:00Z"
}
```

### 2. System Metrics

Get comprehensive system performance metrics.

```bash
GET /api/v1/hardware/metrics?include_processes=true&sample_duration=10
```

**Parameters:**
- `include_processes` (bool): Include process-level metrics
- `include_network` (bool): Include network metrics  
- `include_disk` (bool): Include disk I/O metrics
- `include_gpu` (bool): Include GPU metrics if available
- `sample_duration` (int): Sampling duration in seconds (1-60)

**Response:**
```json
{
  "timestamp": "2025-08-09T12:00:00Z",
  "cpu": {
    "usage_percent": 45.2,
    "cores": 8,
    "frequency_mhz": 2400
  },
  "memory": {
    "usage_percent": 67.8,
    "used_mb": 2048,
    "available_mb": 6144
  },
  "disk": {
    "usage_percent": 34.5,
    "read_mb_s": 12.4,
    "write_mb_s": 8.7
  },
  "network": {
    "bytes_sent": 1024000,
    "bytes_recv": 2048000
  }
}
```

### 3. Real-time Metrics Stream

Stream real-time system metrics using Server-Sent Events.

```bash
GET /api/v1/hardware/metrics/stream?interval=5&include_processes=false
```

**Parameters:**
- `interval` (int): Update interval in seconds (1-60)
- `include_processes` (bool): Include process metrics in stream

**Response:** Server-Sent Events stream with JSON data

### 4. System Optimization

Start a system optimization process.

```bash
POST /api/v1/hardware/optimize
```

**Request Body:**
```json
{
  "optimization_type": "cpu",
  "parameters": {
    "target_usage": 70,
    "aggressive_mode": false
  },
  "priority": "normal",
  "dry_run": false
}
```

**Supported optimization_types:**
- `cpu` - CPU optimization
- `memory` - Memory optimization
- `disk` - Disk I/O optimization
- `network` - Network optimization
- `power` - Power management
- `thermal` - Thermal optimization
- `processes` - Process optimization
- `services` - Service optimization
- `startup` - Startup optimization
- `full_system` - Complete system optimization

**Response:**
```json
{
  "task_id": "opt_12345678",
  "status": "running",
  "optimization_type": "cpu",
  "started_at": "2025-08-09T12:00:00Z",
  "changes_applied": [],
  "dry_run": false
}
```

### 5. Optimization Status

Check the status of an optimization task.

```bash
GET /api/v1/hardware/optimize/{task_id}
```

**Response:**
```json
{
  "task_id": "opt_12345678",
  "status": "completed",
  "optimization_type": "cpu",
  "started_at": "2025-08-09T12:00:00Z",
  "completed_at": "2025-08-09T12:05:30Z",
  "duration_seconds": 330.5,
  "changes_applied": [
    "Adjusted CPU governor to 'performance'",
    "Optimized process priorities",
    "Cleaned up background processes"
  ],
  "performance_impact": {
    "cpu_improvement": 15.2,
    "memory_freed_mb": 256
  }
}
```

### 6. Process Management

View and control system processes.

```bash
GET /api/v1/hardware/processes?sort_by=cpu&limit=50
```

**Parameters:**
- `sort_by` (string): Sort by 'cpu', 'memory', 'name', or 'pid'
- `limit` (int): Maximum processes to return (1-500)
- `filter_pattern` (string): Filter by process name pattern

**Response:**
```json
[
  {
    "pid": 1234,
    "name": "python3",
    "status": "running",
    "cpu_percent": 12.5,
    "memory_mb": 128.4,
    "threads": 4,
    "created_time": "2025-08-09T10:00:00Z"
  }
]
```

### 7. Process Control

Control system processes (requires `hardware:process_control` permission).

```bash
POST /api/v1/hardware/processes/control
```

**Request Body:**
```json
{
  "action": "prioritize",
  "process_id": 1234,
  "priority": 10,
  "resource_limits": {
    "cpu_percent": 50.0,
    "memory_mb": 1024
  }
}
```

**Supported actions:**
- `kill` - Terminate process
- `suspend` - Suspend process
- `resume` - Resume suspended process
- `prioritize` - Change process priority
- `limit` - Apply resource limits

### 8. Hardware Alerts

Get hardware-related alerts and notifications.

```bash
GET /api/v1/hardware/alerts?severity=high&limit=100&since_hours=24
```

**Parameters:**
- `severity` (string): Filter by 'low', 'medium', 'high', or 'critical'
- `limit` (int): Maximum alerts to return (1-1000)
- `since_hours` (int): Get alerts from last N hours (1-168)

### 9. Optimization Recommendations

Get AI-powered optimization recommendations.

```bash
GET /api/v1/hardware/recommendations?category=performance&priority=high
```

**Parameters:**
- `category` (string): Filter by recommendation category
- `priority` (string): Filter by priority level

**Response:**
```json
{
  "recommendations": [
    {
      "category": "cpu",
      "priority": "high",
      "title": "High CPU Usage Detected",
      "description": "Several processes are consuming excessive CPU resources",
      "suggested_actions": [
        "Consider terminating idle processes",
        "Adjust CPU governor settings",
        "Schedule resource-intensive tasks during off-peak hours"
      ],
      "estimated_impact": "15-25% CPU usage reduction",
      "confidence": 0.85
    }
  ]
}
```

### 10. System Benchmark

Run system performance benchmarks (requires `hardware:benchmark` permission).

```bash
POST /api/v1/hardware/benchmark?benchmark_type=cpu&duration_seconds=60
```

**Parameters:**
- `benchmark_type` (string): Type of benchmark ('cpu', 'memory', 'disk', 'network')
- `duration_seconds` (int): Benchmark duration (10-600 seconds)

## Error Handling

The API returns standard HTTP status codes:

- `200` - Success
- `400` - Bad Request (invalid parameters)
- `401` - Unauthorized (invalid/missing token)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found (resource doesn't exist)
- `503` - Service Unavailable (hardware service is down)
- `504` - Gateway Timeout (hardware service timeout)

**Error Response Format:**
```json
{
  "detail": "Error description",
  "error_code": "HARDWARE_SERVICE_UNAVAILABLE",
  "timestamp": "2025-08-09T12:00:00Z"
}
```

## Usage Examples

### Monitor System Performance

```python
import httpx
import asyncio

async def monitor_system():
    headers = {"Authorization": "Bearer YOUR_JWT_TOKEN"}
    
    async with httpx.AsyncClient() as client:
        # Get current metrics
        response = await client.get(
            "http://localhost:10010/api/v1/hardware/metrics",
            headers=headers,
            params={"sample_duration": 5}
        )
        metrics = response.json()
        print(f"CPU Usage: {metrics['cpu']['usage_percent']}%")
        print(f"Memory Usage: {metrics['memory']['usage_percent']}%")

# Run the monitor
asyncio.run(monitor_system())
```

### Run System Optimization

```python
async def optimize_system():
    headers = {"Authorization": "Bearer YOUR_JWT_TOKEN"}
    
    async with httpx.AsyncClient() as client:
        # Start optimization
        optimization_request = {
            "optimization_type": "full_system",
            "priority": "normal",
            "dry_run": False
        }
        
        response = await client.post(
            "http://localhost:10010/api/v1/hardware/optimize",
            headers=headers,
            json=optimization_request
        )
        
        task = response.json()
        task_id = task["task_id"]
        print(f"Optimization started: {task_id}")
        
        # Monitor progress
        while True:
            status_response = await client.get(
                f"http://localhost:10010/api/v1/hardware/optimize/{task_id}",
                headers=headers
            )
            status = status_response.json()
            
            if status["status"] in ["completed", "failed", "cancelled"]:
                print(f"Optimization {status['status']}")
                if status.get("changes_applied"):
                    print("Changes applied:")
                    for change in status["changes_applied"]:
                        print(f"  - {change}")
                break
                
            await asyncio.sleep(5)

# Run optimization
asyncio.run(optimize_system())
```

## Integration Notes

1. **Service Dependencies**: The hardware API requires the `sutazai-hardware-resource-optimizer` service running on port 8080.

2. **Caching**: Many endpoints use Redis caching for performance. Cache TTL varies by endpoint (5 seconds to 30 minutes).

3. **Rate Limiting**: Consider implementing rate limiting for optimization and benchmark endpoints to prevent system overload.

4. **Monitoring**: Use the streaming metrics endpoint for real-time dashboards and monitoring applications.

5. **Background Tasks**: Optimization operations run as background tasks and can be monitored via the task status endpoints.

## Security Considerations

1. **Authentication Required**: All endpoints require valid JWT authentication.

2. **Permission-Based Access**: Different operations require specific permissions.

3. **Audit Logging**: All hardware operations are logged for security auditing.

4. **Resource Limits**: Process control operations have built-in safety limits.

5. **Timeout Protection**: All hardware service calls have appropriate timeouts to prevent hanging requests.