# MCP Automation System - API Reference Documentation

**Version**: 3.0.0  
**API Version**: v1  
**Last Updated**: 2025-08-15 17:00:00 UTC  
**Base URL**: `http://localhost:8080/api/v1`

## Table of Contents

1. [Authentication](#authentication)
2. [API Conventions](#api-conventions)
3. [Core Endpoints](#core-endpoints)
4. [Server Management](#server-management)
5. [Update Operations](#update-operations)
6. [Testing Operations](#testing-operations)
7. [Cleanup Operations](#cleanup-operations)
8. [Monitoring & Metrics](#monitoring--metrics)
9. [Orchestration Control](#orchestration-control)
10. [WebSocket API](#websocket-api)
11. [Error Handling](#error-handling)
12. [Rate Limiting](#rate-limiting)
13. [API Examples](#api-examples)

## Authentication

All API requests require authentication using Bearer tokens.

### Obtaining a Token

```http
POST /api/v1/auth/token
Content-Type: application/json

{
  "username": "admin",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "refresh_token_string"
}
```

### Using the Token

Include the token in the Authorization header:
```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Refreshing Token

```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "refresh_token_string"
}
```

## API Conventions

### Request Format

- **Content-Type**: `application/json` for all POST/PUT requests
- **Accept**: `application/json` for all requests
- **Charset**: UTF-8

### Response Format

All responses follow this structure:

**Success Response:**
```json
{
  "success": true,
  "data": {
    // Response data
  },
  "metadata": {
    "timestamp": "2025-08-15T17:00:00Z",
    "request_id": "uuid-string",
    "version": "v1"
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      // Additional error context
    }
  },
  "metadata": {
    "timestamp": "2025-08-15T17:00:00Z",
    "request_id": "uuid-string"
  }
}
```

### HTTP Status Codes

| Code | Meaning | Usage |
|------|---------|-------|
| 200 | OK | Successful GET, PUT |
| 201 | Created | Successful POST creating resource |
| 202 | Accepted | Request accepted for async processing |
| 204 | No Content | Successful DELETE |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Resource conflict |
| 422 | Unprocessable Entity | Validation error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily unavailable |

## Core Endpoints

### System Status

#### GET /api/v1/status

Get overall system health and status.

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "3.0.0",
    "uptime": 86400,
    "components": {
      "database": "healthy",
      "cache": "healthy",
      "monitoring": "healthy",
      "mcp_servers": "healthy"
    },
    "servers": {
      "total": 17,
      "healthy": 17,
      "unhealthy": 0,
      "updating": 0
    },
    "last_check": "2025-08-15T17:00:00Z"
  }
}
```

### System Information

#### GET /api/v1/info

Get detailed system information.

**Response:**
```json
{
  "success": true,
  "data": {
    "version": "3.0.0",
    "api_version": "v1",
    "environment": "production",
    "features": {
      "auto_update": true,
      "cleanup": true,
      "monitoring": true,
      "orchestration": true
    },
    "limits": {
      "max_concurrent_updates": 5,
      "max_api_requests_per_minute": 100,
      "max_websocket_connections": 1000
    }
  }
}
```

### Health Check

#### GET /api/v1/health

Lightweight health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-15T17:00:00Z"
}
```

## Server Management

### List MCP Servers

#### GET /api/v1/servers

Get list of all MCP servers with their current status.

**Query Parameters:**
- `status` (optional): Filter by status (healthy, unhealthy, updating)
- `sort` (optional): Sort field (name, status, version)
- `order` (optional): Sort order (asc, desc)

**Response:**
```json
{
  "success": true,
  "data": {
    "servers": [
      {
        "name": "github",
        "status": "healthy",
        "version": "1.2.3",
        "description": "GitHub API integration",
        "last_check": "2025-08-15T17:00:00Z",
        "metrics": {
          "uptime": 86400,
          "request_count": 1000,
          "error_rate": 0.001
        }
      }
    ],
    "total": 17,
    "healthy": 17,
    "unhealthy": 0
  }
}
```

### Get Server Details

#### GET /api/v1/servers/{server_name}

Get detailed information about a specific MCP server.

**Response:**
```json
{
  "success": true,
  "data": {
    "name": "github",
    "status": "healthy",
    "version": "1.2.3",
    "description": "GitHub API integration",
    "configuration": {
      "repository": "github.com/example/mcp-github",
      "auto_update": true,
      "health_check_interval": 60
    },
    "metrics": {
      "uptime": 86400,
      "request_count": 1000,
      "response_time_avg": 150,
      "error_rate": 0.001,
      "last_error": null
    },
    "health_checks": [
      {
        "timestamp": "2025-08-15T17:00:00Z",
        "status": "passed",
        "duration_ms": 50
      }
    ]
  }
}
```

### Start Server

#### POST /api/v1/servers/{server_name}/start

Start a stopped MCP server.

**Response:**
```json
{
  "success": true,
  "data": {
    "server": "github",
    "status": "starting",
    "message": "Server start initiated"
  }
}
```

### Stop Server

#### POST /api/v1/servers/{server_name}/stop

Stop a running MCP server.

**Request Body (optional):**
```json
{
  "force": false,
  "timeout": 30
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "server": "github",
    "status": "stopping",
    "message": "Server stop initiated"
  }
}
```

### Restart Server

#### POST /api/v1/servers/{server_name}/restart

Restart an MCP server.

**Request Body (optional):**
```json
{
  "graceful": true,
  "timeout": 30
}
```

## Update Operations

### Check for Updates

#### GET /api/v1/updates/check

Check all MCP servers for available updates.

**Query Parameters:**
- `server` (optional): Check specific server only

**Response:**
```json
{
  "success": true,
  "data": {
    "updates_available": true,
    "servers": [
      {
        "name": "github",
        "current_version": "1.2.3",
        "latest_version": "1.2.4",
        "update_available": true,
        "changelog": "Bug fixes and performance improvements",
        "breaking_changes": false
      }
    ],
    "total_updates": 3
  }
}
```

### Update Server

#### POST /api/v1/updates/{server_name}

Initiate update for a specific MCP server.

**Request Body:**
```json
{
  "version": "latest",
  "rollback_on_failure": true,
  "dry_run": false,
  "backup": true,
  "force": false
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "update_id": "update-uuid-12345",
    "server": "github",
    "from_version": "1.2.3",
    "to_version": "1.2.4",
    "status": "in_progress",
    "started_at": "2025-08-15T17:00:00Z"
  }
}
```

### Get Update Status

#### GET /api/v1/updates/{update_id}

Get status of an ongoing or completed update.

**Response:**
```json
{
  "success": true,
  "data": {
    "update_id": "update-uuid-12345",
    "server": "github",
    "status": "completed",
    "from_version": "1.2.3",
    "to_version": "1.2.4",
    "started_at": "2025-08-15T17:00:00Z",
    "completed_at": "2025-08-15T17:05:00Z",
    "duration_seconds": 300,
    "steps": [
      {
        "name": "backup",
        "status": "completed",
        "duration": 30
      },
      {
        "name": "download",
        "status": "completed",
        "duration": 60
      },
      {
        "name": "install",
        "status": "completed",
        "duration": 120
      },
      {
        "name": "verify",
        "status": "completed",
        "duration": 90
      }
    ]
  }
}
```

### Rollback Update

#### POST /api/v1/updates/{server_name}/rollback

Rollback a server to previous version.

**Request Body:**
```json
{
  "target_version": "auto",
  "force": false
}
```

## Testing Operations

### Run Tests

#### POST /api/v1/tests/run

Execute test suite for MCP servers.

**Request Body:**
```json
{
  "test_type": "all",
  "servers": ["all"],
  "verbose": true,
  "parallel": true,
  "timeout": 300
}
```

**Test Types:**
- `unit`: Unit tests only
- `integration`: Integration tests
- `performance`: Performance tests
- `security`: Security tests
- `all`: All test types

**Response:**
```json
{
  "success": true,
  "data": {
    "test_id": "test-uuid-12345",
    "status": "running",
    "test_type": "all",
    "servers": ["all"],
    "started_at": "2025-08-15T17:00:00Z"
  }
}
```

### Get Test Results

#### GET /api/v1/tests/{test_id}

Get results of a test run.

**Response:**
```json
{
  "success": true,
  "data": {
    "test_id": "test-uuid-12345",
    "status": "completed",
    "test_type": "all",
    "started_at": "2025-08-15T17:00:00Z",
    "completed_at": "2025-08-15T17:10:00Z",
    "duration_seconds": 600,
    "summary": {
      "total": 250,
      "passed": 248,
      "failed": 2,
      "skipped": 0,
      "pass_rate": 0.992
    },
    "results": [
      {
        "server": "github",
        "tests": {
          "total": 50,
          "passed": 49,
          "failed": 1
        },
        "failures": [
          {
            "test": "test_rate_limiting",
            "error": "Expected 429, got 200",
            "trace": "..."
          }
        ]
      }
    ]
  }
}
```

### Test History

#### GET /api/v1/tests/history

Get test execution history.

**Query Parameters:**
- `server` (optional): Filter by server
- `test_type` (optional): Filter by test type
- `status` (optional): Filter by status
- `limit` (optional): Number of results (default: 50)
- `offset` (optional): Pagination offset

**Response:**
```json
{
  "success": true,
  "data": {
    "tests": [
      {
        "test_id": "test-uuid-12345",
        "test_type": "integration",
        "status": "completed",
        "pass_rate": 1.0,
        "duration_seconds": 300,
        "executed_at": "2025-08-15T17:00:00Z"
      }
    ],
    "pagination": {
      "total": 100,
      "limit": 50,
      "offset": 0,
      "has_more": true
    }
  }
}
```

## Cleanup Operations

### Run Cleanup

#### POST /api/v1/cleanup/run

Trigger cleanup operation.

**Request Body:**
```json
{
  "dry_run": false,
  "targets": ["logs", "cache", "temp"],
  "retention_days": 30,
  "force": false
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "cleanup_id": "cleanup-uuid-12345",
    "status": "running",
    "started_at": "2025-08-15T17:00:00Z",
    "targets": ["logs", "cache", "temp"]
  }
}
```

### Get Cleanup Status

#### GET /api/v1/cleanup/{cleanup_id}

Get status of cleanup operation.

**Response:**
```json
{
  "success": true,
  "data": {
    "cleanup_id": "cleanup-uuid-12345",
    "status": "completed",
    "started_at": "2025-08-15T17:00:00Z",
    "completed_at": "2025-08-15T17:02:00Z",
    "summary": {
      "files_deleted": 1500,
      "space_freed_mb": 2048,
      "errors": 0
    },
    "details": [
      {
        "target": "logs",
        "files_deleted": 1000,
        "space_freed_mb": 1024
      }
    ]
  }
}
```

### Storage Statistics

#### GET /api/v1/cleanup/stats

Get storage usage statistics.

**Response:**
```json
{
  "success": true,
  "data": {
    "total_space_gb": 100,
    "used_space_gb": 45,
    "free_space_gb": 55,
    "usage_percentage": 45,
    "breakdown": {
      "logs": {
        "size_mb": 5120,
        "file_count": 15000
      },
      "backups": {
        "size_mb": 10240,
        "file_count": 50
      },
      "cache": {
        "size_mb": 2048,
        "file_count": 5000
      }
    }
  }
}
```

## Monitoring & Metrics

### Get Metrics

#### GET /api/v1/metrics

Get Prometheus-formatted metrics.

**Response:**
```
# HELP mcp_servers_total Total number of MCP servers
# TYPE mcp_servers_total gauge
mcp_servers_total 17

# HELP mcp_servers_healthy Number of healthy MCP servers
# TYPE mcp_servers_healthy gauge
mcp_servers_healthy 17

# HELP mcp_update_duration_seconds Duration of update operations
# TYPE mcp_update_duration_seconds histogram
mcp_update_duration_seconds_bucket{le="30"} 5
mcp_update_duration_seconds_bucket{le="60"} 8
mcp_update_duration_seconds_bucket{le="120"} 10
```

### Get Logs

#### GET /api/v1/logs

Retrieve system logs.

**Query Parameters:**
- `level` (optional): Log level filter (debug, info, warning, error)
- `server` (optional): Filter by server
- `since` (optional): Start timestamp (ISO 8601)
- `until` (optional): End timestamp (ISO 8601)
- `limit` (optional): Maximum entries (default: 100)
- `query` (optional): Search query

**Response:**
```json
{
  "success": true,
  "data": {
    "logs": [
      {
        "timestamp": "2025-08-15T17:00:00Z",
        "level": "INFO",
        "server": "github",
        "message": "Health check completed successfully",
        "metadata": {
          "duration_ms": 50,
          "status": "healthy"
        }
      }
    ],
    "total": 1000,
    "returned": 100
  }
}
```

### Get Alerts

#### GET /api/v1/alerts

Get active alerts.

**Response:**
```json
{
  "success": true,
  "data": {
    "alerts": [
      {
        "id": "alert-uuid-12345",
        "severity": "warning",
        "server": "postgres",
        "title": "High memory usage",
        "description": "Memory usage above 80%",
        "triggered_at": "2025-08-15T17:00:00Z",
        "acknowledged": false
      }
    ],
    "total": 2,
    "critical": 0,
    "warning": 2,
    "info": 0
  }
}
```

### Acknowledge Alert

#### POST /api/v1/alerts/{alert_id}/acknowledge

Acknowledge an alert.

**Request Body:**
```json
{
  "message": "Investigating issue",
  "acknowledged_by": "operator"
}
```

## Orchestration Control

### Get Orchestrator Status

#### GET /api/v1/orchestration/status

Get orchestrator status and configuration.

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "running",
    "mode": "automatic",
    "workers": {
      "active": 5,
      "idle": 3,
      "total": 8
    },
    "queue": {
      "pending": 2,
      "processing": 1,
      "completed": 150
    },
    "configuration": {
      "max_workers": 10,
      "queue_size": 1000,
      "timeout_seconds": 300
    }
  }
}
```

### Execute Workflow

#### POST /api/v1/orchestration/workflows/execute

Execute a predefined workflow.

**Request Body:**
```json
{
  "workflow": "daily_maintenance",
  "parameters": {
    "cleanup": true,
    "update_check": true,
    "test_run": true
  },
  "schedule": "immediate"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "workflow_id": "workflow-uuid-12345",
    "workflow": "daily_maintenance",
    "status": "queued",
    "scheduled_at": "2025-08-15T17:00:00Z"
  }
}
```

### Get Workflow Status

#### GET /api/v1/orchestration/workflows/{workflow_id}

Get workflow execution status.

**Response:**
```json
{
  "success": true,
  "data": {
    "workflow_id": "workflow-uuid-12345",
    "workflow": "daily_maintenance",
    "status": "completed",
    "started_at": "2025-08-15T17:00:00Z",
    "completed_at": "2025-08-15T17:30:00Z",
    "steps": [
      {
        "name": "cleanup",
        "status": "completed",
        "duration_seconds": 120
      }
    ],
    "results": {
      "cleanup": {
        "files_deleted": 500,
        "space_freed_mb": 1024
      }
    }
  }
}
```

## WebSocket API

### Connection

Connect to WebSocket endpoint for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onopen = () => {
  console.log('Connected to WebSocket');
  
  // Authenticate
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'your-bearer-token'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### Subscribe to Events

```javascript
// Subscribe to specific events
ws.send(JSON.stringify({
  type: 'subscribe',
  events: ['server.status', 'update.progress', 'alert.new']
}));
```

### Event Types

| Event | Description | Payload |
|-------|-------------|---------|
| `server.status` | Server status change | `{server, status, timestamp}` |
| `update.progress` | Update progress | `{server, progress, message}` |
| `update.complete` | Update completed | `{server, version, success}` |
| `test.complete` | Test completed | `{test_id, results}` |
| `alert.new` | New alert | `{alert_id, severity, message}` |
| `cleanup.complete` | Cleanup completed | `{cleanup_id, summary}` |

### Unsubscribe

```javascript
ws.send(JSON.stringify({
  type: 'unsubscribe',
  events: ['server.status']
}));
```

## Error Handling

### Error Response Format

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "version",
      "error": "Version must be semantic version or 'latest'"
    },
    "trace_id": "trace-uuid-12345"
  },
  "metadata": {
    "timestamp": "2025-08-15T17:00:00Z",
    "request_id": "request-uuid-12345"
  }
}
```

### Common Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `AUTHENTICATION_REQUIRED` | Missing authentication | 401 |
| `INVALID_TOKEN` | Invalid or expired token | 401 |
| `PERMISSION_DENIED` | Insufficient permissions | 403 |
| `RESOURCE_NOT_FOUND` | Resource not found | 404 |
| `VALIDATION_ERROR` | Request validation failed | 422 |
| `CONFLICT` | Resource conflict | 409 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `INTERNAL_ERROR` | Server error | 500 |
| `SERVICE_UNAVAILABLE` | Service temporarily down | 503 |

## Rate Limiting

API implements rate limiting to ensure fair usage:

### Default Limits

| Endpoint Category | Limit | Window |
|------------------|-------|---------|
| Authentication | 5 requests | 1 minute |
| Read operations | 100 requests | 1 minute |
| Write operations | 20 requests | 1 minute |
| Update operations | 5 requests | 5 minutes |
| Cleanup operations | 2 requests | 10 minutes |

### Rate Limit Headers

Response headers indicate rate limit status:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1692121200
```

### Handling Rate Limits

When rate limited, the API returns:

```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Please retry after 60 seconds",
    "retry_after": 60
  }
}
```

## API Examples

### Complete Update Flow

```python
import requests
import time

# Configuration
BASE_URL = "http://localhost:8080/api/v1"
TOKEN = "your-bearer-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# 1. Check for updates
response = requests.get(f"{BASE_URL}/updates/check", headers=HEADERS)
updates = response.json()["data"]["servers"]

for server in updates:
    if server["update_available"]:
        print(f"Updating {server['name']} from {server['current_version']} to {server['latest_version']}")
        
        # 2. Initiate update
        update_response = requests.post(
            f"{BASE_URL}/updates/{server['name']}",
            headers=HEADERS,
            json={"version": "latest", "rollback_on_failure": True}
        )
        update_id = update_response.json()["data"]["update_id"]
        
        # 3. Monitor progress
        while True:
            status_response = requests.get(
                f"{BASE_URL}/updates/{update_id}",
                headers=HEADERS
            )
            status = status_response.json()["data"]["status"]
            
            if status in ["completed", "failed"]:
                print(f"Update {status} for {server['name']}")
                break
            
            time.sleep(5)
```

### Automated Testing

```javascript
const axios = require('axios');

const API = axios.create({
  baseURL: 'http://localhost:8080/api/v1',
  headers: {
    'Authorization': 'Bearer your-token'
  }
});

async function runTests() {
  try {
    // Start test run
    const testRun = await API.post('/tests/run', {
      test_type: 'integration',
      servers: ['all'],
      verbose: true
    });
    
    const testId = testRun.data.data.test_id;
    console.log(`Test started: ${testId}`);
    
    // Poll for results
    let complete = false;
    while (!complete) {
      const status = await API.get(`/tests/${testId}`);
      
      if (status.data.data.status === 'completed') {
        complete = true;
        const summary = status.data.data.summary;
        console.log(`Tests completed: ${summary.passed}/${summary.total} passed`);
        
        if (summary.failed > 0) {
          console.error('Test failures detected');
          // Handle failures
        }
      }
      
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
  } catch (error) {
    console.error('Test execution failed:', error.response.data);
  }
}

runTests();
```

### Monitoring Integration

```python
import requests
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# Setup
registry = CollectorRegistry()
servers_gauge = Gauge('mcp_servers_healthy', 'Number of healthy MCP servers', registry=registry)

# Get metrics from API
response = requests.get("http://localhost:8080/api/v1/status")
data = response.json()["data"]

# Update Prometheus metrics
servers_gauge.set(data["servers"]["healthy"])

# Push to Prometheus gateway
push_to_gateway('localhost:9091', job='mcp_monitor', registry=registry)
```

## SDK Libraries

### Python SDK

Installation:
```bash
pip install mcp-automation-sdk
```

Usage:
```python
from mcp_automation import Client

client = Client(base_url="http://localhost:8080", token="your-token")

# Get status
status = client.get_status()

# Update server
result = client.update_server("github", version="latest")

# Run tests
test_results = client.run_tests(test_type="all")
```

### JavaScript/TypeScript SDK

Installation:
```bash
npm install @mcp/automation-sdk
```

Usage:
```typescript
import { MCPClient } from '@mcp/automation-sdk';

const client = new MCPClient({
  baseUrl: 'http://localhost:8080',
  token: 'your-token'
});

// Get status
const status = await client.getStatus();

// Update server
const result = await client.updateServer('github', { version: 'latest' });

// Run tests
const testResults = await client.runTests({ testType: 'all' });
```

### Go SDK

Installation:
```bash
go get github.com/mcp/automation-sdk-go
```

Usage:
```go
package main

import (
    "github.com/mcp/automation-sdk-go"
)

func main() {
    client := mcp.NewClient("http://localhost:8080", "your-token")
    
    // Get status
    status, err := client.GetStatus()
    
    // Update server
    result, err := client.UpdateServer("github", "latest")
    
    // Run tests
    testResults, err := client.RunTests("all")
}
```

## API Versioning

The API uses URL path versioning:
- Current version: `/api/v1`
- Previous versions remain supported for backward compatibility
- Deprecation notices provided 6 months in advance
- Version sunset dates announced via API headers

### Version Headers

```http
X-API-Version: v1
X-API-Deprecation: 2026-02-15
X-API-Sunset: 2026-08-15
```

---

**API Reference Version**: 3.0.0  
**Last Updated**: 2025-08-15  
**Next Review**: 2025-09-15