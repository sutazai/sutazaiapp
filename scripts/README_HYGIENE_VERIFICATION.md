# Hygiene Monitoring System Verification

## Overview

This documentation describes the comprehensive verification script for the dockerized hygiene monitoring system. The script tests all endpoints and functionality to ensure the system is working properly in production.

## Script Location

- **Main Script**: `/opt/sutazaiapp/scripts/verify-hygiene-monitoring-system.py`
- **Documentation**: `/opt/sutazaiapp/scripts/README_HYGIENE_VERIFICATION.md`

## What Gets Tested

### 1. HTTP Endpoints

#### Dashboard (http://localhost:3002)
- `/` - Dashboard root page
- `/health` - Dashboard health check
- `/api/status` - Dashboard API status

#### Backend API (http://localhost:8081)
- `/api/hygiene/status` - Backend health status
- `/api/hygiene/violations` - Hygiene violations list
- `/api/hygiene/metrics` - System metrics
- `/api/hygiene/scan` - Hygiene scan trigger
- `/api/hygiene/reports` - Hygiene reports
- `/api/hygiene/rules` - Active rules

#### Rule Control API (http://localhost:8101)
- `/api/health/live` - Rule API health
- `/api/rules` - Rule management
- `/api/rules/profiles` - Rule profiles
- `/api/rules/status` - Rule status
- `/api/config` - Rule configuration

### 2. WebSocket Connectivity

- Tests connection to `ws://localhost:8081/ws`
- Verifies bidirectional communication
- Measures response times

### 3. Real Data Verification

#### System Metrics Validation
- Checks that CPU, memory metrics are real (not hardcoded)
- Validates timestamp freshness
- Ensures system data reflects actual state

#### Violations Data Validation
- Verifies violations contain real file paths
- Checks timestamps are current
- Validates rule IDs and severity levels

### 4. Violations Detection

- Triggers hygiene scans
- Verifies violations are detected and reported
- Tests scan results persistence

## Usage

### Basic Usage

```bash
# Run verification with default settings
python scripts/verify-hygiene-monitoring-system.py

# Run with verbose output
python scripts/verify-hygiene-monitoring-system.py --verbose

# Save report to file
python scripts/verify-hygiene-monitoring-system.py --output-file verification-report.json
```

### Advanced Options

```bash
# Custom timeout and verbose output
python scripts/verify-hygiene-monitoring-system.py --verbose --timeout 60

# Full command with all options
python scripts/verify-hygiene-monitoring-system.py \
    --verbose \
    --timeout 30 \
    --output-file reports/hygiene-verification-$(date +%Y%m%d_%H%M%S).json
```

### Integration with CI/CD

```bash
# Use in automated testing
if python scripts/verify-hygiene-monitoring-system.py --output-file ci-verification.json; then
    echo "✅ Hygiene monitoring system verified"
else
    echo "❌ Hygiene monitoring system verification failed"
    exit 1
fi
```

## Output Format

### Console Output

The script provides real-time feedback with emoji indicators:

- ✅ PASS - Test passed successfully
- ❌ FAIL - Test failed
- ⚠️ WARN - Test passed with warnings
- ⏭️ SKIP - Test was skipped

### JSON Report Structure

```json
{
  "verification_summary": {
    "timestamp": "2025-08-04T12:00:00",
    "total_tests": 25,
    "passed": 22,
    "failed": 1,
    "warnings": 2,
    "skipped": 0,
    "success_rate": 0.88,
    "average_response_time_ms": 156.4,
    "overall_status": "HEALTHY|DEGRADED|UNHEALTHY"
  },
  "endpoints_tested": {
    "dashboard": "http://localhost:3002",
    "backend_api": "http://localhost:8081",
    "rule_control_api": "http://localhost:8101",
    "websocket": "ws://localhost:8081/ws"
  },
  "results_by_category": {
    "Dashboard": {
      "total": 3,
      "passed": 3,
      "failed": 0,
      "warnings": 0,
      "tests": [...]
    }
  },
  "detailed_results": [
    {
      "test_name": "Dashboard Root",
      "category": "Dashboard",
      "endpoint": "http://localhost:3002/",
      "status": "PASS",
      "message": "HTTP 200",
      "response_time_ms": 45.2,
      "timestamp": "2025-08-04T12:00:01"
    }
  ]
}
```

## Exit Codes

- `0` - All tests passed (HEALTHY)
- `1` - Tests passed with warnings (DEGRADED)
- `2` - Tests failed (UNHEALTHY)
- `3` - Script execution error

## Prerequisites

### Required Services

Before running the verification script, ensure these services are running:

```bash
# Start the hygiene monitoring system
docker-compose -f docker-compose.hygiene-monitor.yml up -d

# Verify services are up
docker-compose -f docker-compose.hygiene-monitor.yml ps
```

### Required Python Dependencies

The script requires these packages (included in project requirements):

- `httpx` - HTTP client
- `websockets` - WebSocket client
- `aiofiles` - Async file operations

Install if missing:
```bash
pip install httpx websockets aiofiles
```

## Troubleshooting

### Common Issues

#### Connection Refused
```
❌ Dashboard Root: Request failed
Error: Connection refused
```

**Solution**: Ensure Docker services are running:
```bash
docker-compose -f docker-compose.hygiene-monitor.yml up -d
```

#### Timeout Errors
```
❌ Backend API Health: Request timeout
```

**Solutions**:
1. Increase timeout: `--timeout 60`
2. Check container health: `docker-compose -f docker-compose.hygiene-monitor.yml ps`
3. Review container logs: `docker-compose -f docker-compose.hygiene-monitor.yml logs`

#### WebSocket Connection Failed
```
❌ WebSocket Connection: WebSocket connection failed
```

**Solutions**:
1. Verify backend API is running
2. Check WebSocket endpoint in backend logs
3. Test with: `curl -v --http1.1 --upgrade Upgrade --header "Connection: Upgrade" --header "Upgrade: websocket" http://localhost:8081/ws`

#### No Violations Detected
```
⚠️ Violations Detection: No violations detected
```

This is normal if:
- Codebase is clean and follows all rules
- Scanning is working but no issues found

To verify scanning works:
1. Temporarily create rule violations
2. Run scan manually
3. Check violations endpoint

### Debug Mode

For detailed debugging, run with maximum verbosity:

```bash
python scripts/verify-hygiene-monitoring-system.py --verbose --timeout 120
```

This will show:
- Full HTTP request/response details
- WebSocket message contents
- Complete error stack traces
- Timing information for each test

## Integration Examples

### Pre-Deployment Verification

```bash
#!/bin/bash
# pre-deploy-verification.sh

echo "🔍 Verifying Hygiene Monitoring System..."

# Run verification
if python scripts/verify-hygiene-monitoring-system.py --output-file verification-report.json; then
    echo "✅ Pre-deployment verification passed"
    
    # Extract key metrics
    success_rate=$(jq -r '.verification_summary.success_rate' verification-report.json)
    echo "Success Rate: $(echo "$success_rate * 100" | bc -l)%"
    
    exit 0
else
    echo "❌ Pre-deployment verification failed"
    echo "Check verification-report.json for details"
    exit 1
fi
```

### Monitoring Integration

```bash
#!/bin/bash
# health-check-monitoring.sh

# Run verification and send results to monitoring
python scripts/verify-hygiene-monitoring-system.py --output-file /tmp/health-check.json

# Send to monitoring system (example with curl)
curl -X POST http://monitoring-system/api/health-checks \
    -H "Content-Type: application/json" \
    -d @/tmp/health-check.json
```

### Cron Job Example

```bash
# Add to crontab for regular verification
# crontab -e

# Run verification every 30 minutes
*/30 * * * * /opt/sutazaiapp/scripts/verify-hygiene-monitoring-system.py --output-file /opt/sutazaiapp/logs/hygiene-verification-$(date +\%Y\%m\%d_\%H\%M\%S).json

# Daily summary report
0 9 * * * /opt/sutazaiapp/scripts/verify-hygiene-monitoring-system.py --verbose --output-file /opt/sutazaiapp/reports/daily-hygiene-verification-$(date +\%Y\%m\%d).json
```

## Maintenance

### Regular Tasks

1. **Weekly**: Review verification reports for trends
2. **Monthly**: Update expected response time thresholds
3. **Quarterly**: Review and update test endpoints

### Script Updates

When updating the hygiene monitoring system:

1. Update endpoint URLs in script if changed
2. Add new endpoint tests as needed
3. Update expected response formats
4. Test script with new system version

### Performance Baselines

Track these metrics over time:

- Average response time per endpoint
- Success rate trends
- WebSocket connection stability
- Violations detection rate

## Support

For issues with the verification script:

1. Check container logs: `docker-compose -f docker-compose.hygiene-monitor.yml logs`
2. Verify service health: `docker-compose -f docker-compose.hygiene-monitor.yml ps`
3. Run with `--verbose` flag for detailed output
4. Check network connectivity between containers

The verification script is designed to be robust and provide clear feedback on system health. Regular use ensures the hygiene monitoring system maintains production readiness.