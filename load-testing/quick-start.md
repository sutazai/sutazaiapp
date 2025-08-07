# SutazAI Load Testing - Quick Start Guide

## Overview
This guide helps you quickly set up and run comprehensive load tests for the SutazAI multi-agent platform.

## Prerequisites

### System Requirements
- Linux/macOS/WSL environment
- Docker and Docker Compose running
- At least 8GB RAM and 4 CPU cores
- 10GB free disk space

### Required Tools
```bash
# Install K6 (load testing tool)
curl https://github.com/grafana/k6/releases/download/v0.47.0/k6-v0.47.0-linux-amd64.tar.gz -L | tar xvz --strip-components 1
sudo mv k6 /usr/local/bin/

# Install jq (JSON processing)
sudo apt-get install jq  # Ubuntu/Debian
# or
brew install jq         # macOS

# Verify installations
k6 version
jq --version
```

## Quick Setup

### 1. Ensure SutazAI is Running
```bash
# Check system status
cd /opt/sutazaiapp
docker-compose ps

# Start services if needed
docker-compose up -d

# Verify health
curl http://localhost:8000/health
curl http://localhost:8501/
```

### 2. Generate Agent Test Configuration
```bash
cd /opt/sutazaiapp/load-testing
python3 generate-agent-tests.py
```

## Running Load Tests

### Basic Usage

#### 1. Run All Tests (Recommended for first time)
```bash
./run-load-tests.sh
```

#### 2. Run Specific Test Suites
```bash
# Test individual agents
./run-load-tests.sh -s agents -v 50 -d 300s

# Test database performance
./run-load-tests.sh -s database -v 100 -d 600s

# Test Jarvis concurrent users
./run-load-tests.sh -s jarvis -v 200 -d 900s

# Test service mesh resilience
./run-load-tests.sh -s mesh -v 150 -d 450s

# Test API gateway throughput
./run-load-tests.sh -s gateway -v 300 -d 300s

# Full system integration test
./run-load-tests.sh -s integration -v 100 -d 1200s

# Stress testing to find breaking points
./run-load-tests.sh -s stress -v 500 -d 600s
```

#### 3. Custom Configuration
```bash
# Test staging environment
./run-load-tests.sh --base-url http://staging.sutazai.com -s all -v 100

# Quick smoke test
./run-load-tests.sh -v 10 -d 60s -s agents

# Heavy load test
./run-load-tests.sh -v 1000 -d 1800s -s stress
```

### Advanced Usage

#### Individual K6 Tests
```bash
# Set environment variables
export BASE_URL="http://localhost"
export VUS=50
export DURATION="300s"

# Run specific test files
k6 run tests/agent-performance.js
k6 run tests/database-load.js
k6 run tests/jarvis-concurrent.js
k6 run tests/system-integration.js

# Run with custom options
k6 run --vus 100 --duration 600s tests/breaking-point-stress.js

# Run all agents load test
k6 run tests/all-agents-load.js
```

#### Breaking Point Analysis
```bash
# Progressive stress test
k6 run --scenarios progressive_stress tests/breaking-point-stress.js

# Spike load test
K6_SCENARIO=spike_test k6 run tests/breaking-point-stress.js

# Soak test (30 minutes)
K6_SCENARIO=soak_test k6 run tests/breaking-point-stress.js

# Volume test with large payloads
K6_SCENARIO=volume_test k6 run tests/breaking-point-stress.js
```

## Understanding Results

### Test Output
```bash
# Live test output shows:
✓ Agent responds within SLA
✓ Database query successful  
✓ No connection pool exhaustion
✗ Response time under 2s (actual: 3.2s)

# Summary at end:
checks.........................: 89.2% ✓ 2145 ✗ 260
data_received..................: 45 MB  150 kB/s
data_sent......................: 12 MB  40 kB/s  
http_req_blocked...............: avg=1.2ms   min=0.1ms med=0.8ms  max=45ms  p(90)=2.1ms p(95)=3.4ms
http_req_connecting............: avg=0.5ms   min=0ms   med=0.3ms  max=23ms  p(90)=1.1ms p(95)=1.8ms
http_req_duration..............: avg=1.8s    min=102ms med=1.2s   max=12s   p(90)=3.2s  p(95)=4.5s
http_req_failed................: 10.8% ✗ 260  ✓ 2145
http_reqs......................: 2405   8.0/s
```

### Key Metrics to Monitor
- **http_req_duration**: Response time (aim for p95 < 3s)
- **http_req_failed**: Error rate (aim for < 1%)
- **http_reqs**: Throughput (requests per second)
- **checks**: Pass rate for assertions (aim for > 95%)

### Generated Reports
After tests complete, check the `reports/` directory:
```bash
ls -la reports/
# - comprehensive_report_TIMESTAMP.html (main report)
# - agent-performance_TIMESTAMP.json (raw data)
# - baseline_comparison_TIMESTAMP.json (vs SLAs)
# - breaking_point_report.json (stress test results)
```

## Generate Comprehensive Reports

### Automatic Report Generation
```bash
# Generate reports from existing test data
./generate-reports.sh

# View HTML report
open reports/comprehensive_report_*.html
```

### Manual Report Analysis
```bash
# View specific test results
jq '.metrics.http_req_duration' reports/agent-performance_*.json

# Compare with baselines
jq '.comparisons' reports/baseline_comparison_*.json

# Check breaking points
jq '.breaking_points' reports/breaking_point_report.json
```

## Common Test Scenarios

### 1. Pre-Production Validation
```bash
# Full system validation before deployment
./run-load-tests.sh -s all -v 200 -d 1800s --base-url http://staging.sutazai.com
```

### 2. Performance Regression Testing
```bash
# Quick performance check after code changes
./run-load-tests.sh -s agents -v 100 -d 300s
```

### 3. Capacity Planning
```bash
# Find maximum concurrent users
./run-load-tests.sh -s stress -v 1000 -d 900s
```

### 4. Database Performance Analysis
```bash
# Focus on database under load
./run-load-tests.sh -s database -v 300 -d 1200s
```

### 5. User Experience Testing
```bash
# Simulate real user journeys
./run-load-tests.sh -s jarvis -v 150 -d 1800s
```

## Troubleshooting

### Common Issues

#### Tests Fail to Start
```bash
# Check system health
curl http://localhost:8000/health
docker-compose ps

# Restart services
docker-compose restart
```

#### High Error Rates
```bash
# Check service logs
docker-compose logs backend
docker-compose logs ollama
docker-compose logs postgres

# Reduce load and retry
./run-load-tests.sh -v 25 -d 300s -s agents
```

#### Out of Memory Errors
```bash
# Check system resources
free -h
docker stats

# Reduce virtual users
./run-load-tests.sh -v 50 -d 300s
```

#### Connection Timeouts
```bash
# Check network connectivity
ping localhost
netstat -tuln | grep :8000

# Increase timeouts in tests
export K6_TIMEOUT=60s
```

### Getting Help

#### View Test Configuration
```bash
# Check current baselines
cat performance-baselines.yaml

# View agent configuration  
cat agent-ports.json

# Check test parameters
head -50 k6-config.js
```

#### Debug Individual Tests
```bash
# Run single agent test with verbose output
k6 run --http-debug tests/agent-performance.js

# Test specific agent endpoint
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"test","max_tokens":10}'
```

## Performance Targets

### Acceptance Criteria
- **Response Time P95**: < 3 seconds
- **Response Time P99**: < 5 seconds  
- **Error Rate**: < 1%
- **Throughput**: > 100 req/s
- **Concurrent Users**: 1000+
- **Agent Availability**: > 99%

### Breaking Point Thresholds
- **Critical Response Time**: > 10 seconds
- **Critical Error Rate**: > 5%
- **System Failure**: > 50% error rate
- **Resource Exhaustion**: > 95% CPU/Memory

## Next Steps

### After Running Tests
1. **Review Results**: Check HTML reports and JSON summaries
2. **Compare Baselines**: Validate against performance targets
3. **Identify Issues**: Focus on failed checks and high error rates
4. **Implement Fixes**: Use optimization recommendations
5. **Re-test**: Verify improvements with follow-up tests

### Continuous Testing
```bash
# Set up automated testing (example cron job)
# Run performance tests daily at 2 AM
0 2 * * * cd /opt/sutazaiapp/load-testing && ./run-load-tests.sh -s agents -v 100 -d 600s

# Weekly comprehensive test
0 3 * * 0 cd /opt/sutazaiapp/load-testing && ./run-load-tests.sh -s all -v 200 -d 1800s
```

### Integration with CI/CD
```yaml
# Example GitHub Actions workflow
name: Load Testing
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Load Tests
        run: |
          cd load-testing
          ./run-load-tests.sh -s agents -v 50 -d 300s
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: load-test-results
          path: load-testing/reports/
```

## Support

For issues with load testing:
1. Check the troubleshooting section above
2. Review generated error logs in `logs/` directory
3. Examine system logs: `docker-compose logs`
4. Validate system health: `./health-check.sh`

---

**Happy Load Testing!** 🚀

This framework will help ensure SutazAI can handle production workloads reliably and efficiently.