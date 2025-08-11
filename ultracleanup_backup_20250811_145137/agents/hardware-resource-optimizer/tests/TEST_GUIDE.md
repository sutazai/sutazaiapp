# Hardware Optimizer Integration Testing Guide

## Overview

This comprehensive testing solution ensures the Hardware Resource Optimizer agent functions correctly in all scenarios. The testing framework validates actual system effects, not just API responses.

## Components

### 1. Integration Test Suite (`integration_test_suite.py`)
Complete end-to-end testing of all optimization workflows.

**Features:**
- 6 comprehensive test scenarios
- Real system state validation
- Actual file operations testing
- Performance measurement
- Detailed reporting

**Test Scenarios:**

#### Scenario 1: Full System Optimization
- Tests all optimization features together
- Validates memory, CPU, disk, and Docker cleanup
- Measures actual system improvement

#### Scenario 2: Storage Workflow
- Complete storage optimization pipeline
- Tests: analyze → find duplicates → remove → compress
- Validates actual space savings

#### Scenario 3: Resource Pressure
- Creates artificial memory pressure
- Tests optimization under load
- Validates memory recovery

#### Scenario 4: Docker Lifecycle
- Creates test containers
- Tests Docker cleanup operations
- Validates container and image removal

#### Scenario 5: Concurrent Operations
- Runs multiple optimizations simultaneously
- Tests thread safety and resource management
- Validates all operations complete successfully

#### Scenario 6: Error Recovery
- Tests error handling
- Validates protection of system files
- Tests invalid parameter handling
- Validates concurrent access management

### 2. Continuous Validator (`continuous_validator.py`)
Automated continuous testing with monitoring and alerting.

**Features:**
- Runs tests automatically every hour (configurable)
- Tracks metrics over time
- Generates alerts on failures
- Maintains historical data
- Provides trend analysis

**Alert Conditions:**
- Test failure rate > 20%
- Response time > 5 seconds
- 3 or more consecutive failures
- System resource thresholds exceeded

### 3. Visual Dashboard (`test_dashboard.html`)
Real-time web dashboard for monitoring test results.

**Features:**
- Live test status display
- Success rate and performance charts
- System metrics visualization
- Alert notifications
- Historical trend analysis
- Auto-refresh every 30 seconds

**Access:** http://localhost:8117 (when running with dashboard)

### 4. Test Runner Script (`run_tests.sh`)
Convenient script for running tests.

**Options:**
1. Run all test scenarios
2. Run single scenario
3. Run continuous tests
4. Run with dashboard
5. Install as service
6. Show reports

## Quick Start

### 1. Ensure Agent is Running
```bash
cd /opt/sutazaiapp/agents/hardware-resource-optimizer
python3 app.py
```

### 2. Run Integration Tests
```bash
cd tests
./run_tests.sh all
```

### 3. Start Continuous Monitoring with Dashboard
```bash
./run_tests.sh dashboard
# Access dashboard at http://localhost:8117
```

### 4. Install as System Service (Optional)
```bash
sudo ./run_tests.sh install
```

## Test Execution

### Manual Testing
```bash
# Run all scenarios
python3 integration_test_suite.py

# Run specific scenario
python3 integration_test_suite.py --scenario storage

# Run once with report
python3 continuous_validator.py --once
```

### Continuous Testing
```bash
# Basic continuous testing (60 min interval)
python3 continuous_validator.py

# With dashboard
python3 continuous_validator.py --dashboard

# Custom interval (30 minutes)
python3 continuous_validator.py --interval 30 --dashboard
```

## Understanding Results

### Test Reports
Reports are saved in `continuous_test_reports/`:
- `validation_TIMESTAMP.json` - Individual test runs
- `latest_results.json` - Most recent results
- `summary.txt` - Human-readable summaries
- `alerts.log` - Alert history

### Success Criteria
- **Overall Success**: ≥80% scenarios pass
- **Performance**: <1 second average response time
- **Reliability**: No consecutive failures
- **System Health**: CPU <80%, Memory <85%, Disk <90%

### Alert Levels
- **Warning**: Performance degradation or minor issues
- **Critical**: Test failures or system resource issues

## Troubleshooting

### Agent Not Responding
```bash
# Check if agent is running
curl http://localhost:8116/health

# Check agent logs
journalctl -u hardware-optimizer -f
```

### Test Failures
1. Check `integration_tests.log` for detailed errors
2. Verify system has sufficient resources
3. Ensure test data directory is writable
4. Check Docker daemon is running (for Docker tests)

### Dashboard Not Loading
1. Verify continuous validator is running with `--dashboard`
2. Check port 8117 is not in use
3. Look for errors in `continuous_validator.log`

## Advanced Usage

### Custom Alert Thresholds
Edit thresholds in `continuous_validator.py`:
```python
self.alert_thresholds = {
    "failure_rate": 20.0,      # Alert if >20% tests fail
    "response_time": 5.0,      # Alert if responses >5 seconds
    "consecutive_failures": 3   # Alert after 3 failures
}
```

### Email Alerts (Future Enhancement)
The alert system is designed to support email notifications. To enable:
1. Configure SMTP settings in continuous_validator.py
2. Add email addresses to alert recipients
3. Implement the email sending in `_send_alerts()` method

### Integration with Monitoring Systems
Test results can be integrated with:
- Prometheus (metrics endpoint)
- Grafana (visualization)
- PagerDuty (alerting)
- Slack (notifications)

## Maintenance

### Regular Tasks
1. **Weekly**: Review test reports and trends
2. **Monthly**: Clean old test reports (keep last 30 days)
3. **Quarterly**: Review and update test scenarios

### Updating Tests
1. Add new scenarios to `integration_test_suite.py`
2. Update scenario list in `continuous_validator.py`
3. Add to dashboard display in `test_dashboard.html`

## Best Practices

1. **Run Tests Before Deployment**: Always validate changes
2. **Monitor Trends**: Watch for gradual performance degradation  
3. **Investigate Failures**: Don't ignore intermittent failures
4. **Keep Tests Updated**: Add tests for new features
5. **Document Issues**: Record any persistent problems

## Support

For issues or questions:
1. Check logs in `continuous_test_reports/`
2. Review this guide
3. Check agent documentation
4. Contact system administrators

Remember: These tests ensure the Hardware Optimizer maintains 100% reliability in production!