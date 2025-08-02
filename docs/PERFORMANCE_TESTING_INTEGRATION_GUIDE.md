# SutazAI Performance Testing & Monitoring Integration Guide

## Overview

This guide provides comprehensive instructions for integrating and using the AI-powered performance testing suite and real-time monitoring dashboard for the SutazAI system. The solution includes automated performance benchmarking, AI-driven analysis, and continuous monitoring capabilities.

## Components

### 1. Performance Requirements and Quality Benchmarks
- **Location**: `/opt/sutazaiapp/docs/PERFORMANCE_REQUIREMENTS_AND_QUALITY_BENCHMARKS.md`
- **Purpose**: Defines comprehensive performance targets and quality metrics
- **Key Features**: API response time thresholds, resource utilization targets, reliability requirements

### 2. AI-Powered Performance Testing Suite
- **Location**: `/opt/sutazaiapp/agents/testing-qa-validator/performance_testing_suite.py`
- **Purpose**: Automated performance testing with AI analysis
- **Key Features**: Load testing, stress testing, endurance testing, AI insights

### 3. Real-time Performance Monitoring Dashboard
- **Location**: `/opt/sutazaiapp/agents/testing-qa-validator/performance_monitoring_dashboard.py`
- **Purpose**: Live performance monitoring with AI-powered insights
- **Key Features**: WebSocket-based real-time updates, alerting, predictive analysis

## Installation and Setup

### Prerequisites

1. **Python Dependencies**
```bash
pip install fastapi uvicorn websockets aiohttp redis psutil numpy pandas scikit-learn transformers plotly matplotlib seaborn pytest pytest-asyncio
```

2. **System Requirements**
- Redis server (for metric storage)
- At least 8GB RAM
- Python 3.8+
- Access to SutazAI API endpoints

3. **Optional Dependencies**
```bash
pip install pytest-html pytest-cov k6  # For advanced testing features
```

### Configuration

1. **Environment Variables**
```bash
export SUTAZAI_BASE_URL="http://localhost:8000"
export REDIS_URL="redis://localhost:6379/0"
export PERFORMANCE_LOG_LEVEL="INFO"
export MONITORING_INTERVAL="30"  # seconds
```

2. **Redis Configuration**
Ensure Redis is running and accessible:
```bash
redis-cli ping  # Should return PONG
```

## Usage Guide

### Performance Testing Suite

#### 1. Manual Execution

**Basic Performance Test**
```bash
cd /opt/sutazaiapp/agents/testing-qa-validator
python performance_testing_suite.py
```

**Pytest Integration**
```bash
# Run all performance tests
pytest performance_testing_suite.py -v -m performance

# Run specific test categories
pytest performance_testing_suite.py -m "api_performance"
pytest performance_testing_suite.py -m "load"

# Generate HTML report
pytest performance_testing_suite.py --html=report.html --self-contained-html
```

#### 2. Programmatic Usage

```python
from performance_testing_suite import PerformanceTestSuite

async def run_custom_test():
    suite = PerformanceTestSuite(base_url="http://localhost:8000")
    await suite.initialize()
    
    try:
        # Run comprehensive test suite
        results = await suite.run_comprehensive_suite()
        
        # Generate report
        report = suite.generate_performance_report(results)
        print(report)
        
        # Check specific benchmarks
        compliance = suite._check_benchmark_compliance()
        print(f"Benchmark compliance: {compliance}")
        
    finally:
        await suite.cleanup()

# Run the test
import asyncio
asyncio.run(run_custom_test())
```

#### 3. CI/CD Integration

**GitHub Actions Example**
```yaml
name: Performance Tests

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  push:
    branches: [main]

jobs:
  performance-test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-html
    
    - name: Start services
      run: |
        docker-compose -f docker-compose.minimal.yml up -d
        sleep 30  # Wait for services to start
    
    - name: Run performance tests
      run: |
        cd agents/testing-qa-validator
        pytest performance_testing_suite.py -v --html=performance_report.html
    
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: performance-report
        path: agents/testing-qa-validator/performance_report.html
```

### Real-time Monitoring Dashboard

#### 1. Starting the Dashboard

```bash
cd /opt/sutazaiapp/agents/testing-qa-validator
python performance_monitoring_dashboard.py
```

The dashboard will be available at: `http://localhost:8001`

#### 2. Dashboard Features

**Real-time Metrics**
- System CPU, Memory, Disk usage
- API response times and availability
- Process-specific metrics
- Network I/O statistics

**AI-Powered Insights**
- Anomaly detection
- Performance trend analysis
- Predictive alerts
- Optimization recommendations

**Interactive Charts**
- Live updating visualizations
- Historical data views
- Comparative analysis
- Custom time ranges

#### 3. API Endpoints

**Current Metrics**
```bash
curl http://localhost:8001/api/metrics/current
```

**Historical Data**
```bash
curl "http://localhost:8001/api/metrics/historical/system/cpu_usage?hours=24"
```

**Active Alerts**
```bash
curl http://localhost:8001/api/alerts/active
```

**AI Insights**
```bash
curl http://localhost:8001/api/analysis/ai-insights
```

**Performance Predictions**
```bash
curl "http://localhost:8001/api/analysis/predictions?horizon_minutes=60"
```

#### 4. WebSocket Integration

```javascript
// Connect to real-time metrics
const ws = new WebSocket('ws://localhost:8001/ws/metrics');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'metrics_update') {
        // Handle new metrics
        updateDashboard(data.metrics);
    } else if (data.type === 'ai_insights') {
        // Handle AI analysis
        updateInsights(data.insights);
    }
};
```

## Integration with SutazAI System

### 1. Docker Integration

Add monitoring service to docker-compose:

```yaml
# Add to docker-compose.yml
services:
  performance-monitor:
    build:
      context: ./agents/testing-qa-validator
      dockerfile: Dockerfile.monitoring
    container_name: sutazai-performance-monitor
    ports:
      - "8001:8001"
    environment:
      - SUTAZAI_BASE_URL=http://backend:8000
      - REDIS_URL=redis://redis:6379/0
      - MONITORING_INTERVAL=30
    depends_on:
      - backend
      - redis
    networks:
      - sutazai-network
```

### 2. Agent Integration

```python
# In agent code
from agents.testing_qa_validator.performance_testing_suite import PerformanceTestSuite

class EnhancedAgent:
    def __init__(self):
        self.performance_suite = PerformanceTestSuite()
        
    async def validate_performance(self):
        """Run performance validation before critical operations"""
        results = await self.performance_suite._test_api_latency()
        
        # Check if system is healthy enough for operation
        for endpoint, metrics in results.items():
            if metrics['p95'] > 2000:  # 2 second threshold
                raise PerformanceException(f"System too slow: {endpoint}")
```

### 3. Monitoring Integration

```python
# Custom monitoring setup
from agents.testing_qa_validator.performance_monitoring_dashboard import PerformanceMonitor

class SutazAIWithMonitoring:
    def __init__(self):
        self.monitor = PerformanceMonitor()
        
    async def start_system(self):
        await self.monitor.initialize()
        asyncio.create_task(self.monitor.start_monitoring())
        
        # Start your application
        await self.start_agents()
```

## Performance Benchmarks Summary

### Critical API Endpoints

| Endpoint | P50 Target | P95 Target | P99 Target | Success Rate |
|----------|------------|------------|------------|--------------|
| `/health` | 10ms | 25ms | 50ms | 99.95% |
| `/api/v1/system/status` | 50ms | 150ms | 300ms | 99.9% |
| `/api/v1/coordinator/think` | 500ms | 2000ms | 5000ms | 95.0% |
| `/api/v1/vectors/search` | 100ms | 300ms | 600ms | 98.0% |

### Resource Utilization Targets

| Resource | Normal | Warning | Critical |
|----------|--------|---------|----------|
| CPU Usage | <40% | 70% | 85% |
| Memory Usage | <60% | 75% | 90% |
| Disk Usage | <70% | 80% | 95% |

### Quality Metrics

| Component | Coverage Target | Security | Performance |
|-----------|----------------|----------|-------------|
| Backend API | 85% | Zero Critical | P95 < 500ms |
| Agent Core | 80% | <2 High | P95 < 2000ms |
| ML Pipeline | 75% | <5 Medium | Inference < 3s |

## Alerting and Notifications

### Alert Severity Levels

**Critical (P0)**
- System unavailable
- Data loss risk
- Security breach
- Response: < 5 minutes

**High (P1)**
- Performance degradation
- Agent failures
- High error rates
- Response: < 15 minutes

**Medium (P2)**
- Resource warnings
- Non-critical errors
- Response: < 1 hour

**Low (P3)**
- Maintenance reminders
- Trend notifications
- Response: < 4 hours

### Integration with External Systems

**Slack Integration**
```python
# Add to monitoring dashboard
import slack_sdk

class SlackAlerter:
    def __init__(self, token):
        self.client = slack_sdk.WebClient(token=token)
        
    async def send_alert(self, alert: PerformanceAlert):
        if alert.severity in ['critical', 'high']:
            await self.client.chat_postMessage(
                channel='#alerts',
                text=f"🚨 {alert.severity.upper()}: {alert.message}"
            )
```

**PagerDuty Integration**
```python
import pypagerduty

class PagerDutyAlerter:
    def __init__(self, api_token, service_id):
        self.pager = pypagerduty.PagerDuty(api_token)
        self.service_id = service_id
        
    async def trigger_alert(self, alert: PerformanceAlert):
        if alert.severity == 'critical':
            self.pager.trigger_incident(
                service_id=self.service_id,
                incident_key=alert.id,
                description=alert.message
            )
```

## Troubleshooting

### Common Issues

**1. WebSocket Connection Failures**
```bash
# Check if port is available
netstat -tulpn | grep :8001

# Verify firewall settings
sudo ufw status

# Check Redis connection
redis-cli -h localhost -p 6379 ping
```

**2. Performance Test Timeouts**
```python
# Increase timeout settings
suite = PerformanceTestSuite()
suite.session = aiohttp.ClientSession(
    timeout=aiohttp.ClientTimeout(total=60)  # 60 seconds
)
```

**3. High Memory Usage**
```python
# Limit metric history
monitor = PerformanceMonitor()
monitor.max_history_size = 5000  # Reduce from default 10000
```

**4. AI Model Loading Issues**
```bash
# Install required packages
pip install transformers torch

# Check available disk space
df -h

# Monitor memory during model loading
watch -n 1 free -h
```

## Best Practices

### 1. Performance Testing
- Run tests in isolation to avoid interference
- Use consistent test data and conditions
- Monitor resource usage during tests
- Establish baseline measurements before optimization
- Run tests regularly (daily/weekly) to catch regressions

### 2. Monitoring Setup
- Start with default thresholds and adjust based on observations
- Monitor trends, not just absolute values
- Set up gradual alerting (warning → critical)
- Review and update thresholds quarterly
- Document alert resolution procedures

### 3. AI Insights
- Allow sufficient data collection before relying on AI analysis
- Validate AI recommendations with domain expertise
- Use insights for proactive optimization, not reactive fixes
- Monitor false positive rates and adjust sensitivity
- Combine AI insights with traditional monitoring approaches

### 4. Resource Management
- Implement circuit breakers for critical operations
- Use connection pooling for external services
- Monitor and limit concurrent operations
- Implement graceful degradation strategies
- Plan for traffic spikes and scaling needs

## Advanced Configuration

### Custom Thresholds

```python
# Customize performance thresholds
custom_thresholds = {
    "/api/v1/coordinator/think": BenchmarkThresholds(
        p50_ms=300,    # Faster requirement
        p95_ms=1500,   # Tighter P95
        p99_ms=3000,   # Stricter P99
        max_acceptable_ms=5000,
        success_rate=98.0,  # Higher success rate
        throughput_rps=75   # Higher throughput
    )
}

suite = PerformanceTestSuite()
suite.thresholds.update(custom_thresholds)
```

### Custom Metrics Collection

```python
# Add custom metrics to monitoring
async def collect_custom_metrics():
    # Database connection pool metrics
    pool_size = await get_db_pool_size()
    active_connections = await get_active_db_connections()
    
    return [
        MetricSnapshot(
            timestamp=time.time(),
            component="database",
            metric_name="pool_size",
            value=pool_size,
            unit="connections",
            status="normal"
        ),
        MetricSnapshot(
            timestamp=time.time(),
            component="database", 
            metric_name="active_connections",
            value=active_connections,
            unit="connections",
            status="normal"
        )
    ]

# Integrate with monitoring
monitor = PerformanceMonitor()
monitor.custom_collectors.append(collect_custom_metrics)
```

### Performance Optimization Triggers

```python
# Automatic optimization based on performance metrics
class AutoOptimizer:
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        
    async def check_optimization_triggers(self):
        recent_metrics = self.monitor.metric_history[-100:]
        
        # Check if response times are consistently high
        api_metrics = [m for m in recent_metrics if 'response_time' in m.metric_name]
        if api_metrics:
            avg_response_time = sum(m.value for m in api_metrics) / len(api_metrics)
            
            if avg_response_time > 1000:  # 1 second average
                await self.trigger_cache_warming()
                await self.trigger_connection_pool_scaling()
                
    async def trigger_cache_warming(self):
        # Implement cache warming logic
        pass
        
    async def trigger_connection_pool_scaling(self):
        # Implement connection pool scaling
        pass
```

## Conclusion

This comprehensive performance testing and monitoring solution provides:

1. **Automated Performance Validation**: Continuous testing against defined benchmarks
2. **Real-time Monitoring**: Live system health tracking with AI insights
3. **Predictive Analysis**: Future performance prediction and proactive optimization
4. **Flexible Integration**: Easy integration with existing SutazAI components
5. **Enterprise-Ready**: Scalable, reliable, and production-ready implementation

The system ensures that SutazAI maintains optimal performance, meets quality benchmarks, and provides exceptional user experience through intelligent monitoring and testing capabilities.

For support or questions, refer to the component documentation or contact the SutazAI development team.

---

**Document Version**: 1.0  
**Last Updated**: 2025-08-01  
**Next Review**: 2025-11-01  
**Owner**: Testing QA Validator Agent