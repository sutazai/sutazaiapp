# Circuit Breaker Implementation

## Overview

The circuit breaker pattern has been implemented to provide fault tolerance and resilience for all external service communications in the SutazAI backend. This prevents cascading failures and provides graceful degradation when services are unavailable.

## Features

### Core Capabilities
- **Automatic failure detection** - Monitors consecutive failures and opens circuit after threshold
- **Smart recovery** - Automatic transition to half-open state after recovery timeout
- **Service isolation** - Independent circuit breakers for each service type
- **Comprehensive metrics** - Detailed tracking of success rates, failures, and state changes
- **Manual control** - API endpoints for monitoring and manual circuit reset

### Protected Services
1. **Ollama** - LLM service calls (5 failures, 30s recovery)
2. **Redis** - Cache operations (5 failures, 30s recovery)
3. **Database** - PostgreSQL queries (5 failures, 30s recovery)
4. **Agent Services** - Agent API calls (5 failures, 30s recovery)
5. **External APIs** - Third-party service calls (5 failures, 30s recovery)

## Circuit Breaker States

### CLOSED (Normal Operation)
- All requests pass through normally
- Failures are counted but don't block requests
- Transitions to OPEN after 5 consecutive failures

### OPEN (Circuit Tripped)
- All requests fail immediately with `CircuitBreakerError`
- No load on the failing service
- Waits 30 seconds before attempting recovery

### HALF_OPEN (Testing Recovery)
- Allows one test request through
- Success → transitions to CLOSED
- Failure → transitions back to OPEN

## Usage Examples

### 1. HTTP Requests with Circuit Breaker

```python
from app.core.connection_pool import get_pool_manager
from app.core.circuit_breaker import CircuitBreakerError

pool_manager = await get_pool_manager()

try:
    # Protected HTTP request to Ollama
    response = await pool_manager.make_http_request(
        service='ollama',
        method='POST',
        url='/api/generate',
        json={'model': 'tinyllama', 'prompt': 'Hello'}
    )
    print(f"Success: {response.json()}")
    
except CircuitBreakerError as e:
    print(f"Service unavailable (circuit open): {e}")
    # Implement fallback logic here
    
except Exception as e:
    print(f"Request failed: {e}")
```

### 2. Redis Operations with Circuit Breaker

```python
try:
    # Protected Redis operation
    result = await pool_manager.execute_redis_command(
        'set', 'my_key', 'my_value'
    )
    print(f"Redis SET successful: {result}")
    
except CircuitBreakerError:
    print("Redis unavailable, using fallback")
    # Use in-memory cache or other fallback
```

### 3. Database Queries with Circuit Breaker

```python
try:
    # Protected database query
    result = await pool_manager.execute_db_query(
        "SELECT * FROM users WHERE id = $1",
        user_id,
        fetch_one=True
    )
    print(f"User data: {result}")
    
except CircuitBreakerError:
    print("Database unavailable")
    # Return cached data or error response
```

### 4. Using the Decorator Pattern

```python
from app.core.circuit_breaker import with_circuit_breaker

@with_circuit_breaker("my_service", failure_threshold=3, recovery_timeout=20)
async def call_external_service():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()

# Usage
try:
    data = await call_external_service()
except CircuitBreakerError:
    # Handle circuit open scenario
    pass
```

## API Endpoints

### Get Circuit Breaker Status
```bash
GET /api/v1/circuit-breaker/status

# Response
{
  "summary": {
    "healthy_circuits": 4,
    "degraded_circuits": 0,
    "failed_circuits": 1
  },
  "details": {
    "global": {
      "total_circuits": 5,
      "open_circuits": 1,
      "half_open_circuits": 0,
      "closed_circuits": 4
    },
    "breakers": {
      "ollama": {
        "name": "ollama",
        "state": "open",
        "metrics": {
          "total_calls": 100,
          "successful_calls": 80,
          "failed_calls": 20,
          "success_rate": "80.0%",
          "consecutive_failures": 5
        }
      }
    }
  }
}
```

### Get Service-Specific Status
```bash
GET /api/v1/circuit-breaker/status/ollama
```

### Get Detailed Metrics
```bash
GET /api/v1/circuit-breaker/metrics
```

### Reset Circuit Breaker
```bash
POST /api/v1/circuit-breaker/reset/ollama
POST /api/v1/circuit-breaker/reset-all
```

## Configuration

Circuit breakers are configured in `connection_pool.py`:

```python
self._breaker_manager.get_or_create(
    'service_name',
    failure_threshold=5,      # Failures before opening
    recovery_timeout=30.0,     # Seconds before recovery attempt
    success_threshold=1,       # Successes to close from half-open
    expected_exception=(...)   # Exceptions to count as failures
)
```

## Monitoring

### Health Check Integration
The `/health` endpoint now includes circuit breaker status:

```bash
curl http://localhost:10010/health | jq .circuit_breakers
```

### Prometheus Metrics (Future Enhancement)
Circuit breaker metrics can be exposed for Prometheus:
- `circuit_breaker_state` - Current state (0=closed, 1=half-open, 2=open)
- `circuit_breaker_calls_total` - Total calls per service
- `circuit_breaker_failures_total` - Total failures per service
- `circuit_breaker_trips_total` - Number of times circuit opened

### Logging
All state changes and circuit trips are logged:
```
INFO - Circuit breaker 'ollama' transitioned from CLOSED to OPEN after 5 consecutive failures
WARNING - Circuit breaker OPEN for service 'ollama', request to /api/generate blocked
INFO - Circuit breaker 'ollama' transitioned from OPEN to HALF_OPEN
INFO - Circuit breaker 'ollama' transitioned from HALF_OPEN to CLOSED after 1 successful call(s)
```

## Testing

### Run Unit Tests
```bash
cd /opt/sutazaiapp/backend
python -m pytest tests/test_circuit_breaker.py -v
```

### Run Integration Tests
```bash
# Test circuit breaker endpoints
python test_circuit_breaker.py

# Run demonstration
python demo_circuit_breaker.py
```

### Simulate Failures
To test circuit breaker behavior:

1. **Stop a service** (e.g., Redis):
```bash
docker stop sutazai-redis
```

2. **Make requests** and observe circuit opening:
```bash
curl http://localhost:10010/api/v1/circuit-breaker/status
```

3. **Restart service**:
```bash
docker start sutazai-redis
```

4. **Wait 30 seconds** and observe recovery

## Best Practices

### 1. Appropriate Timeouts
- Set recovery timeout based on service characteristics
- Faster recovery for caches (Redis)
- Slower recovery for external APIs

### 2. Failure Thresholds
- Lower threshold (3) for critical services
- Higher threshold (5-10) for services with occasional hiccups

### 3. Fallback Strategies
Always implement fallback logic:
```python
try:
    result = await get_from_service()
except CircuitBreakerError:
    result = get_from_cache()  # or return default
```

### 4. Monitoring and Alerting
- Set up alerts for circuit breaker trips
- Monitor success rates and response times
- Track circuit breaker state changes

### 5. Testing
- Regular chaos engineering tests
- Verify fallback mechanisms work
- Test recovery scenarios

## Troubleshooting

### Circuit Stuck Open
- Check if service is actually healthy
- Manually reset: `POST /api/v1/circuit-breaker/reset/service_name`
- Review logs for underlying errors

### Too Many False Positives
- Increase failure threshold
- Adjust timeout settings
- Check for network issues

### Recovery Too Slow
- Decrease recovery timeout
- Implement health check endpoints
- Use more aggressive retry strategies

## Future Enhancements

1. **Dynamic configuration** - Adjust thresholds based on load
2. **Sliding window** - Use time-based windows instead of consecutive failures
3. **Bulkhead pattern** - Isolate thread pools per service
4. **Metrics export** - Prometheus/Grafana integration
5. **Distributed state** - Share circuit state across instances
6. **Adaptive recovery** - Adjust recovery time based on failure patterns

## Support

For issues or questions about the circuit breaker implementation:
1. Check logs in `/var/log/sutazai/circuit_breaker.log`
2. Review metrics at `/api/v1/circuit-breaker/metrics`
3. Consult the test suite for usage examples
4. File issues in the project repository