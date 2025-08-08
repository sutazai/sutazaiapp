# Ollama Integration Agent Documentation

## Overview

The Ollama Integration Agent provides a robust interface for integrating the local TinyLlama model (via Ollama) into agent workflows. It handles request validation, error handling, retry logic, and response parsing with comprehensive monitoring and logging.

## Architecture

### Components

1. **Pydantic Schemas** (`/schemas/ollama_schemas.py`)
   - `OllamaGenerateRequest`: Validates generation requests
   - `OllamaGenerateResponse`: Parses and validates responses
   - `OllamaErrorResponse`: Structured error tracking
   - `OllamaModelsResponse`: Model listing validation

2. **Integration Agent** (`/agents/ollama_integration/app.py`)
   - Async HTTP client with connection pooling
   - Exponential backoff retry logic
   - Request hashing for tracking
   - Structured logging with trace IDs

3. **FastAPI Service**
   - Health endpoint with model verification
   - Generation endpoint with full validation
   - Models listing endpoint
   - Automatic startup/shutdown lifecycle

## Request Schema

### Generation Request

```json
{
  "model": "tinyllama:latest",
  "prompt": "Your prompt here",
  "temperature": 0.7,
  "num_predict": 128,
  "top_p": 0.9,
  "top_k": 40,
  "stop": ["\\n", "END"],
  "seed": 42
}
```

### Field Validation

| Field | Type | Range | Default | Description |
|-------|------|-------|---------|-------------|
| model | string | - | tinyllama:latest | Model to use |
| prompt | string | 1-32768 chars | required | Input text |
| temperature | float | 0.0-2.0 | 0.7 | Sampling temperature |
| num_predict | int | 1-2048 | 128 | Max tokens to generate |
| top_p | float | 0.0-1.0 | 0.9 | Nucleus sampling |
| top_k | int | 1-100 | 40 | Top-k sampling |
| stop | list[str] | max 5, 50 chars each | null | Stop sequences |
| seed | int | any | null | Random seed |

## Response Format

### Successful Response

```json
{
  "response": "Generated text content",
  "tokens": 42,
  "latency": 125.5,
  "tokens_per_second": 33.6
}
```

### Error Response

```json
{
  "error": "Model not found",
  "code": 404,
  "timestamp": "2024-12-19T10:30:00Z",
  "request_hash": "a1b2c3d4"
}
```

## Retry Logic

The agent implements exponential backoff with jitter for transient failures:

### Retry Configuration

- **Max Retries**: 3 attempts
- **Base Delay**: 2.0 seconds
- **Backoff Formula**: `delay = base ^ attempt + jitter`
- **Max Delay**: 30 seconds
- **Jitter**: 0-10% of calculated delay

### Retry Behavior by Status Code

| Status Code | Action | Retry? |
|------------|--------|--------|
| 200 | Success | No |
| 400 | Log bad request | No |
| 404 | Model not found | No |
| 429 | Rate limited | Yes (with Retry-After) |
| 500+ | Server error | Yes |
| Timeout | Connection timeout | Yes |
| Network Error | Connection failed | Yes |

### Backoff Intervals

| Attempt | Base Delay | Jitter Range | Total Range |
|---------|------------|--------------|-------------|
| 1 | 2.0s | 0-0.2s | 2.0-2.2s |
| 2 | 4.0s | 0-0.4s | 4.0-4.4s |
| 3 | 8.0s | 0-0.8s | 8.0-8.8s |

## Logging

All operations are logged with structured format including:

### Log Format

```
2024-12-19 10:30:00 - ollama_integration - INFO - [hash=a1b2c3d4] Generating text prompt_length=50 max_tokens=128
```

### Log Levels

- **INFO**: Successful operations, model verification
- **WARNING**: Retries, rate limiting, model not found
- **ERROR**: Final failures, validation errors, network issues

### Request Tracking

Each request is assigned a 16-character hash for correlation:

```python
request_hash = hashlib.sha256(json.dumps(payload)).hexdigest()[:16]
```

## Performance Metrics

### Expected Latency Ranges

| Operation | P50 | P95 | P99 |
|-----------|-----|-----|-----|
| Model List | 10ms | 50ms | 100ms |
| Generation (50 tokens) | 500ms | 1500ms | 3000ms |
| Generation (500 tokens) | 2000ms | 5000ms | 8000ms |

### Token Generation Rates

- **TinyLlama on CPU**: 20-50 tokens/second
- **TinyLlama on GPU**: 100-200 tokens/second

## API Endpoints

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "ollama_reachable": true,
  "tinyllama_available": true,
  "models_count": 1
}
```

### Generate Text

```bash
POST /generate
Content-Type: application/json

{
  "prompt": "What is Docker?",
  "temperature": 0.7,
  "num_predict": 100
}
```

### List Models

```bash
GET /models
```

Response:
```json
{
  "models": [
    {
      "name": "tinyllama:latest",
      "size_mb": 637.5
    }
  ]
}
```

## Usage Examples

### Basic Generation

```python
async with OllamaIntegrationAgent() as agent:
    result = await agent.generate(
        prompt="Explain microservices architecture",
        temperature=0.7,
        max_tokens=200
    )
    print(f"Response: {result['response']}")
    print(f"Tokens: {result['tokens']}")
```

### With Stop Sequences

```python
result = await agent.generate(
    prompt="List three benefits:\n1.",
    temperature=0.5,
    max_tokens=100,
    stop=["4.", "\n\n"]
)
```

### Concurrent Requests

```python
prompts = ["What is AI?", "What is ML?", "What is DL?"]
tasks = [agent.generate(p, max_tokens=50) for p in prompts]
results = await asyncio.gather(*tasks)
```

## Error Handling

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Model not found | TinyLlama not pulled | Run `ollama pull tinyllama` |
| Connection refused | Ollama not running | Start Ollama service |
| Timeout | Slow generation | Increase timeout or reduce max_tokens |
| Validation error | Invalid parameters | Check parameter ranges |
| Rate limited | Too many requests | Implement request throttling |

### Error Recovery Example

```python
try:
    result = await agent.generate(prompt="Test")
except ValidationError as e:
    logger.error(f"Invalid request: {e}")
    # Fix parameters and retry
except Exception as e:
    logger.error(f"Generation failed: {e}")
    # Fall back to default response
```

## Testing

### Unit Tests

```bash
pytest tests/test_ollama_integration.py -v
```

### Integration Tests

The test suite covers:
- Model verification
- Basic generation
- Empty responses
- Long outputs (>500 tokens)
- Stop sequences
- Concurrent requests (5 parallel)
- Retry behavior
- Validation errors
- Request hashing
- Performance metrics

### CI/CD Integration

The CI pipeline:
1. Starts Ollama container
2. Pulls TinyLlama model
3. Verifies model availability
4. Runs integration tests
5. Tests CI prompt with assertion
6. Performs performance check

## Configuration

### Environment Variables

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=tinyllama:latest
OLLAMA_TIMEOUT=30
OLLAMA_MAX_RETRIES=3
OLLAMA_BACKOFF_BASE=2.0
```

### Docker Integration

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 5s
      retries: 3
```

## Monitoring

### Prometheus Metrics

```python
# Metrics exposed
ollama_request_total{status="success|failure"}
ollama_request_duration_seconds{quantile="0.5|0.95|0.99"}
ollama_tokens_generated_total
ollama_retry_count_total
```

### Grafana Dashboard

Key panels:
- Request rate and success ratio
- P95 latency trends
- Token generation rate
- Error rate by type
- Retry frequency

## Best Practices

1. **Model Management**
   - Always verify model availability before generation
   - Cache model list for 5 minutes
   - Use model-specific parameters

2. **Request Optimization**
   - Batch similar requests when possible
   - Use appropriate max_tokens for use case
   - Implement client-side caching for repeated prompts

3. **Error Handling**
   - Log all errors with request hash
   - Implement circuit breaker for repeated failures
   - Provide fallback responses for critical paths

4. **Performance**
   - Use streaming for long generations
   - Implement request pooling
   - Monitor and alert on P95 latency

## Troubleshooting

### Debug Mode

Enable detailed logging:
```python
logging.getLogger("ollama_integration").setLevel(logging.DEBUG)
```

### Common Issues

1. **Slow Generation**
   - Check CPU/GPU utilization
   - Reduce max_tokens
   - Use lower temperature

2. **Memory Issues**
   - Monitor Ollama container memory
   - Implement request queuing
   - Use smaller models if needed

3. **Network Errors**
   - Check Docker network configuration
   - Verify firewall rules
   - Test with curl directly

## Future Enhancements

- [ ] Streaming response support
- [ ] Model warm-up on startup
- [ ] Request queuing with priority
- [ ] Caching layer for repeated prompts
- [ ] Multi-model support
- [ ] Conversation context management
- [ ] Token usage tracking and limits
- [ ] A/B testing framework