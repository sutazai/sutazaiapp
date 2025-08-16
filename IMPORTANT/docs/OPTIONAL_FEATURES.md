# Optional Features Configuration

## Overview

SutazAI now supports optional features that can be enabled or disabled based on your needs. By default, all optional features are **disabled** to ensure a lightweight base installation.

## Supported Optional Features

### 1. FSDP (Fully Sharded Data Parallel) Training
- **Purpose**: Distributed training for large language models
- **Flag**: `ENABLE_FSDP`
- **Default**: `false`
- **Service**: `sutazai-fsdp` on port 8596

### 2. TabbyML Code Completion
- **Purpose**: AI-powered code completion service  
- **Flag**: `ENABLE_TABBY`
- **Default**: `false`
- **Service**: `sutazai-tabbyml` on port 10303
- **Configuration**:
  - `TABBY_URL`: TabbyML service URL (default: `http://tabbyml:8080`)
  - `TABBY_API_KEY`: Optional API key for authentication

## Configuration

### Method 1: Environment Variables (.env file)

Add to your `.env` file:

```bash
# Optional Features
ENABLE_FSDP=false
ENABLE_TABBY=false
TABBY_URL=http://tabbyml:8080
TABBY_API_KEY=
```

### Method 2: Command Line

```bash
# Enable FSDP only
ENABLE_FSDP=true docker-compose --profile fsdp up -d

# Enable TabbyML only  
ENABLE_TABBY=true docker-compose --profile tabby up -d

# Enable both
ENABLE_FSDP=true ENABLE_TABBY=true docker-compose --profile optional up -d
```

### Method 3: Using the Helper Script

```bash
# Edit .env to set flags, then:
./scripts/start-with-features.sh
```

## API Integration

### Checking Feature Status

Query the features endpoint to see which features are enabled:

```bash
curl http://localhost:10010/api/v1/features
```

Response:
```json
{
  "fsdp": {
    "enabled": false,
    "description": "Fully Sharded Data Parallel training support"
  },
  "tabby": {
    "enabled": false,
    "url": null,
    "description": "TabbyML code completion service"
  },
  "gpu": {
    "enabled": false,
    "description": "GPU acceleration support"
  },
  "monitoring": {
    "enabled": true,
    "description": "System monitoring and metrics collection"
  }
}
```

### Code Completion Service

When `ENABLE_TABBY=false`:
```python
# Returns placeholder message
response = await completion_client.complete(request)
# response.completion = "# Code completion is disabled..."
```

When `ENABLE_TABBY=true`:
```python
# Returns actual TabbyML completions
response = await completion_client.complete(request)
# response.completion = "def calculate_sum(a, b):\n    return a + b"
```

### Training Service

When `ENABLE_FSDP=false`:
```python
# Uses default single-process trainer
trainer = trainer_factory(settings)
result = await trainer.train(config)
# Runs local training or Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test training
```

When `ENABLE_FSDP=true`:
```python
# Uses FSDP distributed trainer
trainer = trainer_factory(settings)
result = await trainer.train(config)
# Submits job to FSDP service for distributed training
```

## Installing Optional Dependencies

### Base Installation (no optional features)
```bash
pip install -r requirements.txt
```

### With FSDP Support
```bash
pip install -e ".[fsdp]"
# or
pip install -r requirements-optional.txt
```

### With All Optional Features
```bash
pip install -e ".[all]"
```

## Docker Compose Profiles

The services are organized into profiles:

- **No profile**: Core services (always run)
- **`fsdp` profile**: FSDP training service
- **`tabby` profile**: TabbyML code completion
- **`optional` profile**: All optional services

### Starting Services by Profile

```bash
# Core services only (default)
docker-compose up -d

# Core + FSDP
docker-compose --profile fsdp up -d

# Core + TabbyML
docker-compose --profile tabby up -d

# Core + All optional
docker-compose --profile optional up -d
```

## Migration Guide

### Upgrading from Previous Version

1. **Update your `.env` file**:
   ```bash
   echo "ENABLE_FSDP=false" >> .env
   echo "ENABLE_TABBY=false" >> .env
   ```

2. **Update docker-compose.yml**:
   - Already done in this PR

3. **Restart services**:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

### Enabling Features Post-Installation

1. **Edit `.env`**:
   ```bash
   ENABLE_FSDP=true
   ENABLE_TABBY=true
   ```

2. **Start optional services**:
   ```bash
   ./scripts/start-with-features.sh
   ```

3. **Verify services are running**:
   ```bash
   docker-compose ps
   curl http://localhost:10010/api/v1/features
   ```

## Rollback Instructions

If you need to rollback to the previous version:

1. **Disable all optional features**:
   ```bash
   ENABLE_FSDP=false
   ENABLE_TABBY=false
   ```

2. **Stop optional services**:
   ```bash
   docker-compose --profile optional down
   ```

3. **Remove feature flag configuration** (optional):
   - Remove the new lines from `.env`
   - The system will use defaults (disabled)

## Troubleshooting

### Service Won't Start

Check if the feature is enabled:
```bash
grep ENABLE_ .env
docker-compose --profile optional ps
```

### Import Errors

Install optional dependencies:
```bash
# For FSDP
pip install torch transformers accelerate

# For TabbyML (client only)
# httpx is already included in base requirements
```

### Service Health Checks

```bash
# FSDP health
curl http://localhost:8596/health

# TabbyML health  
curl http://localhost:10303/health

# Backend features status
curl http://localhost:10010/api/v1/features
```

## CI/CD Integration

### GitHub Actions Feature Matrix

The project includes automated testing for all feature combinations:

```yaml
# .github/workflows/feature-matrix.yml
matrix:
  include:
    - name: " "
      enable_fsdp: "false"
      enable_tabby: "false"
    - name: "FSDP Only"
      enable_fsdp: "true"
      enable_tabby: "false"
    - name: "TabbyML Only"
      enable_fsdp: "false"
      enable_tabby: "true"
    - name: "Full Features"
      enable_fsdp: "true"
      enable_tabby: "true"
```

## Performance Impact

| Configuration | Memory Usage | Startup Time | CPU Impact |
|--------------|-------------|--------------|------------|
|   | ~2GB | 30s | Low |
| +FSDP | +1GB | +10s | Medium |
| +TabbyML | +2GB | +20s | Medium |
| Full Features | ~5GB | 60s | High |

## Best Practices

1. **Start with features disabled** - Enable only what you need
2. **Test in development first** - Verify features work before production
3. **Monitor resource usage** - Optional features may increase resource consumption
4. **Use profiles for deployment** - Simplifies environment-specific configurations
5. **Document your configuration** - Keep track of which features are enabled per environment
6. **Use setup.py for dependencies** - Install only required optional dependencies
7. **Test feature combinations** - Ensure features work together as expected