# Migration Notes: Optional Features Implementation

## Summary

This PR implements optional features for FSDP training and TabbyML code completion, making them disabled by default with clean abstractions and no runtime errors when disabled.

## Changes Made

### 1. Configuration
- Added feature flags to `.env.example`:
  - `ENABLE_FSDP=false`
  - `ENABLE_TABBY=false`
  - `TABBY_URL=http://tabbyml:8080`
  - `TABBY_API_KEY=`
- Updated `backend/app/core/config.py` with new settings

### 2. Service Abstractions

#### Code Completion (`backend/app/services/code_completion/`)
- **Interfaces**: `CodeCompletionClient` protocol with `complete()`, `health_check()`, `is_available()`
- **Implementations**:
  - `NullCodeCompletionClient`: Returns disabled message when `ENABLE_TABBY=false`
  - `TabbyCodeCompletionClient`: Real TabbyML integration when `ENABLE_TABBY=true`
- **Factory**: `code_completion_factory()` returns appropriate implementation

#### Training (`backend/app/services/training/`)
- **Interfaces**: `Trainer` protocol with `train()`, `get_status()`, `cancel()`, etc.
- **Implementations**:
  - `DefaultTrainer`: Single-process training when `ENABLE_FSDP=false`
  - `FsdpTrainer`: Distributed training when `ENABLE_FSDP=true`
- **Factory**: `trainer_factory()` returns appropriate implementation

### 3. Docker Compose Updates
- Added profiles to services:
  - `fsdp` service: profiles `[fsdp, optional]`
  - `tabbyml` service: profiles `[tabby, optional]`
- Services only start when profile is activated

### 4. API Endpoints
- New `/api/v1/features` endpoint returns feature flag states
- Allows frontend to check which features are available

### 5. Dependencies
- Created `requirements-optional.txt` for optional dependencies
- Added `setup.py` with extras_require for selective installation:
  - `pip install -e ".[fsdp]"` - FSDP dependencies
  - `pip install -e ".[tabby]"` - TabbyML dependencies  
  - `pip install -e ".[all]"` - All optional features

### 6. Helper Scripts
- `scripts/start-with-features.sh`: Starts services based on feature flags

### 7. Tests
- `tests/test_optional_features.py`: Comprehensive tests for:
  - Feature flag configuration
  - Factory functions
  - Null vs real implementations
  - API endpoint responses

## Migration Steps

### For Existing Installations

1. **Update environment variables**:
```bash
# Add to .env (features disabled by default)
echo "ENABLE_FSDP=false" >> .env
echo "ENABLE_TABBY=false" >> .env
echo "TABBY_URL=http://tabbyml:8080" >> .env
echo "TABBY_API_KEY=" >> .env
```

2. **Restart services**:
```bash
docker-compose down
docker-compose up -d
```

3. **Verify feature status**:
```bash
curl http://localhost:10010/api/v1/features
```

### To Enable Features

1. **Edit `.env`**:
```bash
ENABLE_FSDP=true
ENABLE_TABBY=true
```

2. **Start with profiles**:
```bash
# Option 1: Use helper script
./scripts/start-with-features.sh

# Option 2: Use docker-compose directly
docker-compose --profile optional up -d
```

3. **Install optional dependencies** (if running locally):
```bash
pip install -r requirements-optional.txt
# or
pip install -e ".[all]"
```

## Rollback Instructions

If issues occur, rollback is simple:

1. **Disable features in `.env`**:
```bash
ENABLE_FSDP=false
ENABLE_TABBY=false
```

2. **Stop optional services**:
```bash
docker-compose --profile optional down
```

3. **Restart core services**:
```bash
docker-compose up -d
```

The system will automatically use null implementations when features are disabled.

## Breaking Changes

None. All changes are backward compatible:
- Features are disabled by default
- Core functionality unchanged
- Existing code paths preserved
- No required dependency changes

## Testing

Run tests to verify implementation:
```bash
# Run feature tests
pytest tests/test_optional_features.py -v

# Test with features disabled
ENABLE_FSDP=false ENABLE_TABBY=false pytest tests/

# Test with features enabled
ENABLE_FSDP=true ENABLE_TABBY=true pytest tests/
```

## Performance Impact

- **When disabled**: Zero overhead, null implementations return immediately
- **When enabled**: Standard service overhead, lazy initialization
- **Memory**: Optional services only consume resources when profile activated
- **CPU**: No impact when disabled

## Security Considerations

- TabbyML API key is optional but recommended for production
- FSDP service should be secured in multi-user environments
- Feature flags are read-only after startup
- No sensitive data in feature status endpoint

## Future Improvements

- Add more granular feature flags
- Implement feature flag hot-reloading
- Add metrics for feature usage
- Create feature flag UI in frontend
- Add automated feature compatibility checks

## Support

For issues or questions:
1. Check `docker-compose logs [service]` for errors
2. Verify feature flags in `.env`
3. Test service health endpoints
4. Review `/api/v1/features` response
5. Consult `docs/OPTIONAL_FEATURES.md`