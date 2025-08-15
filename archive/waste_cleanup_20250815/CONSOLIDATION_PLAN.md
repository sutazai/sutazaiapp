# DEPLOYMENT SCRIPT CONSOLIDATION PLAN

## Current State
- `deploy.sh` (root): 1023 lines - Main deployment script with hardware optimization
- `scripts/deploy.sh`: 521 lines - Older, simpler version
- `scripts/deployment_manager.sh`: 900 lines - Service tier management utilities

## Investigation Findings

### Unique Functions in deployment_manager.sh:
1. `build_required_images()` - Docker image building logic
2. `start_service_tier()` - Tiered service startup
3. `run_tier_health_checks()` - Tier-specific health validation
4. `setup_ollama_model()` - Ollama model initialization
5. `deploy_tier()` - Tier-based deployment
6. `show_access_urls()` - Display service URLs
7. `show_service_status()` - Service status display
8. `show_service_logs()` - Log viewing utilities

### Functions in root deploy.sh:
- Hardware detection and optimization (unique)
- Environment setup and validation
- Infrastructure deployment
- Rollback capabilities

## Consolidation Decision

**MERGE REQUIRED**: The deployment_manager.sh contains unique tier-based deployment logic that would be valuable in the main deploy.sh script.

**ACTION PLAN:**
1. Extract unique functions from deployment_manager.sh
2. Integrate them into root deploy.sh
3. Archive scripts/deploy.sh (older version)
4. Archive scripts/deployment_manager.sh after merge

**RISK ASSESSMENT:** 
- Medium risk - requires careful merging to preserve functionality
- Recommendation: Keep archives for 30 days before permanent deletion

## Files to Remove After Consolidation:
1. scripts/deploy.sh - Older, less complete version
2. scripts/deployment_manager.sh - After merging unique functions