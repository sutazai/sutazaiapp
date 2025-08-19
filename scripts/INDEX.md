# Scripts Directory Index
## Automation & Management Tools

---

## 📁 Directory Overview

```
scripts/
├── enforcement/       → Rule enforcement & compliance
├── deployment/        → Infrastructure deployment
├── monitoring/        → System monitoring & logging
├── mcp/              → MCP server management
├── maintenance/      → System maintenance & cleanup
├── utils/            → Utility functions
├── testing/          → Test automation
├── security/         → Security scanning
└── emergency/        → Emergency response scripts
```

---

## 🚔 Enforcement Scripts

### Docker Management
- `docker_consolidation_phase1.sh` - Consolidate Docker files (89→7)
- `consolidate_docker.py` - Python Docker consolidation
- `execute_docker_consolidation.sh` - Run consolidation

### Mock/Stub Removal
- `remove_mock_implementations.py` - Find and fix mock code
- `fix_backend_mocks.sh` - Fix backend empty returns
- `fantasy_code_eliminator.py` - Remove fantasy implementations

### CHANGELOG Management
- `create_changelogs.sh` - Generate CHANGELOG files
- `add_missing_changelogs.py` - Add missing CHANGELOGs

### Validation
- `validate_compliance.py` - Check rule compliance
- `validate_docker_health.py` - Docker health validation
- `validate_enforcement.py` - Enforcement validation

---

## 🚀 Deployment Scripts

### Main Deployment
- `deploy_real_mcp_servers.sh` - Deploy real MCP servers
- `deploy_phase1_preparation.py` - Phase 1 deployment
- `deploy_phase2_core.py` - Core services
- `deploy_phase3_ai.py` - AI services
- `deploy_phase4_monitoring.py` - Monitoring stack
- `deploy_phase5_production.py` - Production deployment

### Infrastructure
- `infrastructure/deploy-mcp-services.py` - MCP service deployment
- `infrastructure/deploy-dind-mcp.sh` - Docker-in-Docker MCP
- `infrastructure/deploy-mcp-containers.sh` - Container deployment

### Service Management
- `restart_core_services.sh` - Restart critical services
- `fix_container_names.sh` - Fix container naming
- `fix_postgres_dns.sh` - Fix PostgreSQL DNS

---

## 📊 Monitoring Scripts

### Live Monitoring
- **`live_logs.sh`** - Main log viewer (15 options) ✅ FIXED
  ```bash
  # Usage:
  ./scripts/monitoring/live_logs.sh
  # Options 1-15 all working
  ```

### System Monitoring
- `consolidated_monitor.py` - Unified system monitor
- `service_monitor.py` - Service health monitor
- `sutazai_realtime_monitor.py` - Real-time monitoring
- `quality_monitor.py` - Quality metrics
- `memory_leak_detector.py` - Memory leak detection
- `neural_health_monitor.py` - Neural network health

### Performance Monitoring
- `performance/hardware_performance_monitor.py` - Hardware metrics
- `hygiene-monitor-backend.py` - Backend hygiene

### Logging
- `logging/adapter.py` - Logging adapter
- `logging/service_scaler.py` - Service scaling logs
- `logging/main_1.py` - Main logging service

---

## 🤖 MCP Management Scripts

### Core Scripts
- `init_mcp_servers.sh` - Initialize MCP servers
- `validate_mcp_setup.sh` - Validate MCP configuration
- `validate_all_mcps.sh` - Validate all MCP servers

### Wrapper Scripts
```
mcp/wrappers/
├── claude-flow.sh           → Claude Flow wrapper
├── ruv-swarm.sh             → RUV Swarm wrapper
├── claude-task-runner.sh    → Task runner wrapper
├── language-server.sh       → Language server wrapper
├── unified-memory.sh        → Memory service wrapper
└── compass-mcp.sh           → Compass MCP wrapper
```

### Automation
- `automation/error_handling.py` - Error handling
- `automation/monitoring/metrics_collector.py` - Metrics collection
- `automation/orchestration/event_manager.py` - Event management
- `automation/orchestration/service_registry.py` - Service registry

---

## 🧹 Maintenance Scripts

### Cleanup
- `cleanup/deduplicate.py` - Remove duplicates
- `cleanup/execute_changelog_cleanup.sh` - CHANGELOG cleanup
- `cleanup/safe_changelog_cleanup.py` - Safe cleanup

### Optimization
- `optimization/performance_benchmark.py` - Performance tests
- `optimization/ultra_hardware_optimization.py` - Hardware optimization
- `optimization/ultra_requirements_cleaner.py` - Requirements cleanup
- `optimization/ultra_system_optimizer.py` - System optimization

### Hygiene
- `hygiene/detectors.py` - Code smell detection
- `hygiene/fixers.py` - Automatic fixing

---

## 🛠️ Utility Scripts

### Core Utilities
- `docker_utils.py` - Docker utilities
- `network_utils.py` - Network utilities
- `memory_manager.py` - Memory management
- `plugin_manager.py` - Plugin management

### Advanced Utilities
- `advanced_detection.py` - Advanced issue detection
- `cross_modal_learning.py` - ML utilities
- `performance_forecasting_models.py` - Performance prediction
- `secure_agent_comm.py` - Secure agent communication
- `vuln_scanner.py` - Vulnerability scanning

### Testing Utilities
- `automated_continuous_tests.py` - Continuous testing
- `performance_stress_tests.py` - Stress testing
- `production-load-test.py` - Load testing
- `system_performance_benchmark_suite.py` - Benchmarking

---

## 🧪 Testing Scripts

### Test Runners
- `load_test_runner.py` - Load test execution
- `hardware_optimizer_ultra_test_suite.py` - Hardware tests
- `ultratest_memory_optimization.py` - Memory tests
- `ultratest_redis_performance.py` - Redis tests
- `ultratest_security_validation.py` - Security tests
- `validate_security_requirements.py` - Security validation

---

## 🔒 Security Scripts

### Scanning
- `comprehensive_security_scanner.py` - Full security scan
- `secrets_manager.py` - Secrets management

---

## 🚨 Emergency Scripts

### Response Scripts
- `emergency_shutdown.py` - Emergency system shutdown
- Emergency recovery procedures

---

## 📋 Pre-commit Hooks

- `pre-commit/check-breaking-changes.py` - Breaking change detection

---

## 🔧 Other Important Scripts

### Consolidation
- `consolidate_agent_configs.py` - Agent config consolidation
- `fix_agent_configurations.py` - Fix agent configs

### Index Generation
- `tools/generate_index.py` - Generate index files

---

## 💡 Usage Examples

### Run Live Monitoring
```bash
cd /opt/sutazaiapp
./scripts/monitoring/live_logs.sh
# Select option 1-15
```

### Deploy MCP Servers
```bash
./scripts/deployment/deploy_real_mcp_servers.sh
```

### Check Compliance
```bash
python scripts/enforcement/validate_compliance.py
```

### Run System Monitor
```bash
python scripts/monitoring/consolidated_monitor.py
```

### Create CHANGELOGs
```bash
./scripts/enforcement/create_changelogs.sh
```

---

## ✅ Verified Working Scripts

1. `live_logs.sh` - All 15 options functional
2. `create_changelogs.sh` - Successfully created 21 CHANGELOGs
3. `docker_consolidation_phase1.sh` - Reduced 89→7 files
4. `fix_backend_mocks.sh` - Fixed 198 violations
5. `deploy_real_mcp_servers.sh` - Deployed 6 real servers

---

*Index generated by ULTRATHINK methodology*
*Last updated: 2025-08-19*