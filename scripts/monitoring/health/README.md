# SutazAI Health Check System

**CONSOLIDATED FROM 49+ HEALTH CHECK SCRIPTS**

This directory contains the canonical health checking system for SutazAI, consolidating 49+ individual health check scripts into a organized, maintainable structure.

## Consolidated Scripts

### Original Scripts (Now Consolidated)
```
✅ CONSOLIDATED FROM:
- scripts/pre-commit/validate_system_health.py
- scripts/deployment/check_services_health.py
- scripts/deployment/health_check_gateway.py
- scripts/deployment/health_check_ollama.py
- scripts/deployment/health_check_dataservices.py
- scripts/deployment/infrastructure_health_check.py
- scripts/deployment/health-check-server.py
- scripts/deployment/health_check_monitoring.py
- scripts/deployment/health_check_vectordb.py
- scripts/master/health-master.py
- scripts/monitoring/container-health-monitor.py
- scripts/monitoring/permanent-health-monitor.py
- scripts/monitoring/distributed-health-monitor.py
- scripts/monitoring/system-health-validator.py
- scripts/monitoring/validate-production-health.py
- scripts/monitoring/database_health_check.py
- scripts/monitoring/fix-agent-health-checks.py
- scripts/monitoring/comprehensive-agent-health-monitor.py
- scripts/utils/health_monitor.py
- frontend/agent_health_dashboard.py
- Various Docker health check scripts
- Backend and agent health monitoring modules
- Test health validation scripts
```

## New Canonical Structure

### 1. Master Health Controller
**File:** `master-health-controller.py`
**Purpose:** Single source of truth for all health checking
**Usage:**
```bash
python master-health-controller.py                    # Check all services
python master-health-controller.py --service backend  # Check specific service
python master-health-controller.py --critical-only    # Check only critical services
python master-health-controller.py --monitor          # Continuous monitoring
python master-health-controller.py --report health.txt # Generate report
```

**Features:**
- ✅ Comprehensive service health checking
- ✅ Critical vs non-critical service classification  
- ✅ Continuous monitoring mode
- ✅ Detailed health reporting with JSON output
- ✅ Integration with existing SutazAI infrastructure
- ✅ Automatic retry logic with exponential backoff

### 2. Deployment Health Checker
**File:** `deployment-health-checker.py`
**Purpose:** Specialized health validation for deployment scenarios
**Usage:**
```bash
python deployment-health-checker.py  # Full deployment validation
```

**Features:**
- ✅ Database connectivity validation with actual queries
- ✅ AI model availability checking
- ✅ Service mesh connectivity verification
- ✅ Monitoring stack validation
- ✅ Resource availability assessment
- ✅ Deployment readiness assessment

### 3. Container Health Monitor
**File:** `container-health-monitor.py`
**Purpose:** Real-time Docker container monitoring with auto-healing
**Usage:**
```bash
python container-health-monitor.py                    # Start monitoring
python container-health-monitor.py --interval 60     # Custom interval
python container-health-monitor.py --no-auto-heal    # Disable auto-healing
```

**Features:**
- ✅ Real-time container health monitoring
- ✅ Automatic restart of unhealthy containers
- ✅ Resource usage monitoring (CPU, memory, network)
- ✅ Intelligent failure tracking
- ✅ Critical container identification
- ✅ Auto-healing with restart throttling

### 4. Pre-commit Health Validator
**File:** `pre-commit-health-validator.py`
**Purpose:** Fast system validation before commits
**Usage:**
```bash
python pre-commit-health-validator.py           # Basic validation
python pre-commit-health-validator.py --strict  # Block commit on issues
python pre-commit-health-validator.py --quiet   #   output
```

**Features:**
- ✅ Fast validation optimized for pre-commit hooks
- ✅ Critical service checking with   overhead
- ✅ Docker container status validation
- ✅ Configurable strictness levels

### 5. Monitoring Health Aggregator  
**File:** `monitoring-health-aggregator.py`
**Purpose:** Advanced monitoring with metrics collection and alerting
**Usage:**
```bash
python monitoring-health-aggregator.py                    # Single check
python monitoring-health-aggregator.py --continuous      # Continuous monitoring
python monitoring-health-aggregator.py --output report.txt # Save report
```

**Features:**
- ✅ Comprehensive metrics collection (system, application, database, Docker)
- ✅ Alert condition checking with configurable thresholds
- ✅ Historical data tracking
- ✅ Performance monitoring with response time tracking
- ✅ Resource utilization monitoring

## Service Categories

### Critical Services (Must be healthy for system operation)
- **Backend FastAPI** (port 10010) - Core API with database connectivity
- **Frontend Streamlit** (port 10011) - User interface
- **PostgreSQL** (port 10000) - Primary database  
- **Redis** (port 10001) - Caching layer
- **Ollama** (port 10104) - AI model server with TinyLlama

### Semi-Critical Services (Important but system can degrade gracefully)
- **Vector Databases** - Qdrant, ChromaDB, FAISS
- **Service Mesh** - RabbitMQ messaging
- **Neo4j** - Graph database

### Non-Critical Services (Optional but enhance functionality)
- **Agent Services** - Hardware optimizer, AI orchestrator, etc.
- **Monitoring Stack** - Prometheus, Grafana, Loki
- **Service Discovery** - Consul, Kong

## Integration Points

### Makefile Targets
```bash
make health          # Run master health controller
make health-deploy   # Run deployment validation
make health-monitor  # Start container monitoring
make health-quick    # Quick pre-commit validation
```

### Docker Compose Integration
Health checks are integrated with Docker Compose healthcheck directives.

### CI/CD Pipeline
Pre-commit hooks use the fast health validator to ensure system stability.

### Monitoring Integration
Metrics are exported to Prometheus and can be visualized in Grafana dashboards.

## Migration Notes

### Backward Compatibility
Symlinks are created for critical legacy scripts to maintain backward compatibility:
```bash
# Legacy script paths still work via symlinks
scripts/deployment/check_services_health.py -> ../health/deployment-health-checker.py
scripts/monitoring/system-health-validator.py -> ../health/master-health-controller.py
scripts/pre-commit/validate_system_health.py -> ../health/pre-commit-health-validator.py
```

### Configuration Migration
Old configuration files and environment variables are automatically detected and migrated.

## Performance Optimization

### Parallel Execution
Health checks run in parallel using ThreadPoolExecutor for faster execution.

### Caching
Service discovery results are cached to reduce overhead.

### Timeout Management
Configurable timeouts prevent hanging on unresponsive services.

### Resource Efficiency
  memory footprint with intelligent data retention.

## Alerting and Notifications

### Alert Levels
- **Critical:** Service down, database connectivity lost
- **Warning:** High resource usage, slow response times
- **Info:** Service degraded but functional

### Alert Channels
- Log files with structured formatting
- JSON output for integration with external systems
- Console output with color coding and icons
- File-based reports for audit trails

## Troubleshooting

### Common Issues

**Service Not Responding:**
```bash
# Check if container is running
docker ps | grep sutazai

# Check service logs
docker logs sutazai-<service>

# Manual health check
curl http://localhost:<port>/health
```

**High Resource Usage:**
```bash
# Check system resources
python monitoring-health-aggregator.py

# Monitor specific container
docker stats sutazai-<service>
```

**Database Connectivity:**
```bash
# Test database connections
python deployment-health-checker.py
```

### Debug Mode
All health check scripts support verbose logging:
```bash
export DEBUG=true
python <health-script>.py
```

## Future Enhancements

### Planned Features
- Integration with external monitoring systems (DataDog, New Relic)
- Advanced anomaly detection using machine learning
- Predictive health monitoring
- Integration with cloud provider health checks
- Mobile alerts via webhook integration

### Extensibility
The health check system is designed for easy extension:
- Add new service definitions in the master controller
- Create custom health check functions
- Implement new alert channels
- Add custom metrics collectors

## Documentation Standards

All health check scripts follow the same documentation standards:
- Clear docstrings with purpose and usage examples
- Comprehensive argument parsing with help text
- Structured logging with appropriate levels
- Error handling with meaningful messages
- Return code standards (0=success, 1=warning, 2=critical)

---

**Created:** 2025-08-10  
**Author:** ULTRA SCRIPT CONSOLIDATION MASTER  
**Consolidated:** 49 health check scripts → 5 canonical scripts  
**Follows:** CLAUDE.md Rules 4, 7, 19