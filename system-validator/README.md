# System Validator - SutazAI Distributed System Validation Suite

A comprehensive validation framework for testing distributed system reliability, performance, and integration.

## Purpose

The System Validator provides automated testing and continuous validation for microservice architectures. It validates service health, integration points, performance metrics, and reliability mechanisms in distributed systems.

## Prerequisites

- Docker and Docker Compose (version 3.8+)
- Python 3.9 or higher
- At least 8GB of available RAM
- Linux or macOS (Windows users should use WSL2)

## Setup

1. **Clone the repository** (or create a new directory with these files):
   ```bash
   mkdir system-validator
   cd system-validator
   ```

2. **Create stub directories for services**:
   ```bash
   mkdir -p stubs/{api,frontend,storage,prometheus}
   ```

3. **Create stub files for HTTP services**:
   ```bash
   # API health endpoint
   echo '{"status": "ok", "version": "1.0.0", "uptime": 3600}' > stubs/api/health
   
   # Frontend index
   echo '<html><body>Frontend OK</body></html>' > stubs/frontend/index.html
   
   # Storage health endpoint
   echo '{"status": "healthy"}' > stubs/storage/health
   
   # Prometheus config
   cat > stubs/prometheus/prometheus.yml << EOF
   global:
     scrape_interval: 15s
   scrape_configs:
     - job_name: 'api'
       static_configs:
         - targets: ['api:8000']
   EOF
   ```

4. **Start the environment**:
   ```bash
   docker-compose up -d
   ```

5. **Wait for services to be ready** (about 30 seconds):
   ```bash
   docker-compose ps
   ```

## Running Tests

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Check system status**:
   ```bash
   python system_validator.py status
   ```

3. **Run validation tests**:
   ```bash
   python system_validator.py test
   ```

4. **Investigate system health**:
   ```bash
   python system_validator.py investigate
   ```

5. **Run specific pytest tests**:
   ```bash
   # Run only integration tests
   pytest system_validator.py -m integration -v
   
   # Run reliability tests
   pytest system_validator.py -m reliability -v
   ```

## Test Categories

- **Integration Tests**: Validate service-to-service communication
- **Load Tests**: Test system performance under various load patterns
- **Reliability Tests**: Verify failover mechanisms and monitoring accuracy
- **Chaos Tests**: Test system resilience (requires additional setup)

## Monitoring

Access the monitoring interfaces:
- **API**: http://localhost:8000
- **Frontend**: http://localhost:3000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001 (admin/admin)

## Troubleshooting

1. **Services not starting**: Check Docker logs
   ```bash
   docker-compose logs [service-name]
   ```

2. **Port conflicts**: Ensure ports 3000, 3001, 5432, 6379, 8000, 8003, 9090 are available

3. **Permission issues**: Run with appropriate Docker permissions
   ```bash
   sudo docker-compose up -d  # Linux
   ```

## Next Steps

This is a foundation implementation with stubbed services. For full functionality:

1. **Replace stub services** with actual implementations
2. **Implement real health endpoints** in your services
3. **Add actual metrics** to Prometheus exporters
4. **Configure Grafana dashboards** for visualization
5. **Set up AlertManager** for production alerting
6. **Implement chaos engineering** tests with tools like Chaos Mesh
7. **Add performance benchmarks** specific to your use case

## Architecture

The System Validator follows a modular architecture:
- **SystemValidator**: Core validation engine
- **MicroserviceCoordinationValidator**: Tests inter-service communication
- **ReliabilityMonitoringValidator**: Validates monitoring and alerting
- **ContinuousSystemValidator**: Provides ongoing validation in production

## Contributing

When adding new validators:
1. Create a new class inheriting from the base validator
2. Add pytest markers for test categorization
3. Implement both positive and negative test cases
4. Document expected service behavior

## License

This is a reference implementation. Adapt it to your specific needs and compliance requirements.