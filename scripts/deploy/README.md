# Deploy Scripts

Deployment and startup scripts for SutazaiApp infrastructure.

## Scripts in this directory:

### start-infrastructure.sh
Starts all Docker services in the correct dependency order.
```bash
./start-infrastructure.sh
```

### stop-infrastructure.sh
Stops all Docker services gracefully.
```bash
./stop-infrastructure.sh
```

### configure-kong-routes.sh
Configures Kong API Gateway routes for service routing.
```bash
./configure-kong-routes.sh
```

### register-consul-services.sh
Registers services with Consul for service discovery.
```bash
./register-consul-services.sh
```

## Usage

For complete system deployment:
```bash
# Start everything
./start-infrastructure.sh

# Configure routing
./configure-kong-routes.sh

# Register services
./register-consul-services.sh
```

## Dependencies

- Docker and Docker Compose installed
- Proper .env file configured
- Network connectivity to pull Docker images
- Sufficient system resources (32GB RAM recommended)