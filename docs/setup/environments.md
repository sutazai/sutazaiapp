# Environment Configuration

## Environment Variables

The system uses environment variables for configuration across different deployment environments. All variables are managed through `.env` files.

## Environment Files

### Main Configuration
- **`.env`**: Primary environment configuration
- **`.env.example`**: Template with default values
- **`.env.local`**: Local development overrides (optional)
- **`.env.production`**: Production-specific settings

### Creating Environment Files
```bash
# Copy example to create your configuration
cp .env.example .env

# Edit configuration
nano .env
```

## Core Environment Variables

### Database Configuration
```bash
# PostgreSQL Database
POSTGRES_DB=sutazai_db
POSTGRES_USER=sutazai_user
POSTGRES_PASSWORD=secure_password_here
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Database URL (auto-generated)
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}
```

### Redis Configuration
```bash
# Redis Cache & Message Queue
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=redis_password
REDIS_DB=0

# Redis URL (auto-generated)
REDIS_URL=redis://:${REDIS_PASSWORD}@${REDIS_HOST}:${REDIS_PORT}/${REDIS_DB}
```

### AI Model Configuration
```bash
# Ollama Local LLM Server
OLLAMA_HOST=ollama
OLLAMA_PORT=11434
OLLAMA_BASE_URL=http://${OLLAMA_HOST}:${OLLAMA_PORT}

# Default model settings
DEFAULT_MODEL=tinyllama:latest
MODEL_TEMPERATURE=0.7
MAX_TOKENS=2048
```

### Vector Database Configuration
```bash
# ChromaDB for Document Embeddings
CHROMA_HOST=chromadb
CHROMA_PORT=8001
CHROMA_PERSIST_DIRECTORY=/data/chromadb

# Qdrant for High-Performance Vector Search
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_API_KEY=your_qdrant_api_key

# Neo4j Graph Database
NEO4J_HOST=neo4j
NEO4J_PORT=7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=neo4j_password
```

### API Configuration
```bash
# Backend API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_DEBUG=false

# Frontend Settings
FRONTEND_HOST=0.0.0.0
FRONTEND_PORT=8501
FRONTEND_DEBUG=false

# CORS settings
ALLOWED_ORIGINS=["http://localhost:8501", "http://localhost:3000"]
```

### Security Configuration
```bash
# JWT Token Settings
JWT_SECRET_KEY=your_super_secret_jwt_key_here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# API Keys for External Services
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
HUGGINGFACE_API_KEY=your_hf_key_here
```

### Monitoring Configuration
```bash
# Prometheus Metrics
PROMETHEUS_PORT=9090
PROMETHEUS_RETENTION=30d

# Grafana Dashboard
GRAFANA_PORT=3000
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin_password

# Loki Log Aggregation
LOKI_PORT=3100
```

### Workflow Engine Configuration
```bash
# n8n Workflow Automation
N8N_PORT=5678
N8N_BASIC_AUTH_ACTIVE=true
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=n8n_password

# LangFlow Visual AI Workflows
LANGFLOW_PORT=7860
LANGFLOW_HOST=0.0.0.0

# Flowise AI Workflow Builder
FLOWISE_PORT=3001
FLOWISE_USERNAME=admin
FLOWISE_PASSWORD=flowise_password
```

## Environment-Specific Configurations

### Development Environment
```bash
# Development specific settings
ENV=development
DEBUG=true
LOG_LEVEL=DEBUG

# Disable authentication for development
AUTH_REQUIRED=false
CORS_ALLOW_ALL=true

# Use smaller resource limits
OLLAMA_NUM_THREAD=4
POSTGRES_MAX_CONNECTIONS=50
```

### Staging Environment
```bash
# Staging specific settings
ENV=staging
DEBUG=false
LOG_LEVEL=INFO

# Enable authentication
AUTH_REQUIRED=true
CORS_ALLOW_ALL=false

# Moderate resource limits
OLLAMA_NUM_THREAD=6
POSTGRES_MAX_CONNECTIONS=100
```

### Production Environment
```bash
# Production specific settings
ENV=production
DEBUG=false
LOG_LEVEL=WARNING

# Strict security settings
AUTH_REQUIRED=true
CORS_ALLOW_ALL=false
SECURE_COOKIES=true
HTTPS_ONLY=true

# Optimized resource limits
OLLAMA_NUM_THREAD=8
POSTGRES_MAX_CONNECTIONS=200
```

## Docker Compose Environment Integration

### Environment File Loading
```yaml
# docker-compose.yml
services:
  backend:
    env_file:
      - .env
      - .env.local  # Optional overrides
    environment:
      - ENV=${ENV:-development}
```

### Environment Variable Substitution
```yaml
# Example service configuration
services:
  postgres:
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "${POSTGRES_PORT}:5432"
```

## Secret Management

### Development Secrets
```bash
# Store in .env file (not committed to git)
API_SECRET=dev_secret_key
DB_PASSWORD=dev_password
```

### Production Secrets
```bash
# Use external secret management
# Docker Swarm Secrets
echo "production_secret" | docker secret create api_secret -

# Kubernetes Secrets
kubectl create secret generic api-secret --from-literal=key=production_secret

# AWS Secrets Manager
aws secretsmanager create-secret --name sutazai/api-key --secret-string "production_secret"
```

### Environment Variable Sources
1. **Docker Compose**: `.env` files
2. **Kubernetes**: ConfigMaps and Secrets
3. **Cloud Providers**: Parameter stores
4. **CI/CD**: Pipeline variables

## Configuration Validation

### Required Variables Check
```bash
# Check if all required variables are set
./scripts/validate_environment.sh

# Validate specific environment
ENV=production ./scripts/validate_environment.sh
```

### Environment Testing
```bash
# Test database connection
python -c "
import os
from sqlalchemy import create_engine
engine = create_engine(os.getenv('DATABASE_URL'))
print('Database connection:', engine.execute('SELECT 1').scalar())
"

# Test Redis connection
python -c "
import redis
import os
r = redis.from_url(os.getenv('REDIS_URL'))
print('Redis connection:', r.ping())
"
```

## Best Practices

### Security
- **Never commit `.env` files** to version control
- **Use strong, unique passwords** for all services
- **Rotate secrets regularly** in production
- **Use external secret management** for production

### Organization
- **Group related variables** together
- **Use consistent naming** conventions
- **Document complex variables** with comments
- **Provide sensible defaults** in `.env.example`

### Deployment
- **Use environment-specific files** for different stages
- **Validate configuration** before deployment
- **Monitor environment changes** in production
- **Backup configuration** before updates

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**
   ```bash
   # Check if variable is set
   echo $POSTGRES_PASSWORD
   
   # Load environment file manually
   source .env
   ```

2. **Database Connection Errors**
   ```bash
   # Verify database URL
   echo $DATABASE_URL
   
   # Test connection
   docker exec sutazai-postgres pg_isready
   ```

3. **Service Discovery Issues**
   ```bash
   # Check Docker network
   docker network ls
   
   # Verify service connectivity
   docker exec sutazai-backend ping postgres
   ```

### Environment Debugging
```bash
# Print all environment variables
docker exec sutazai-backend env | sort

# Check specific variables
docker exec sutazai-backend env | grep POSTGRES

# Validate configuration
docker exec sutazai-backend python -c "from app.config import settings; print(settings.dict())"
```