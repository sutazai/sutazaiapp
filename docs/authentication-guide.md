# SutazAI Authentication System

## Overview

The SutazAI Authentication System provides centralized authentication and authorization for all 69 AI agents and services. It implements industry-standard security practices with OAuth2/JWT tokens, RBAC policies, and API gateway integration.

## Architecture

### Components

1. **Keycloak** - Identity Provider and User Management
2. **Kong Gateway** - API Gateway with Authentication Plugins
3. **JWT Service** - Token Generation and Validation
4. **Service Account Manager** - Manages AI Agent Service Accounts
5. **RBAC Policy Engine** - Role-Based Access Control
6. **HashiCorp Vault** - Secret Management

### Authentication Flow

```
AI Agent → Kong Gateway → JWT Validation → RBAC Check → Backend Service
    ↓            ↓              ↓              ↓            ↓
  Request     Auth Plugin    JWT Service    RBAC Engine   Protected API
```

## Quick Start

### 1. Deploy Authentication Services

```bash
cd /opt/sutazaiapp
docker-compose -f docker-compose.auth.yml up -d
```

### 2. Initialize Authentication System

```bash
./scripts/setup-authentication.sh
```

### 3. Update Agent Configurations

```bash
python3 ./scripts/update-agent-auth.py
```

### 4. Test Authentication System

```bash
python3 ./scripts/test-authentication.py
```

## Service Details

### Keycloak Identity Provider

**URL**: http://localhost:10050
**Admin Console**: http://localhost:10050/admin
**Default Credentials**: admin / sutazai_auth_admin

**Features**:
- OAuth2/OpenID Connect provider
- User and service account management
- Realm configuration for SutazAI
- Client credentials for agents

### Kong API Gateway

**Proxy URL**: http://localhost:10051
**Admin API**: http://localhost:10052

**Plugins Enabled**:
- JWT Authentication
- OAuth2
- Rate Limiting
- CORS
- Prometheus Metrics

### JWT Service

**URL**: http://localhost:10054

**Endpoints**:
- `POST /auth/token` - Generate JWT token
- `POST /auth/validate` - Validate JWT token
- `POST /auth/revoke` - Revoke JWT token
- `GET /metrics` - Service metrics

### Service Account Manager

**URL**: http://localhost:10055

**Endpoints**:
- `POST /service-accounts` - Create service account
- `GET /service-accounts` - List service accounts
- `POST /service-accounts/bulk` - Create multiple accounts
- `POST /service-accounts/create-all-agents` - Create all 69 agent accounts

### RBAC Policy Engine

**URL**: http://localhost:10056

**Endpoints**:
- `POST /access/check` - Check access permissions
- `POST /policies` - Add RBAC policy
- `GET /policies` - List all policies
- `POST /roles/assign` - Assign role to user

## AI Agent Integration

### Authentication Configuration

Each AI agent receives an authentication configuration file:

```json
{
  "authentication": {
    "enabled": true,
    "provider": "keycloak",
    "keycloak_url": "http://keycloak:8080",
    "realm": "sutazai",
    "client_id": "agent-{AGENT_NAME}",
    "client_secret": "[STORED_IN_VAULT]",
    "jwt_service_url": "http://jwt-service:8080",
    "kong_proxy_url": "http://kong:8000",
    "scopes": ["read", "write", "agent"]
  },
  "endpoints": {
    "token": "/auth/token",
    "validate": "/auth/validate",
    "revoke": "/auth/revoke"
  }
}
```

### Agent Authentication Flow

1. **Token Request**: Agent requests JWT token from JWT Service
2. **Token Storage**: Agent stores token with expiry information
3. **API Requests**: Agent includes token in Authorization header
4. **Token Refresh**: Agent refreshes token before expiry
5. **Error Handling**: Agent handles authentication errors gracefully

### Example Python Integration

```python
import httpx
from datetime import datetime, timedelta

class SutazAIAuth:
    def __init__(self, config):
        self.config = config
        self.access_token = None
        self.token_expires_at = None
        
    async def get_access_token(self):
        if self.access_token and self.token_expires_at:
            if datetime.utcnow() < self.token_expires_at - timedelta(seconds=300):
                return self.access_token
                
        return await self.refresh_token()
        
    async def refresh_token(self):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.config['jwt_service_url']}/auth/token",
                json={
                    "service_name": "my-agent",
                    "scopes": self.config['scopes'],
                    "expires_in": 3600
                }
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data["access_token"]
                return self.access_token
                
        return None
        
    async def make_authenticated_request(self, method, url, **kwargs):
        token = await self.get_access_token()
        if not token:
            raise Exception("Failed to get access token")
            
        headers = kwargs.get('headers', {})
        headers['Authorization'] = f'Bearer {token}'
        kwargs['headers'] = headers
        
        async with httpx.AsyncClient() as client:
            return await client.request(method, url, **kwargs)
```

## Service Accounts

### 69 AI Agent Service Accounts

All AI agents have dedicated service accounts with the following naming convention:
- **Client ID**: `agent-{AGENT_NAME}`
- **Scopes**: `["read", "write", "agent"]`
- **Role**: `role:ai-agent`

### Service Account List

1. adversarial-attack-detector
2. agent-creator
3. agent-debugger
4. agent-orchestrator
5. agentgpt-autonomous-executor
6. agentgpt
7. agentzero-coordinator
8. ai-agent-debugger
9. ai-product-manager
10. ai-qa-team-lead
... (all 69 agents)

## RBAC Policies

### Default Roles

- **role:admin** - Full system access
- **role:system** - System-level operations
- **role:ai-agent** - Standard agent permissions
- **role:developer** - Development and API access
- **role:monitor** - Monitoring and metrics access

### Default Policies

```
role:admin → * → *
role:ai-agent → api:agents → read,write
role:ai-agent → ollama:* → read
role:ai-agent → vector-db:* → read,write
role:developer → api:* → read
role:monitor → metrics:* → read
```

### Role Assignments

- `agent-orchestrator` → `role:admin`
- `ai-system-validator` → `role:system`
- All AI agents → `role:ai-agent`

## Security Features

### Token Security

- **JWT Signatures**: HS256 algorithm
- **Token Expiry**: 1-hour default (configurable)
- **Token Revocation**: Blacklist support
- **Scope-based Access**: Fine-grained permissions

### Secret Management

- **Vault Integration**: All secrets stored in HashiCorp Vault
- **Client Secrets**: 32-byte randomly generated
- **JWT Secrets**: Rotatable signing keys
- **Environment Isolation**: Separate secrets per environment

### Rate Limiting

- **Kong Gateway**: Request rate limiting
- **Per-Service Limits**: Different limits per agent type
- **Redis Backend**: Distributed rate limiting

### CORS Protection

- **Origin Control**: Configured allowed origins  
- **Credential Support**: Secure cookie handling
- **Headers Control**: Limited exposed headers

## Monitoring and Logging

### Metrics Endpoints

All authentication services expose `/metrics` endpoints:

- JWT Service metrics: Token generation/validation rates
- Service Account metrics: Account creation/usage
- RBAC metrics: Access checks and policy evaluations
- Kong metrics: Request rates and response times

### Log Aggregation

- **Structured Logging**: JSON format for all services
- **Correlation IDs**: Request tracing across services
- **Access Logs**: All authentication attempts logged
- **Error Tracking**: Failed authentication analysis

### Health Checks

All services implement `/health` endpoints for monitoring:

```bash
curl http://localhost:10054/health  # JWT Service
curl http://localhost:10055/health  # Service Account Manager
curl http://localhost:10056/health  # RBAC Engine
```

## API Documentation

### JWT Service API

#### Generate Token
```http
POST /auth/token
Content-Type: application/json

{
  "service_name": "my-agent",
  "scopes": ["read", "write"],
  "expires_in": 3600
}
```

#### Validate Token
```http
POST /auth/validate
Content-Type: application/json

{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### Service Account Manager API

#### Create Service Account
```http
POST /service-accounts
Content-Type: application/json

{
  "name": "my-agent",
  "description": "My AI agent",
  "scopes": ["read", "write", "agent"]
}
```

#### List Service Accounts
```http
GET /service-accounts
```

### RBAC Policy Engine API

#### Check Access
```http
POST /access/check
Content-Type: application/json

{
  "subject": "agent-my-agent",
  "object": "api:ollama",
  "action": "read"
}
```

#### Add Policy
```http
POST /policies
Content-Type: application/json
Authorization: Bearer {ADMIN_TOKEN}

{
  "subject": "role:my-role",
  "object": "resource:my-resource",
  "action": "read"
}
```

## Troubleshooting

### Common Issues

#### Service Not Starting
```bash
# Check service logs
docker-compose -f docker-compose.auth.yml logs keycloak
docker-compose -f docker-compose.auth.yml logs kong

# Verify dependencies
docker-compose -f docker-compose.yml ps postgres redis
```

#### Token Generation Failing
```bash
# Check service account exists
curl http://localhost:10055/service-accounts/my-agent

# Verify JWT service health
curl http://localhost:10054/health

# Check Keycloak connectivity
curl http://localhost:10050/health/ready
```

#### Access Denied Errors
```bash
# Test RBAC policy
curl -X POST http://localhost:10056/access/check \
  -H "Content-Type: application/json" \
  -d '{"subject": "agent-my-agent", "object": "api:test", "action": "read"}'

# Check token validity
curl -X POST http://localhost:10054/auth/validate \
  -H "Content-Type: application/json" \
  -d '{"token": "YOUR_TOKEN_HERE"}'
```

### Debug Mode

Enable debug logging for services:

```bash
# Set environment variables
export LOG_LEVEL=DEBUG
export JWT_DEBUG=true
export RBAC_DEBUG=true

# Restart services
docker-compose -f docker-compose.auth.yml restart
```

## Production Considerations

### Security Hardening

1. **Change Default Credentials**: Update all default passwords
2. **Enable HTTPS**: Configure SSL certificates
3. **Network Isolation**: Use private networks for internal communication
4. **Secret Rotation**: Implement regular secret rotation
5. **Audit Logging**: Enable comprehensive audit logging

### Scalability

1. **Load Balancing**: Deploy multiple instances behind load balancer
2. **Database Clustering**: Use PostgreSQL clustering for high availability
3. **Redis Clustering**: Configure Redis cluster for session storage
4. **Kong Clustering**: Deploy Kong in cluster mode

### Backup and Recovery

1. **Database Backups**: Regular PostgreSQL backups
2. **Vault Backups**: Vault data backup and recovery procedures
3. **Configuration Backups**: Version control all configurations
4. **Disaster Recovery**: Document recovery procedures

## Support

### Log Locations

- Authentication logs: `/opt/sutazaiapp/logs/`
- Service-specific logs: Available via `docker logs`
- Kong access logs: Available via Kong admin API

### Configuration Files

- Authentication services: `/opt/sutazaiapp/auth/`
- Agent configurations: `/opt/sutazaiapp/auth/agent-configs/`
- Environment variables: `/opt/sutazaiapp/auth/.env`

### Useful Commands

```bash
# Restart authentication services
docker-compose -f docker-compose.auth.yml restart

# View service logs
docker-compose -f docker-compose.auth.yml logs -f

# Test authentication flow
python3 /opt/sutazaiapp/scripts/test-authentication.py

# Update agent configurations
python3 /opt/sutazaiapp/scripts/update-agent-auth.py

# Check service health
curl http://localhost:10054/health
```

For additional support, check the logs directory and service health endpoints.