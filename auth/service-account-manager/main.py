"""
SutazAI Service Account Manager
Manages service accounts for 69 AI agents with Keycloak integration
"""

import os
import json
import secrets
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import hvac
import httpx
import structlog
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from keycloak import KeycloakAdmin
import redis.asyncio as redis
import asyncpg

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Configuration
KEYCLOAK_SERVER_URL = os.getenv('KEYCLOAK_SERVER_URL', 'http://keycloak:8080')
KEYCLOAK_REALM = os.getenv('KEYCLOAK_REALM', 'sutazai')
KEYCLOAK_ADMIN_CLIENT_ID = os.getenv('KEYCLOAK_ADMIN_CLIENT_ID', 'admin-cli')
KEYCLOAK_ADMIN_USERNAME = os.getenv('KEYCLOAK_ADMIN_USERNAME', 'admin')
KEYCLOAK_ADMIN_PASSWORD = os.getenv('KEYCLOAK_ADMIN_PASSWORD', '')

VAULT_ADDR = os.getenv('VAULT_ADDR', 'http://vault:8200')
VAULT_TOKEN = os.getenv('VAULT_TOKEN', '')

# Database URL should be provided via environment variable
# Format: postgresql://user:password@host:port/database
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    # Build from individual components if DATABASE_URL not provided
    db_user = os.getenv('POSTGRES_USER', 'postgres')
    db_pass = os.getenv('POSTGRES_PASSWORD', '')
    db_host = os.getenv('POSTGRES_HOST', 'postgres')
    db_port = os.getenv('POSTGRES_PORT', '5432')
    db_name = os.getenv('POSTGRES_DB', 'sutazai')
    if db_pass:
        DATABASE_URL = f'postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}'
    else:
        DATABASE_URL = f'postgresql://{db_user}@{db_host}:{db_port}/{db_name}'
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379/0')

# 69 AI Agents that need service accounts
AI_AGENTS = [
    'adversarial-attack-detector', 'agent-creator', 'agent-debugger', 'agent-orchestrator',
    'agentgpt-autonomous-executor', 'agentgpt', 'agentzero-coordinator', 'ai-agent-debugger',
    'ai-product-manager', 'ai-qa-team-lead', 'ai-scrum-master', 'ai-senior-backend-developer',
    'ai-senior-engineer', 'ai-senior-frontend-developer', 'ai-senior-full-stack-developer',
    'ai-system-architect', 'ai-system-validator', 'ai-testing-qa-validator', 'aider',
    'attention-optimizer', 'autogen', 'autogpt', 'automated-incident-responder',
    'autonomous-task-executor', 'awesome-code-ai', 'babyagi', 'bias-and-fairness-auditor',
    'bigagi-system-manager', 'browser-automation-orchestrator', 'causal-inference-expert',
    'cicd-pipeline-orchestrator', 'code-improver', 'code-quality-gateway-sonarqube',
    'codebase-team-lead', 'cognitive-architecture-designer', 'cognitive-load-monitor',
    'compute-scheduler-and-optimizer', 'container-orchestrator-k3s', 'container-vulnerability-scanner-trivy',
    'context-framework', 'cpu-only-hardware-optimizer', 'crewai', 'data-analysis-engineer',
    'data-drift-detector', 'data-lifecycle-manager', 'data-pipeline-engineer', 'data-version-controller-dvc',
    'deep-learning-brain-architect', 'deep-learning-brain-manager', 'deep-local-brain-builder',
    'deploy-automation-master', 'deployment-automation-master', 'devika', 'dify-automation-specialist',
    'distributed-computing-architect', 'distributed-tracing-analyzer-jaeger', 'document-knowledge-manager',
    'edge-computing-optimizer', 'edge-inference-proxy', 'emergency-shutdown-coordinator',
    'energy-consumption-optimize', 'episodic-memory-engineer', 'ethical-governor',
    'evolution-strategy-trainer', 'experiment-tracker', 'explainability-and-transparency-agent',
    'explainable-ai-specialist', 'federated-learning-coordinator', 'finrobot', 'flowiseai-flow-manager',
    'fsdp', 'garbage-collector-coordinator', 'garbage-collector', 'genetic-algorithm-tuner',
    'goal-setting-and-planning-agent', 'gpt-engineer', 'gpu-hardware-optimizer'
]

# Global connections
db_pool = None
redis_client = None
vault_client = None
keycloak_admin = None

class ServiceAccountRequest(BaseModel):
    """Service account creation request"""
    name: str = Field(..., description="Service account name")
    description: Optional[str] = Field(None, description="Service account description")
    scopes: List[str] = Field(default=["read", "write"], description="Service account scopes")
    attributes: Optional[Dict[str, Any]] = Field(default={}, description="Additional attributes")

class ServiceAccountResponse(BaseModel):
    """Service account response"""
    name: str
    client_id: str
    client_secret: str
    scopes: List[str]
    keycloak_user_id: Optional[str] = None
    created_at: datetime
    active: bool = True

class BulkServiceAccountRequest(BaseModel):
    """Bulk service account creation request"""
    agent_names: List[str]
    default_scopes: List[str] = Field(default=["read", "write", "agent"], description="Default scopes for all agents")

async def init_connections():
    """Initialize database and external service connections"""
    global db_pool, redis_client, vault_client, keycloak_admin
    
    try:
        # Initialize database pool
        db_pool = await asyncpg.create_pool(DATABASE_URL)
        logger.info("Database connection pool initialized")
        
        # Initialize Redis
        redis_client = redis.from_url(REDIS_URL)
        await redis_client.ping()
        logger.info("Redis connection initialized")
        
        # Initialize Vault client
        if VAULT_TOKEN:
            vault_client = hvac.Client(url=VAULT_ADDR, token=VAULT_TOKEN)
            if vault_client.is_authenticated():
                logger.info("Vault client initialized and authenticated")
        
        # Initialize Keycloak admin client
        keycloak_admin = KeycloakAdmin(
            server_url=KEYCLOAK_SERVER_URL,
            username=KEYCLOAK_ADMIN_USERNAME,
            password=KEYCLOAK_ADMIN_PASSWORD,
            realm_name=KEYCLOAK_REALM,
            verify=False
        )
        
        logger.info("Keycloak admin client initialized")
        
        # Initialize database schema
        await init_database_schema()
        
        # Ensure Keycloak realm and clients are configured
        await setup_keycloak_realm()
        
    except Exception as e:
        logger.error("Failed to initialize connections", error=str(e))
        raise

async def init_database_schema():
    """Initialize database schema for service accounts"""
    schema_sql = """
    CREATE TABLE IF NOT EXISTS service_accounts (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) UNIQUE NOT NULL,
        client_id VARCHAR(255) UNIQUE NOT NULL,
        client_secret VARCHAR(255) NOT NULL,
        scopes TEXT[] DEFAULT ARRAY['read'],
        keycloak_user_id VARCHAR(255),
        attributes JSONB DEFAULT '{}',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        active BOOLEAN DEFAULT TRUE
    );
    
    CREATE INDEX IF NOT EXISTS idx_service_accounts_name ON service_accounts(name);
    CREATE INDEX IF NOT EXISTS idx_service_accounts_client_id ON service_accounts(client_id);
    CREATE INDEX IF NOT EXISTS idx_service_accounts_active ON service_accounts(active);
    """
    
    async with db_pool.acquire() as conn:
        await conn.execute(schema_sql)
    
    logger.info("Database schema initialized")

async def setup_keycloak_realm():
    """Setup Keycloak realm and client configurations"""
    try:
        # Check if realm exists, create if not
        try:
            realm = keycloak_admin.get_realm(KEYCLOAK_REALM)
            logger.info("Keycloak realm exists", realm=KEYCLOAK_REALM)
        except Exception:
            # Create realm
            realm_config = {
                "realm": KEYCLOAK_REALM,
                "enabled": True,
                "displayName": "SutazAI Authentication Realm",
                "loginTheme": "keycloak",
                "adminTheme": "keycloak",
                "accountTheme": "keycloak",
                "emailTheme": "keycloak",
                "sslRequired": "none",
                "registrationAllowed": False,
                "loginWithEmailAllowed": True,
                "duplicateEmailsAllowed": False,
                "resetPasswordAllowed": True,
                "editUsernameAllowed": False,
                "bruteForceProtected": True
            }
            
            keycloak_admin.create_realm(realm_config)
            logger.info("Keycloak realm created", realm=KEYCLOAK_REALM)
        
        # Create client scopes
        client_scopes = [
            {
                "name": "agent",
                "description": "AI Agent access scope",
                "protocol": "openid-connect",
                "attributes": {
                    "consent.screen.text": "${agentScopeConsentText}",
                    "display.on.consent.screen": "true"
                }
            },
            {
                "name": "read",
                "description": "Read access scope",
                "protocol": "openid-connect"
            },
            {
                "name": "write",
                "description": "Write access scope",
                "protocol": "openid-connect"
            },
            {
                "name": "admin",
                "description": "Administrative access scope",
                "protocol": "openid-connect"
            }
        ]
        
        for scope in client_scopes:
            try:
                keycloak_admin.create_client_scope(scope)
                logger.info("Created client scope", scope=scope["name"])
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning("Failed to create client scope", scope=scope["name"], error=str(e))
        
        # Create main backend client
        backend_client_config = {
            "clientId": "sutazai-backend",
            "name": "SutazAI Backend Client",
            "description": "Main backend client for SutazAI",
            "enabled": True,
            "clientAuthenticatorType": "client-secret",
            "secret": os.getenv('KEYCLOAK_CLIENT_SECRET', secrets.token_urlsafe(32)),
            "redirectUris": ["http://localhost:10051/auth/callback", "http://localhost:10010/auth/callback"],
            "webOrigins": ["*"],
            "standardFlowEnabled": True,
            "serviceAccountsEnabled": True,
            "authorizationServicesEnabled": True,
            "protocol": "openid-connect",
            "attributes": {
                "saml.assertion.signature": "false",
                "saml.force.post.binding": "false", 
                "saml.multivalued.roles": "false",
                "saml.encrypt": "false",
                "oauth2.device.authorization.grant.enabled": "false",
                "backchannel.logout.revoke.offline.tokens": "false"
            }
        }
        
        try:
            keycloak_admin.create_client(backend_client_config)
            logger.info("Created backend client")
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning("Failed to create backend client", error=str(e))
        
        logger.info("Keycloak realm setup completed")
        
    except Exception as e:
        logger.error("Failed to setup Keycloak realm", error=str(e))
        # Don't fail the startup for Keycloak issues
        pass

async def close_connections():
    """Close all connections"""
    global db_pool, redis_client
    
    if redis_client:
        await redis_client.close()
    
    if db_pool:
        await db_pool.close()
    
    logger.info("All connections closed")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    await init_connections()
    yield
    await close_connections()

# Initialize FastAPI app
app = FastAPI(
    title="SutazAI Service Account Manager",
    description="Manages service accounts for 69 AI agents",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def create_keycloak_service_account(name: str, scopes: List[str]) -> Dict[str, str]:
    """Create service account in Keycloak"""
    try:
        client_id = f"agent-{name}"
        client_secret = secrets.token_urlsafe(32)
        
        client_config = {
            "clientId": client_id,
            "name": f"AI Agent: {name}",
            "description": f"Service account for AI agent {name}",
            "enabled": True,
            "clientAuthenticatorType": "client-secret",
            "secret": client_secret,
            "serviceAccountsEnabled": True,
            "standardFlowEnabled": False,
            "implicitFlowEnabled": False,
            "directAccessGrantsEnabled": True,
            "protocol": "openid-connect",
            "attributes": {
                "access.token.lifespan": "3600",
                "client_credentials.use_refresh_token": "false"
            }
        }
        
        # Create client in Keycloak
        client_uuid = keycloak_admin.create_client(client_config)
        
        # Get service account user
        service_account_user = keycloak_admin.get_client_service_account_user(client_uuid)
        
        logger.info("Created Keycloak service account", 
                   client_id=client_id, 
                   user_id=service_account_user['id'])
        
        return {
            'client_id': client_id,
            'client_secret': client_secret,
            'keycloak_user_id': service_account_user['id']
        }
        
    except Exception as e:
        logger.error("Failed to create Keycloak service account", name=name, error=str(e))
        raise

async def store_secret_in_vault(name: str, secret: str):
    """Store service account secret in Vault"""
    if vault_client and vault_client.is_authenticated():
        try:
            vault_client.secrets.kv.v2.create_or_update_secret(
                path=f"service-accounts/{name}",
                secret={'client_secret': secret}
            )
            logger.info("Stored secret in Vault", name=name)
        except Exception as e:
            logger.warning("Failed to store secret in Vault", name=name, error=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        
        # Check Redis connection
        await redis_client.ping()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "services": {
                "database": "ok",
                "redis": "ok",
                "vault": "ok" if vault_client and vault_client.is_authenticated() else "unavailable",
                "keycloak": "ok" if keycloak_admin else "unavailable"
            }
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )

@app.post("/service-accounts", response_model=ServiceAccountResponse)
async def create_service_account(request: ServiceAccountRequest):
    """Create a single service account"""
    try:
        # Check if service account already exists
        async with db_pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT * FROM service_accounts WHERE name = $1",
                request.name
            )
            
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Service account '{request.name}' already exists"
                )
        
        # Create Keycloak service account
        keycloak_data = await create_keycloak_service_account(request.name, request.scopes)
        
        # Store secret in Vault
        await store_secret_in_vault(request.name, keycloak_data['client_secret'])
        
        # Store in database
        async with db_pool.acquire() as conn:
            account_id = await conn.fetchval(
                """
                INSERT INTO service_accounts (name, client_id, client_secret, scopes, keycloak_user_id, attributes)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
                """,
                request.name,
                keycloak_data['client_id'],
                keycloak_data['client_secret'],
                request.scopes,
                keycloak_data['keycloak_user_id'],
                json.dumps(request.attributes or {})
            )
        
        logger.info("Service account created", name=request.name, id=account_id)
        
        return ServiceAccountResponse(
            name=request.name,
            client_id=keycloak_data['client_id'],
            client_secret=keycloak_data['client_secret'],
            scopes=request.scopes,
            keycloak_user_id=keycloak_data['keycloak_user_id'],
            created_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create service account", name=request.name, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create service account: {str(e)}"
        )

@app.post("/service-accounts/bulk")
async def create_bulk_service_accounts(request: BulkServiceAccountRequest):
    """Create service accounts for multiple agents"""
    results = []
    errors = []
    
    for agent_name in request.agent_names:
        try:
            account_request = ServiceAccountRequest(
                name=agent_name,
                description=f"Service account for AI agent {agent_name}",
                scopes=request.default_scopes,
                attributes={"agent_type": "ai", "auto_created": True}
            )
            
            result = await create_service_account(account_request)
            results.append(result)
            
        except HTTPException as e:
            if e.status_code == status.HTTP_409_CONFLICT:
                # Skip existing accounts
                logger.info("Skipping existing service account", name=agent_name)
                continue
            else:
                error_msg = f"Failed to create {agent_name}: {e.detail}"
                errors.append(error_msg)
                logger.error("Bulk creation error", name=agent_name, error=error_msg)
        
        except Exception as e:
            error_msg = f"Failed to create {agent_name}: {str(e)}"
            errors.append(error_msg)
            logger.error("Bulk creation error", name=agent_name, error=error_msg)
    
    return {
        "created": len(results),
        "errors": len(errors),
        "accounts": results,
        "error_details": errors
    }

@app.post("/service-accounts/create-all-agents")
async def create_all_agent_accounts():
    """Create service accounts for all 69 AI agents"""
    request = BulkServiceAccountRequest(
        agent_names=AI_AGENTS,
        default_scopes=["read", "write", "agent"]
    )
    
    return await create_bulk_service_accounts(request)

@app.get("/service-accounts")
async def list_service_accounts():
    """List all service accounts"""
    try:
        async with db_pool.acquire() as conn:
            accounts = await conn.fetch(
                """
                SELECT name, client_id, scopes, keycloak_user_id, attributes, 
                       created_at, updated_at, active
                FROM service_accounts 
                ORDER BY name
                """
            )
        
        return [
            {
                "name": account['name'],
                "client_id": account['client_id'],
                "scopes": account['scopes'],
                "keycloak_user_id": account['keycloak_user_id'],
                "attributes": json.loads(account['attributes']) if account['attributes'] else {},
                "created_at": account['created_at'],
                "updated_at": account['updated_at'],
                "active": account['active']
            }
            for account in accounts
        ]
        
    except Exception as e:
        logger.error("Failed to list service accounts", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list service accounts"
        )

@app.get("/service-accounts/{name}")
async def get_service_account(name: str):
    """Get specific service account"""
    try:
        async with db_pool.acquire() as conn:
            account = await conn.fetchrow(
                """
                SELECT name, client_id, scopes, keycloak_user_id, attributes,
                       created_at, updated_at, active
                FROM service_accounts 
                WHERE name = $1
                """,
                name
            )
            
            if not account:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Service account '{name}' not found"
                )
        
        return {
            "name": account['name'],
            "client_id": account['client_id'],
            "scopes": account['scopes'],
            "keycloak_user_id": account['keycloak_user_id'],
            "attributes": json.loads(account['attributes']) if account['attributes'] else {},
            "created_at": account['created_at'],
            "updated_at": account['updated_at'],
            "active": account['active']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get service account", name=name, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get service account"
        )

@app.delete("/service-accounts/{name}")
async def delete_service_account(name: str):
    """Delete service account"""
    try:
        async with db_pool.acquire() as conn:
            account = await conn.fetchrow(
                "SELECT client_id FROM service_accounts WHERE name = $1",
                name
            )
            
            if not account:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Service account '{name}' not found"
                )
            
            # Delete from Keycloak
            try:
                client_id = keycloak_admin.get_client_id(account['client_id'])
                keycloak_admin.delete_client(client_id)
                logger.info("Deleted Keycloak client", client_id=account['client_id'])
            except Exception as e:
                logger.warning("Failed to delete Keycloak client", error=str(e))
            
            # Delete from database
            await conn.execute(
                "DELETE FROM service_accounts WHERE name = $1",
                name
            )
            
            # Delete from Vault
            if vault_client and vault_client.is_authenticated():
                try:
                    vault_client.secrets.kv.v2.delete_metadata_and_all_versions(
                        path=f"service-accounts/{name}"
                    )
                    logger.info("Deleted secret from Vault", name=name)
                except Exception as e:
                    logger.warning("Failed to delete secret from Vault", error=str(e))
        
        logger.info("Service account deleted", name=name)
        return {"message": f"Service account '{name}' deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete service account", name=name, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete service account"
        )

@app.get("/metrics")
async def get_metrics():
    """Get service account metrics"""
    try:
        async with db_pool.acquire() as conn:
            total_accounts = await conn.fetchval("SELECT COUNT(*) FROM service_accounts")
            active_accounts = await conn.fetchval("SELECT COUNT(*) FROM service_accounts WHERE active = TRUE")
            
            # Get accounts by scope
            scope_stats = await conn.fetch(
                """
                SELECT UNNEST(scopes) as scope, COUNT(*) as count
                FROM service_accounts
                WHERE active = TRUE
                GROUP BY scope
                ORDER BY count DESC
                """
            )
        
        return {
            "total_accounts": total_accounts,
            "active_accounts": active_accounts,
            "inactive_accounts": total_accounts - active_accounts,
            "scope_distribution": {row['scope']: row['count'] for row in scope_stats},
            "ai_agents_total": len(AI_AGENTS),
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get metrics"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)