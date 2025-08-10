"""
SutazAI RBAC Policy Engine
Role-Based Access Control for AI agents and services
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import hvac
import httpx
import structlog
import casbin
from fastapi import FastAPI, HTTPException, Depends, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from keycloak import KeycloakAdmin, KeycloakOpenID
import redis.asyncio as redis
import asyncpg
from jose import jwt, JWTError

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
JWT_SECRET = os.getenv('JWT_SECRET')
if not JWT_SECRET:
    raise ValueError("JWT_SECRET environment variable is required for security")
JWT_ALGORITHM = 'HS256'

KEYCLOAK_SERVER_URL = os.getenv('KEYCLOAK_SERVER_URL', 'http://keycloak:8080')
KEYCLOAK_REALM = os.getenv('KEYCLOAK_REALM', 'sutazai')

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

# Global connections
db_pool = None
redis_client = None
vault_client = None
keycloak_admin = None
enforcer = None

# RBAC Model Configuration
RBAC_MODEL = """
[request_definition]
r = sub, obj, act

[policy_definition]
p = sub, obj, act

[role_definition]
g = _, _

[policy_effect]
e = some(where (p.eft == allow))

[matchers]
m = g(r.sub, p.sub) && r.obj == p.obj && r.act == p.act
"""

# Default RBAC Policies
DEFAULT_POLICIES = [
    # Admin roles
    ("role:admin", "*", "*"),
    ("role:system", "system:*", "*"),
    
    # Agent roles
    ("role:agent", "api:agents", "read"),
    ("role:agent", "api:agents", "write"),
    ("role:agent", "ollama:*", "read"),
    ("role:agent", "vector-db:*", "read"),
    ("role:agent", "vector-db:*", "write"),
    
    # Monitoring roles
    ("role:monitor", "metrics:*", "read"),
    ("role:monitor", "logs:*", "read"),
    
    # Developer roles
    ("role:developer", "api:*", "read"),
    ("role:developer", "api:agents", "write"),
    ("role:developer", "system:deploy", "write"),
    
    # Service roles
    ("role:service", "api:health", "read"),
    ("role:service", "metrics:basic", "read"),
]

# Role assignments for agent types
AGENT_ROLE_ASSIGNMENTS = [
    ("agent-orchestrator", "role:admin"),
    ("ai-system-validator", "role:system"),
    ("ai-senior-backend-developer", "role:developer"),
    ("ai-senior-frontend-developer", "role:developer"),
    ("ai-system-architect", "role:developer"),
    ("monitoring-agent", "role:monitor"),
]

security = HTTPBearer()

class AccessRequest(BaseModel):
    """Access control request"""
    subject: str = Field(..., description="Subject (user/service) requesting access")
    object: str = Field(..., description="Resource being accessed")
    action: str = Field(..., description="Action being performed")

class AccessResponse(BaseModel):
    """Access control response"""
    allowed: bool
    subject: str
    object: str
    action: str
    reason: Optional[str] = None
    timestamp: datetime

class PolicyRequest(BaseModel):
    """Policy creation/update request"""
    subject: str
    object: str
    action: str
    effect: str = "allow"

class RoleAssignmentRequest(BaseModel):
    """Role assignment request"""
    user: str
    role: str

async def init_connections():
    """Initialize database and external service connections"""
    global db_pool, redis_client, vault_client, keycloak_admin, enforcer
    
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
            username=os.getenv('KEYCLOAK_ADMIN_USERNAME', 'admin'),
            password=os.getenv('KEYCLOAK_ADMIN_PASSWORD', ''),
            realm_name=KEYCLOAK_REALM,
            verify=False
        )
        
        logger.info("Keycloak admin client initialized")
        
        # Initialize database schema
        await init_database_schema()
        
        # Initialize Casbin enforcer
        await init_casbin_enforcer()
        
    except Exception as e:
        logger.error("Failed to initialize connections", error=str(e))
        raise

async def init_database_schema():
    """Initialize database schema for RBAC"""
    schema_sql = """
    CREATE TABLE IF NOT EXISTS casbin_rule (
        id SERIAL PRIMARY KEY,
        ptype VARCHAR(100) NOT NULL,
        v0 VARCHAR(255),
        v1 VARCHAR(255),
        v2 VARCHAR(255),
        v3 VARCHAR(255),
        v4 VARCHAR(255),
        v5 VARCHAR(255)
    );
    
    CREATE INDEX IF NOT EXISTS idx_casbin_rule_ptype ON casbin_rule(ptype);
    CREATE INDEX IF NOT EXISTS idx_casbin_rule_v0 ON casbin_rule(v0);
    
    CREATE TABLE IF NOT EXISTS access_logs (
        id SERIAL PRIMARY KEY,
        subject VARCHAR(255) NOT NULL,
        object VARCHAR(255) NOT NULL,
        action VARCHAR(255) NOT NULL,
        allowed BOOLEAN NOT NULL,
        reason TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        ip_address INET,
        user_agent TEXT
    );
    
    CREATE INDEX IF NOT EXISTS idx_access_logs_subject ON access_logs(subject);
    CREATE INDEX IF NOT EXISTS idx_access_logs_timestamp ON access_logs(timestamp);
    """
    
    async with db_pool.acquire() as conn:
        await conn.execute(schema_sql)
    
    logger.info("Database schema initialized")

async def init_casbin_enforcer():
    """Initialize Casbin enforcer with database adapter"""
    global enforcer
    
    try:
        # Create model from string
        model = casbin.Model()
        model.load_model_from_text(RBAC_MODEL)
        
        # For now, use file adapter (could be upgraded to DB adapter)
        adapter = casbin.FileAdapter('/app/policies/policy.csv')
        
        # Create enforcer
        enforcer = casbin.Enforcer(model, adapter)
        
        # Load default policies
        await load_default_policies()
        
        logger.info("Casbin enforcer initialized")
        
    except Exception as e:
        logger.error("Failed to initialize Casbin enforcer", error=str(e))
        # Create a simple in-memory enforcer as fallback
        enforcer = casbin.Enforcer()

async def load_default_policies():
    """Load default RBAC policies"""
    try:
        # Clear existing policies
        enforcer.clear_policy()
        
        # Add default policies
        for subject, obj, action in DEFAULT_POLICIES:
            enforcer.add_policy(subject, obj, action)
        
        # Add role assignments
        for user, role in AGENT_ROLE_ASSIGNMENTS:
            enforcer.add_role_for_user(user, role)
        
        # Save policies
        enforcer.save_policy()
        
        logger.info("Default RBAC policies loaded", 
                   policies=len(DEFAULT_POLICIES),
                   role_assignments=len(AGENT_ROLE_ASSIGNMENTS))
        
    except Exception as e:
        logger.error("Failed to load default policies", error=str(e))

async def log_access_attempt(subject: str, obj: str, action: str, allowed: bool, reason: str = None):
    """Log access attempt to database"""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO access_logs (subject, object, action, allowed, reason)
                VALUES ($1, $2, $3, $4, $5)
                """,
                subject, obj, action, allowed, reason
            )
    except Exception as e:
        logger.error("Failed to log access attempt", error=str(e))

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
    title="SutazAI RBAC Policy Engine",
    description="Role-Based Access Control for SutazAI agents and services",
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

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    """Get current authenticated user from JWT token"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

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
                "keycloak": "ok" if keycloak_admin else "unavailable",
                "enforcer": "ok" if enforcer else "unavailable"
            }
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )

@app.post("/access/check", response_model=AccessResponse)
async def check_access(request: AccessRequest):
    """Check if subject has access to perform action on object"""
    try:
        # Check access using Casbin enforcer
        allowed = False
        reason = "Access denied by policy"
        
        if enforcer:
            allowed = enforcer.enforce(request.subject, request.object, request.action)
            if allowed:
                reason = "Access granted by policy"
        else:
            # Fallback logic if enforcer is not available
            if request.subject.startswith("role:admin"):
                allowed = True
                reason = "Admin access granted"
            elif request.action == "read" and "read" in request.subject:
                allowed = True
                reason = "Read access granted"
        
        # Log the access attempt
        await log_access_attempt(request.subject, request.object, request.action, allowed, reason)
        
        logger.info("Access check performed", 
                   subject=request.subject,
                   object=request.object,
                   action=request.action,
                   allowed=allowed)
        
        return AccessResponse(
            allowed=allowed,
            subject=request.subject,
            object=request.object,
            action=request.action,
            reason=reason,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error("Access check failed", error=str(e))
        # Default to deny on error
        await log_access_attempt(request.subject, request.object, request.action, False, f"Error: {str(e)}")
        
        return AccessResponse(
            allowed=False,
            subject=request.subject,
            object=request.object,
            action=request.action,
            reason=f"Error during access check: {str(e)}",
            timestamp=datetime.utcnow()
        )

@app.post("/policies")
async def add_policy(request: PolicyRequest, current_user: Dict = Depends(get_current_user)):
    """Add new RBAC policy"""
    try:
        if not enforcer:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Policy enforcer not available"
            )
        
        # Add the policy
        success = enforcer.add_policy(request.subject, request.object, request.action)
        
        if success:
            # Save policies
            enforcer.save_policy()
            
            logger.info("Policy added", 
                       subject=request.subject,
                       object=request.object,
                       action=request.action,
                       by=current_user.get('sub'))
            
            return {"message": "Policy added successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Policy already exists"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to add policy", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add policy"
        )

@app.delete("/policies")
async def remove_policy(request: PolicyRequest, current_user: Dict = Depends(get_current_user)):
    """Remove RBAC policy"""
    try:
        if not enforcer:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Policy enforcer not available"
            )
        
        # Remove the policy
        success = enforcer.remove_policy(request.subject, request.object, request.action)
        
        if success:
            # Save policies
            enforcer.save_policy()
            
            logger.info("Policy removed", 
                       subject=request.subject,
                       object=request.object,
                       action=request.action,
                       by=current_user.get('sub'))
            
            return {"message": "Policy removed successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Policy not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to remove policy", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove policy"
        )

@app.get("/policies")
async def list_policies(current_user: Dict = Depends(get_current_user)):
    """List all RBAC policies"""
    try:
        if not enforcer:
            return {"policies": [], "roles": []}
        
        policies = enforcer.get_policy()
        roles = enforcer.get_grouping_policy()
        
        return {
            "policies": [
                {"subject": p[0], "object": p[1], "action": p[2]}
                for p in policies
            ],
            "roles": [
                {"user": r[0], "role": r[1]}
                for r in roles
            ]
        }
        
    except Exception as e:
        logger.error("Failed to list policies", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list policies"
        )

@app.post("/roles/assign")
async def assign_role(request: RoleAssignmentRequest, current_user: Dict = Depends(get_current_user)):
    """Assign role to user"""
    try:
        if not enforcer:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Policy enforcer not available"
            )
        
        # Assign role
        success = enforcer.add_role_for_user(request.user, request.role)
        
        if success:
            # Save policies
            enforcer.save_policy()
            
            logger.info("Role assigned", 
                       user=request.user,
                       role=request.role,
                       by=current_user.get('sub'))
            
            return {"message": f"Role '{request.role}' assigned to '{request.user}'"}
        else:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Role assignment already exists"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to assign role", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to assign role"
        )

@app.delete("/roles/assign")
async def remove_role(request: RoleAssignmentRequest, current_user: Dict = Depends(get_current_user)):
    """Remove role from user"""
    try:
        if not enforcer:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Policy enforcer not available"
            )
        
        # Remove role
        success = enforcer.delete_role_for_user(request.user, request.role)
        
        if success:
            # Save policies
            enforcer.save_policy()
            
            logger.info("Role removed", 
                       user=request.user,
                       role=request.role,
                       by=current_user.get('sub'))
            
            return {"message": f"Role '{request.role}' removed from '{request.user}'"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Role assignment not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to remove role", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove role"
        )

@app.get("/users/{user}/roles")
async def get_user_roles(user: str, current_user: Dict = Depends(get_current_user)):
    """Get roles for a user"""
    try:
        if not enforcer:
            return {"user": user, "roles": []}
        
        roles = enforcer.get_roles_for_user(user)
        
        return {
            "user": user,
            "roles": roles
        }
        
    except Exception as e:
        logger.error("Failed to get user roles", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user roles"
        )

@app.get("/access-logs")
async def get_access_logs(limit: int = 100, current_user: Dict = Depends(get_current_user)):
    """Get recent access logs"""
    try:
        async with db_pool.acquire() as conn:
            logs = await conn.fetch(
                """
                SELECT subject, object, action, allowed, reason, timestamp
                FROM access_logs
                ORDER BY timestamp DESC
                LIMIT $1
                """,
                limit
            )
        
        return [
            {
                "subject": log['subject'],
                "object": log['object'],
                "action": log['action'],
                "allowed": log['allowed'],
                "reason": log['reason'],
                "timestamp": log['timestamp']
            }
            for log in logs
        ]
        
    except Exception as e:
        logger.error("Failed to get access logs", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get access logs"
        )

@app.get("/metrics")
async def get_metrics():
    """Get RBAC metrics"""
    try:
        policy_count = 0
        role_count = 0
        
        if enforcer:
            policy_count = len(enforcer.get_policy())
            role_count = len(enforcer.get_grouping_policy())
        
        # Get access statistics
        async with db_pool.acquire() as conn:
            total_requests = await conn.fetchval("SELECT COUNT(*) FROM access_logs")
            allowed_requests = await conn.fetchval("SELECT COUNT(*) FROM access_logs WHERE allowed = TRUE")
            denied_requests = total_requests - allowed_requests if total_requests else 0
            
            # Get recent activity (last 24 hours)
            recent_requests = await conn.fetchval(
                "SELECT COUNT(*) FROM access_logs WHERE timestamp > NOW() - INTERVAL '24 hours'"
            )
        
        return {
            "policies": {
                "total": policy_count
            },
            "roles": {
                "total": role_count
            },
            "access_requests": {
                "total": total_requests,
                "allowed": allowed_requests,
                "denied": denied_requests,
                "recent_24h": recent_requests
            },
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