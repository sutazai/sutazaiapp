"""
SutazAI JWT Authentication Service
Provides JWT token generation, validation, and management for AI agents
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import jwt
import hvac
import httpx
import structlog
from fastapi import FastAPI, HTTPException, Depends, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from keycloak import KeycloakAdmin, KeycloakOpenID
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
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
JWT_SECRET = os.getenv('JWT_SECRET', 'sutazai_jwt_secret_key')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRY = int(os.getenv('JWT_EXPIRY', 3600))
JWT_ISSUER = os.getenv('JWT_ISSUER', 'sutazai-auth')
JWT_AUDIENCE = os.getenv('JWT_AUDIENCE', 'sutazai-api')

KEYCLOAK_SERVER_URL = os.getenv('KEYCLOAK_SERVER_URL', 'http://keycloak:8080')
KEYCLOAK_REALM = os.getenv('KEYCLOAK_REALM', 'sutazai')
KEYCLOAK_CLIENT_ID = os.getenv('KEYCLOAK_CLIENT_ID', 'sutazai-backend')
KEYCLOAK_CLIENT_SECRET = os.getenv('KEYCLOAK_CLIENT_SECRET', '')

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
keycloak_openid = None

class TokenRequest(BaseModel):
    """Token generation request model"""
    service_name: str = Field(..., description="Name of the service requesting token")
    scopes: List[str] = Field(default=["read"], description="Requested token scopes")
    expires_in: Optional[int] = Field(default=None, description="Token expiry in seconds")

class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    scope: str
    issued_at: datetime
    issuer: str

class TokenValidationRequest(BaseModel):
    """Token validation request model"""
    token: str

class TokenValidationResponse(BaseModel):
    """Token validation response model"""
    valid: bool
    service_name: Optional[str] = None
    scopes: Optional[List[str]] = None
    expires_at: Optional[datetime] = None
    issued_at: Optional[datetime] = None

class ServiceAccount(BaseModel):
    """Service account model"""
    name: str
    client_id: str
    client_secret: str
    scopes: List[str]
    created_at: datetime
    updated_at: datetime
    active: bool = True

security = HTTPBearer()

async def init_connections():
    """Initialize database and external service connections"""
    global db_pool, redis_client, vault_client, keycloak_admin, keycloak_openid
    
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
            else:
                logger.warning("Vault client initialized but not authenticated")
        
        # Initialize Keycloak clients
        keycloak_admin = KeycloakAdmin(
            server_url=KEYCLOAK_SERVER_URL,
            username=os.getenv('KEYCLOAK_ADMIN', 'admin'),
            password=os.getenv('KEYCLOAK_ADMIN_PASSWORD', ''),
            realm_name=KEYCLOAK_REALM,
            verify=False
        )
        
        keycloak_openid = KeycloakOpenID(
            server_url=KEYCLOAK_SERVER_URL,
            client_id=KEYCLOAK_CLIENT_ID,
            realm_name=KEYCLOAK_REALM,
            client_secret_key=KEYCLOAK_CLIENT_SECRET,
            verify=False
        )
        
        logger.info("Keycloak clients initialized")
        
        # Initialize database schema
        await init_database_schema()
        
    except Exception as e:
        logger.error("Failed to initialize connections", error=str(e))
        raise

async def init_database_schema():
    """Initialize database schema for JWT service"""
    schema_sql = """
    CREATE TABLE IF NOT EXISTS service_accounts (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) UNIQUE NOT NULL,
        client_id VARCHAR(255) UNIQUE NOT NULL,
        client_secret VARCHAR(255) NOT NULL,
        scopes TEXT[] DEFAULT ARRAY['read'],
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        active BOOLEAN DEFAULT TRUE
    );
    
    CREATE TABLE IF NOT EXISTS jwt_tokens (
        id SERIAL PRIMARY KEY,
        jti VARCHAR(255) UNIQUE NOT NULL,
        service_name VARCHAR(255) NOT NULL,
        token_hash VARCHAR(255) NOT NULL,
        scopes TEXT[],
        issued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP NOT NULL,
        revoked BOOLEAN DEFAULT FALSE
    );
    
    CREATE INDEX IF NOT EXISTS idx_jwt_tokens_jti ON jwt_tokens(jti);
    CREATE INDEX IF NOT EXISTS idx_jwt_tokens_service ON jwt_tokens(service_name);
    CREATE INDEX IF NOT EXISTS idx_jwt_tokens_expires ON jwt_tokens(expires_at);
    """
    
    async with db_pool.acquire() as conn:
        await conn.execute(schema_sql)
    
    logger.info("Database schema initialized")

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
    title="SutazAI JWT Authentication Service",
    description="JWT token generation and validation for SutazAI agents",
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

def generate_jwt_token(service_name: str, scopes: List[str], expires_in: int = JWT_EXPIRY) -> Dict[str, Any]:
    """Generate JWT token for service"""
    now = datetime.utcnow()
    exp = now + timedelta(seconds=expires_in)
    jti = f"{service_name}_{int(now.timestamp())}"
    
    payload = {
        'iss': JWT_ISSUER,
        'aud': JWT_AUDIENCE,
        'sub': service_name,
        'iat': int(now.timestamp()),
        'exp': int(exp.timestamp()),
        'jti': jti,
        'scopes': scopes,
        'service_name': service_name
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    return {
        'token': token,
        'payload': payload,
        'jti': jti,
        'expires_at': exp
    }

async def validate_jwt_token(token: str) -> Dict[str, Any]:
    """Validate JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        
        # Check if token is revoked
        jti = payload.get('jti')
        if jti:
            async with db_pool.acquire() as conn:
                revoked = await conn.fetchval(
                    "SELECT revoked FROM jwt_tokens WHERE jti = $1",
                    jti
                )
                if revoked:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Token has been revoked"
                    )
        
        return payload
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

async def get_current_service(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    """Dependency to get current authenticated service"""
    token = credentials.credentials
    return await validate_jwt_token(token)

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

@app.post("/auth/token", response_model=TokenResponse)
async def generate_token(request: TokenRequest):
    """Generate JWT token for service"""
    try:
        # Validate service account exists
        async with db_pool.acquire() as conn:
            service_account = await conn.fetchrow(
                "SELECT * FROM service_accounts WHERE name = $1 AND active = TRUE",
                request.service_name
            )
            
            if not service_account:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Service account '{request.service_name}' not found or inactive"
                )
        
        expires_in = request.expires_in or JWT_EXPIRY
        token_data = generate_jwt_token(request.service_name, request.scopes, expires_in)
        
        # Store token in database
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO jwt_tokens (jti, service_name, token_hash, scopes, expires_at)
                VALUES ($1, $2, $3, $4, $5)
                """,
                token_data['jti'],
                request.service_name,
                token_data['token'][:64],  # Store hash of token for tracking
                request.scopes,
                token_data['expires_at']
            )
        
        # Cache token in Redis for fast validation
        await redis_client.setex(
            f"jwt:{token_data['jti']}",
            expires_in,
            json.dumps({
                'service_name': request.service_name,
                'scopes': request.scopes,
                'expires_at': token_data['expires_at'].isoformat()
            })
        )
        
        logger.info("JWT token generated", service=request.service_name, jti=token_data['jti'])
        
        return TokenResponse(
            access_token=token_data['token'],
            expires_in=expires_in,
            scope=' '.join(request.scopes),
            issued_at=datetime.utcnow(),
            issuer=JWT_ISSUER
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to generate token", error=str(e), service=request.service_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate token"
        )

@app.post("/auth/validate", response_model=TokenValidationResponse)
async def validate_token(request: TokenValidationRequest):
    """Validate JWT token"""
    try:
        payload = await validate_jwt_token(request.token)
        
        return TokenValidationResponse(
            valid=True,
            service_name=payload.get('service_name'),
            scopes=payload.get('scopes', []),
            expires_at=datetime.fromtimestamp(payload.get('exp', 0)),
            issued_at=datetime.fromtimestamp(payload.get('iat', 0))
        )
        
    except HTTPException:
        return TokenValidationResponse(valid=False)
    except Exception as e:
        logger.error("Token validation failed", error=str(e))
        return TokenValidationResponse(valid=False)

@app.post("/auth/revoke")
async def revoke_token(request: TokenValidationRequest, current_service: Dict = Depends(get_current_service)):
    """Revoke JWT token"""
    try:
        payload = jwt.decode(request.token, JWT_SECRET, algorithms=[JWT_ALGORITHM], verify=False)
        jti = payload.get('jti')
        
        if not jti:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid token format"
            )
        
        # Mark token as revoked in database
        async with db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE jwt_tokens SET revoked = TRUE WHERE jti = $1",
                jti
            )
        
        # Remove from Redis cache
        await redis_client.delete(f"jwt:{jti}")
        
        logger.info("JWT token revoked", jti=jti)
        
        return {"message": "Token revoked successfully"}
        
    except Exception as e:
        logger.error("Failed to revoke token", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke token"
        )

@app.get("/auth/service-accounts")
async def list_service_accounts(current_service: Dict = Depends(get_current_service)):
    """List all service accounts"""
    try:
        async with db_pool.acquire() as conn:
            accounts = await conn.fetch(
                "SELECT name, client_id, scopes, created_at, updated_at, active FROM service_accounts ORDER BY name"
            )
        
        return [
            ServiceAccount(
                name=account['name'],
                client_id=account['client_id'],
                client_secret="*****",  # Never expose secrets
                scopes=account['scopes'],
                created_at=account['created_at'],
                updated_at=account['updated_at'],
                active=account['active']
            )
            for account in accounts
        ]
        
    except Exception as e:
        logger.error("Failed to list service accounts", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list service accounts"
        )

@app.get("/metrics")
async def get_metrics():
    """Get authentication metrics"""
    try:
        async with db_pool.acquire() as conn:
            # Get token statistics
            total_tokens = await conn.fetchval("SELECT COUNT(*) FROM jwt_tokens")
            active_tokens = await conn.fetchval(
                "SELECT COUNT(*) FROM jwt_tokens WHERE expires_at > NOW() AND revoked = FALSE"
            )
            revoked_tokens = await conn.fetchval("SELECT COUNT(*) FROM jwt_tokens WHERE revoked = TRUE")
            
            # Get service account statistics
            total_accounts = await conn.fetchval("SELECT COUNT(*) FROM service_accounts")
            active_accounts = await conn.fetchval("SELECT COUNT(*) FROM service_accounts WHERE active = TRUE")
        
        return {
            "tokens": {
                "total": total_tokens,
                "active": active_tokens,
                "revoked": revoked_tokens
            },
            "service_accounts": {
                "total": total_accounts,
                "active": active_accounts
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