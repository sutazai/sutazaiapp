"""Main module for the SutazAI API application."""
from datetime import datetime
import logging
import time
from typing import List
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from v1.database import SessionLocal, engine
from v1 import models, schemas
from v1.config import settings

# Optional imports, handle potential import errors
try:
    from v1.sutazai_service import SutazAiService
    from v1.system_verify import verify_system
    from v1.system_optimizer import optimize_system
    from v1.system_validator import validate_system
except ImportError as e:
    logging.warning(f"Optional module import failed: {e}")
    SutazAiService = None
    verify_system = None
    optimize_system = None
    validate_system = None

from tenacity import retry, stop_after_attempt, wait_fixed
from redis import Redis
import cProfile
import pstats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/sutazai/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create database tables
try:
    models.Base.metadata.create_all(bind=engine)
except SQLAlchemyError as e:
    logger.error(f"Failed to create database tables: {str(e)}")
    raise

# Main FastAPI application
app = FastAPI(
    title="SutazAI API",
    version="1.0.0",
    description="SutazAI REST API"
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=settings.ALLOWED_HOSTS
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize Redis client
redis_client = Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)

class RateLimiter:
    """Rate limiter to control the number of requests per minute."""
    def __init__(self, requests_per_minute: int = 60):
        self.requests = {}
        self.limit = requests_per_minute

    async def check(self, ip: str) -> bool:
        """Check if the IP has exceeded the request limit."""
        now = time.time()
        self.cleanup(now)
        
        if ip not in self.requests:
            self.requests[ip] = []
        
        self.requests[ip].append(now)
        return len(self.requests[ip]) <= self.limit

    def cleanup(self, now: float):
        """Remove outdated requests from the log."""
        minute_ago = now - 60
        for ip in list(self.requests.keys()):
            self.requests[ip] = [t for t in self.requests[ip] if t > minute_ago]
            if not self.requests[ip]:
                del self.requests[ip]

rate_limiter = RateLimiter()

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to the response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to incoming requests."""
    if not await rate_limiter.check(request.client.host):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests"
        )
    return await call_next(request)

def get_db():
    """Provide a database session with error handling."""
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred"
        )
    finally:
        db.close()

@app.get("/", response_model=schemas.Message)
async def root():
    """Root endpoint returning a welcome message."""
    return {"message": "Welcome to SutazAI"}

@app.get("/health", response_model=schemas.HealthCheck)
async def health_check():
    """Health check endpoint to verify service status."""
    try:
        # Check database connection
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        return {
            "status": "ok",
            "timestamp": datetime.utcnow(),
            "version": app.version,
            "database": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions and log them."""
    logger.error(f"HTTP error occurred: {exc.detail}")
    return {"detail": exc.detail, "status_code": exc.status_code}

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions and log them."""
    logger.error(f"Unexpected error occurred: {str(exc)}")
    return {
        "detail": "An unexpected error occurred",
        "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR
    }

# Added quantum health monitoring
class QuantumHealthMonitor:
    def __init__(self):
        self.entropy_source = QuantumEntropyGenerator()
        
    def get_system_entropy(self):
        return self.entropy_source.get_entropy_bits(256)

def initialize_quantum():
    # original initialization...
    # service = QuantumService()
    service = SutazAiService()  # updated service
    service.start()

# Profile the main function
profiler = cProfile.Profile()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def main():
    """Main entry point for the system."""
    profiler.enable()
    try:
        verify_system()
        optimize_system()
        validate_system()
        print("System initialization complete")
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        raise
    finally:
        profiler.disable()
        # Save profiling results
        with open('profile_results.prof', 'w') as f:
            ps = pstats.Stats(profiler, stream=f)
            ps.strip_dirs().sort_stats('cumulative').print_stats(10)
        print("Profiling results saved to profile_results.prof")

def initialize_app():
    try:
        app = App()
        app.initialize()
        return app
    except Exception as e:
        logging.error(f"Application initialization failed: {e}")
        raise

@app.get("/cached-data")
async def get_cached_data():
    """Endpoint to demonstrate caching with Redis."""
    cache_key = "some_unique_key"
    cached_data = redis_client.get(cache_key)
    if cached_data:
        return {"data": cached_data.decode('utf-8')}
    
    # Simulate data retrieval
    data = "This is some data to cache"
    redis_client.setex(cache_key, 3600, data)  # Cache for 1 hour
    return {"data": data}

if __name__ == "__main__":
    main()