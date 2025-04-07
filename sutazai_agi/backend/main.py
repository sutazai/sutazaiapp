import logging
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio # Added for async middleware

# Import routers and other necessary components
from .api.endpoints import router as api_router
from sutazai_agi.core.config_loader import get_setting, load_settings, load_agents_config
from sutazai_agi.models.llm_interface import get_llm_interface # To initialize early
from sutazai_agi.memory.vector_store import get_vector_store # To initialize early
from sutazai_agi.agents.agent_manager import get_agent_manager # To initialize early

# Configure logging (uses config_loader's setup)
logger = logging.getLogger(__name__)

# --- Application Lifecycle --- 

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Backend application starting up...")
    try:
        # Load configurations
        load_settings()
        load_agents_config()
        logger.info("Configurations loaded.")
        
        # Initialize core components (LLM, Vector Store, Agent Manager)
        # These get_* functions handle singleton creation and potential errors
        get_llm_interface() 
        get_vector_store()
        get_agent_manager()
        logger.info("Core components initialized.")
        
        logger.info("Backend application startup complete.")
        yield # Application runs here
        
    except Exception as e:
         logger.critical(f"Fatal error during application startup: {e}", exc_info=True)
         # Exit or prevent the app from serving requests if critical components failed
         # FastAPI might handle this automatically if exception propagates
         raise
    finally:
        # Shutdown logic (if any needed)
        logger.info("Backend application shutting down...")
        # Add cleanup tasks here (e.g., close database connections)
        logger.info("Backend application shutdown complete.")

# --- FastAPI App Creation --- 

app = FastAPI(
    title="SutazAI AGI/ASI Backend",
    description="Provides API endpoints for the SutazAI system, including chat, document analysis, code tools, and agent management.",
    version="0.1.0", # Increment as features are added
    lifespan=lifespan # Use the lifespan context manager
)

# --- Middleware Configuration --- 

# CORS (Cross-Origin Resource Sharing)
cors_origins = get_setting("cors_origins", [])
if cors_origins:
    logger.info(f"Configuring CORS for origins: {cors_origins}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins, # List of allowed origins
        allow_credentials=True,
        allow_methods=["*"], # Allow all methods
        allow_headers=["*"], # Allow all headers
    )
else:
     logger.warning("CORS is not configured. API might not be accessible from web UIs on different origins.")

# Basic Request Logging Middleware (Example)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.debug(f"Request received: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.debug(f"Response status code: {response.status_code}")
    return response

# --- API Routers --- 

# Include the main API router
app.include_router(api_router, prefix="/api/v1") # Add version prefix

# --- Root Endpoint --- 

@app.get("/", tags=["General"], summary="Root endpoint for health check")
async def read_root():
    """Provides a simple health check endpoint."""
    logger.debug("Root endpoint / accessed.")
    return {"message": "SutazAI Backend is running."}

# Add other middleware or application setup as needed 