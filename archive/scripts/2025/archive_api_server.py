from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
import sys
import os
from typing import Optional

# Add project root to sys.path - Keep for potential sub-dependencies
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging (optional, but good practice)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sutazai_api")

# Declare global availability flag first
MODEL_MANAGER_AVAILABLE: bool = False
ModelManager = None

# Attempt to import ModelManager using explicit relative import
try:
    # Use explicit relative import
    from .ai_agents.model_manager import ModelManager
    MODEL_MANAGER_AVAILABLE = True
    logger.info("ModelManager imported successfully.")
except ImportError as e:
    MODEL_MANAGER_AVAILABLE = False
    logger.error(f"Failed to import ModelManager: {e}. API endpoints relying on it will fail.")

# Global variable to hold the manager instance
manager_instance: Optional[ModelManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize ModelManager
    global manager_instance
    if MODEL_MANAGER_AVAILABLE and ModelManager:
        logger.info("Initializing ModelManager...")
        try:
            # Assuming ModelManager needs default config paths relative to project root
            # Default paths from ModelManager.__init__:
            # models_dir: str = "data/models",
            # config_path: str = "config/models.json",
            # cache_dir: str = "data/model_cache",
            # We need to make these paths absolute from the project root
            proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            models_dir_abs = os.path.join(proj_root, "data", "models")
            config_path_abs = os.path.join(proj_root, "config", "models.json")
            cache_dir_abs = os.path.join(proj_root, "data", "model_cache")

            logger.info(f"Using models_dir: {models_dir_abs}")
            logger.info(f"Using config_path: {config_path_abs}")
            logger.info(f"Using cache_dir: {cache_dir_abs}")

            manager_instance = ModelManager(
                models_dir=models_dir_abs,
                config_path=config_path_abs,
                cache_dir=cache_dir_abs
                # Add other params like max_memory_gb, device if needed/configurable
            )
            logger.info("ModelManager initialized.")
        except FileNotFoundError as fnf_error:
             logger.error(f"Initialization Error: Config file/directory not found: {fnf_error}")
             manager_instance = None
        except Exception as init_error:
            logger.error(f"Error initializing ModelManager: {init_error}")
            manager_instance = None
    else:
        logger.warning("ModelManager not available during lifespan startup. Related API endpoints will not function.")
    yield
    # Shutdown
    logger.info("API Server shutting down.")

app = FastAPI(
    title="SutazAI Backend API",
    description="API for interacting with the SutazAI Model Management System.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/api/v1/status")
async def get_status():
    """Returns the status of the API server and ModelManager."""
    return {
        "status": "ok",
        "model_manager_available": MODEL_MANAGER_AVAILABLE,
        "model_manager_initialized": manager_instance is not None
    }

@app.get("/api/v1/models")
async def list_models():
    """Lists available models managed by the ModelManager."""
    if not manager_instance:
        return {"error": "ModelManager not initialized or available"}, 503 # Service Unavailable

    try:
        # Assuming list_models method exists and returns a list or dict
        # Based on search results, it does exist.
        result = manager_instance.list_models()

        # Adapt response based on expected output of list_models
        # The search result from scripts/setup_models.py suggests list_models returns a list.
        # The original controller expected a dict like {"success": bool, "models": list}
        # Let's assume manager_instance.list_models() returns the list directly
        if isinstance(result, list):
             return {"models": result}
        elif isinstance(result, dict) and "success" in result: # Handle potential dict format
             if result.get("success", False):
                 return {"models": result.get("models", [])}
             else:
                  return {"error": result.get("error", "Failed to list models from manager (dict)")}, 500
        else:
            # Fallback if the format is unexpected
            logger.warning(f"Unexpected format from list_models: {type(result)}")
            return {"models": []} # Return empty list? Or error?

    except Exception as e:
        logger.exception("Error listing models")
        return {"error": f"Internal server error: {str(e)}"}, 500

# Add more endpoints here later for other functionalities (e.g., generation)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting SutazAI API server directly...")
    # Run uvicorn programmatically
    uvicorn.run(
        app, # Pass the FastAPI app instance directly
        host="0.0.0.0",
        port=8002,
        log_level="info"
        # Add use_colors=True here if needed for port reuse, though direct run might avoid the issue
    ) 