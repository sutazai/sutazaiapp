from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from .api_routes import router as api_router
from .model_server import app as model_app
import uvicorn
import multiprocessing
from hypercorn.config import Config
from hypercorn.asyncio import serve
from .monitoring import METRICS, MetricsMiddleware
from prometheus_client import start_http_server
from config import settings

# Main FastAPI application
app = FastAPI(title="SutazAI Backend")

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Add metrics middleware
app.add_middleware(MetricsMiddleware)

@app.on_event("startup")
async def startup_event():
    """
    Startup event to initialize prometheus metrics endpoint
    """
    start_http_server(port=8000)  # Prometheus metrics endpoint

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Basic health check endpoint to verify application is running.
    
    Returns:
        JSONResponse: A simple health status response
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "healthy",
            "message": "SutazAI backend is up and running"
        }
    )

async def run_servers():
    """
    Run multiple servers in parallel using multiprocessing
    """
    # Model server configuration
    model_config = Config()
    model_config.bind = [f"0.0.0.0:{settings.model_port}"]
    model_config.worker_class = "uvloop"

    # Main API configuration
    api_config = Config()
    api_config.bind = [f"0.0.0.0:{settings.api_port}"]
    api_config.worker_class = "asyncio"

    # Start servers in parallel
    with multiprocessing.Pool(2) as pool:
        pool.starmap(serve, [
            (model_app, model_config),
            (app, api_config)
        ])

def main():
    """
    Main entry point for the SutazAi backend
    """
    uvicorn.run(
        "backend.main:app", 
        host="0.0.0.0", 
        port=settings.api_port, 
        reload=settings.debug
    )

if __name__ == "__main__":
    main() 