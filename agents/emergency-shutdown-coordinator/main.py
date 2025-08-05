#!/usr/bin/env python3
"""
Main entry point wrapper for emergency-shutdown-coordinator
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Try to import the FastAPI app
app = None

# Try different import patterns
try:
    # First try to import from agent module
    from agent import app
    print("Loaded app from agent module")
except ImportError:
    try:
        # Try to import everything from agent
        from agent import *
        if 'app' in locals():
            print("Found app in agent module globals")
        else:
            # Create a default app if none exists
            from fastapi import FastAPI
            app = FastAPI(title="Emergency Shutdown Coordinator")
            print("Created default FastAPI app")
    except ImportError:
        # If no agent module, create a basic app
        from fastapi import FastAPI
        from datetime import datetime
        
        app = FastAPI(
            title="Emergency Shutdown Coordinator",
            description="Agent service",
            version="1.0.0"
        )
        
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "agent": "emergency-shutdown-coordinator",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @app.get("/")
        async def root():
            return {
                "agent": "emergency-shutdown-coordinator",
                "status": "running"
            }
        
        print("Created basic FastAPI app with health endpoint")

# Ensure app is available at module level
if app is None:
    raise RuntimeError("Could not create or import FastAPI app")

# Run if called directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
