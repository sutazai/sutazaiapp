#!/usr/bin/env python3
"""
JAX Web Interface Service
Provides REST API endpoints for JAX functionality
"""

import os
import jax
import jax.numpy as jnp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Any

app = FastAPI(title="JAX Service", version="1.0.0")

class ComputeRequest(BaseModel):
    operation: str
    data: List[float]
    params: Dict[str, Any] = {}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "JAX ML Framework",
        "status": "running",
        "version": jax.__version__,
        "device": str(jax.devices()[0])
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test basic JAX operation
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x)
        return {
            "status": "healthy",
            "jax_version": jax.__version__,
            "devices": [str(d) for d in jax.devices()],
            "test_result": float(y)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compute")
async def compute(request: ComputeRequest):
    """Perform JAX computations"""
    try:
        data = jnp.array(request.data)
        
        if request.operation == "sum":
            result = jnp.sum(data)
        elif request.operation == "mean":
            result = jnp.mean(data)
        elif request.operation == "std":
            result = jnp.std(data)
        elif request.operation == "dot":
            # Expecting params to have 'other' key with another array
            other = jnp.array(request.params.get('other', []))
            result = jnp.dot(data, other)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown operation: {request.operation}")
        
        return {
            "operation": request.operation,
            "result": float(result),
            "shape": list(data.shape)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info")
async def system_info():
    """Get JAX system information"""
    return {
        "jax_version": jax.__version__,
        "devices": [
            {
                "id": d.id,
                "platform": d.platform,
                "device_kind": d.device_kind,
            }
            for d in jax.devices()
        ],
        "default_backend": jax.default_backend(),
        "numpy_compatible": True
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)