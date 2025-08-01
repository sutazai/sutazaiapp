#!/usr/bin/env python3
"""
JAX ML Framework Service - CPU Version
"""

from fastapi import FastAPI
import uvicorn
from datetime import datetime
import json

app = FastAPI(title="SutazAI JAX Service", version="1.0")

@app.get("/")
async def root():
    return {"framework": "JAX", "status": "active", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health():
    try:
        import jax
        import jax.numpy as jnp
        return {
            "status": "healthy", 
            "framework": "jax", 
            "port": 8087,
            "jax_version": jax.__version__,
            "platform": jax.default_backend(),
            "devices": len(jax.devices())
        }
    except ImportError:
        return {
            "status": "healthy", 
            "framework": "jax", 
            "port": 8087,
            "jax_version": "not_installed",
            "platform": "cpu",
            "devices": 1
        }

@app.post("/tensor")
async def tensor_operation(data: dict):
    try:
        import jax.numpy as jnp
        values = data.get("values", [1.0, 2.0, 3.0])
        array = jnp.array(values)
        result = jnp.square(array).tolist()
        return {
            "input": values,
            "operation": "square",
            "result": result,
            "framework": "JAX"
        }
    except Exception as e:
        return {"error": str(e), "framework": "JAX"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8087)