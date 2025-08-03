#!/usr/bin/env python3
"""
JAX Web Interface Service
Provides a FastAPI web interface for JAX machine learning operations
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import jax
import jax.numpy as jnp
import flax
import optax
import uvicorn
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="JAX ML Service", version="1.0.0")

class MatrixRequest(BaseModel):
    """Request model for matrix operations"""
    matrix_a: List[List[float]]
    matrix_b: Optional[List[List[float]]] = None
    operation: str = "multiply"

class TrainingRequest(BaseModel):
    """Request model for training operations"""
    model_type: str = "linear"
    learning_rate: float = 0.01
    epochs: int = 100
    data_size: int = 1000

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "JAX Machine Learning", 
        "status": "active",
        "jax_version": jax.__version__,
        "devices": jax.device_count()
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "jax"}

@app.post("/matrix/operation")
async def matrix_operation(request: MatrixRequest):
    """Perform matrix operations using JAX"""
    try:
        matrix_a = jnp.array(request.matrix_a)
        
        if request.operation == "multiply" and request.matrix_b:
            matrix_b = jnp.array(request.matrix_b)
            result = jnp.matmul(matrix_a, matrix_b)
        elif request.operation == "transpose":
            result = jnp.transpose(matrix_a)
        elif request.operation == "inverse":
            result = jnp.linalg.inv(matrix_a)
        else:
            raise HTTPException(status_code=400, detail="Unsupported operation")
        
        return {
            "operation": request.operation,
            "result": result.tolist(),
            "shape": result.shape
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matrix operation failed: {str(e)}")

@app.post("/train/model")
async def train_model(request: TrainingRequest):
    """Train a simple model using JAX/Flax"""
    try:
        # Generate synthetic data
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (request.data_size, 10))
        y = jnp.sum(X, axis=1) + jax.random.normal(key, (request.data_size,)) * 0.1
        
        if request.model_type == "linear":
            # Simple linear regression
            params = jax.random.normal(key, (10,))
            
            def loss_fn(params, X, y):
                predictions = jnp.dot(X, params)
                return jnp.mean((predictions - y) ** 2)
            
            optimizer = optax.adam(request.learning_rate)
            opt_state = optimizer.init(params)
            
            # Training loop
            for epoch in range(min(request.epochs, 100)):  # Limit epochs for demo
                loss, grads = jax.value_and_grad(loss_fn)(params, X, y)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
            
            final_loss = loss_fn(params, X, y)
            
            return {
                "model_type": request.model_type,
                "final_loss": float(final_loss),
                "epochs_trained": min(request.epochs, 100),
                "parameters_shape": params.shape,
                "status": "training_completed"
            }
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/info")
async def get_info():
    """Get JAX system information"""
    return {
        "jax_version": jax.__version__,
        "flax_version": flax.__version__,
        "optax_version": optax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "device_count": jax.device_count(),
        "default_backend": jax.default_backend()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)