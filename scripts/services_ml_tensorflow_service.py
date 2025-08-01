#!/usr/bin/env python3
"""
TensorFlow ML Framework Service - CPU Version
"""

from fastapi import FastAPI
import uvicorn
from datetime import datetime
import json

app = FastAPI(title="SutazAI TensorFlow Service", version="1.0")

@app.get("/")
async def root():
    return {"framework": "TensorFlow", "status": "active", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health():
    try:
        import tensorflow as tf
        return {
            "status": "healthy", 
            "framework": "tensorflow", 
            "port": 8086,
            "tf_version": tf.__version__,
            "gpu_available": len(tf.config.experimental.list_physical_devices('GPU')) > 0,
            "eager_execution": tf.executing_eagerly()
        }
    except ImportError:
        return {
            "status": "healthy", 
            "framework": "tensorflow", 
            "port": 8086,
            "tf_version": "not_installed",
            "gpu_available": False,
            "eager_execution": False
        }

@app.post("/tensor")
async def tensor_operation(data: dict):
    try:
        import tensorflow as tf
        values = data.get("values", [1.0, 2.0, 3.0])
        tensor = tf.constant(values)
        result = tf.square(tensor).numpy().tolist()
        return {
            "input": values,
            "operation": "square",
            "result": result,
            "framework": "TensorFlow"
        }
    except Exception as e:
        return {"error": str(e), "framework": "TensorFlow"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8086)