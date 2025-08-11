#!/usr/bin/env python3
"""
PyTorch ML Framework Service - CPU Version
"""

from fastapi import FastAPI
import uvicorn
from datetime import datetime

app = FastAPI(title="SutazAI PyTorch Service", version="1.0")

@app.get("/")
async def root():
    return {"framework": "PyTorch", "status": "active", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health():
    try:
        import torch
        return {
            "status": "healthy", 
            "framework": "pytorch", 
            "port": 8085,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
    except ImportError:
        return {
            "status": "healthy", 
            "framework": "pytorch", 
            "port": 8085,
            "torch_version": "not_installed",
            "cuda_available": False,
            "device": "cpu"
        }

@app.post("/tensor")
async def tensor_operation(data: dict):
    try:
        import torch
        values = data.get("values", [1.0, 2.0, 3.0])
        tensor = torch.tensor(values)
        result = tensor.square().tolist()
        return {
            "input": values,
            "operation": "square",
            "result": result,
            "framework": "PyTorch"
        }
    except Exception as e:
        return {"error": str(e), "framework": "PyTorch"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8085)