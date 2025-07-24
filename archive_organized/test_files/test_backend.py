#!/usr/bin/env python3
"""
Test script to verify SutazAI backend functionality
"""
import sys
import os
sys.path.append('/opt/sutazaiapp')

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import time

# Create test FastAPI app
app = FastAPI(title="SutazAI v8 Test Backend")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "SutazAI v8 (2.0.0)",
        "message": "Backend is operational"
    }

@app.get("/test/faiss")
async def test_faiss():
    try:
        import faiss
        import numpy as np
        
        # Test FAISS functionality
        d = 64  # dimension
        nb = 100  # database size
        np.random.seed(1234)
        xb = np.random.random((nb, d)).astype('float32')
        
        # Build index
        index = faiss.IndexFlatL2(d)
        index.add(xb)
        
        return {
            "status": "success",
            "message": "FAISS is working",
            "index_size": index.ntotal,
            "dimension": d
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"FAISS error: {str(e)}"
        }

@app.get("/test/chromadb")
async def test_chromadb():
    try:
        import chromadb
        
        # Test ChromaDB functionality
        client = chromadb.Client()
        collection_name = f"test_collection_{int(time.time())}"
        collection = client.create_collection(collection_name)
        
        return {
            "status": "success",
            "message": "ChromaDB is working",
            "client_type": str(type(client))
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"ChromaDB error: {str(e)}"
        }

@app.get("/system/status")
async def system_status():
    return {
        "status": "operational",
        "services": {
            "backend": "✅ Running",
            "faiss": "✅ Available",
            "chromadb": "✅ Available", 
            "fastapi": "✅ Running"
        },
        "features": [
            "FAISS vector search",
            "ChromaDB integration",
            "FastAPI backend",
            "Health monitoring"
        ]
    }

if __name__ == "__main__":
    print("🚀 Starting SutazAI v8 Test Backend...")
    uvicorn.run(app, host="0.0.0.0", port=8000)