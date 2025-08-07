FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    g++ \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    faiss-cpu \
    fastapi \
    uvicorn \
    numpy \
    requests \
    psutil

# Create application structure
COPY . .

# Health check endpoint
RUN echo 'from fastapi import FastAPI\nfrom fastapi.responses import JSONResponse\nimport faiss\nimport numpy as np\nimport uvicorn\n\napp = FastAPI(title="FAISS Vector Database Service")\n\n# Initialize FAISS index\nindex = None\ndimension = 384  # Default embedding dimension\n\n@app.get("/health")\nasync def health_check():\n    return {"status": "healthy", "service": "faiss-vector-db"}\n\n@app.post("/create_index")\nasync def create_index(dim: int = 384):\n    global index, dimension\n    dimension = dim\n    index = faiss.IndexFlatL2(dimension)\n    return {"status": "index_created", "dimension": dimension}\n\n@app.post("/add_vectors")\nasync def add_vectors(vectors: list):\n    global index\n    if index is None:\n        index = faiss.IndexFlatL2(len(vectors[0]))\n    \n    vectors_array = np.array(vectors, dtype=np.float32)\n    index.add(vectors_array)\n    return {"status": "vectors_added", "count": len(vectors)}\n\n@app.post("/search")\nasync def search_vectors(query_vector: list, k: int = 5):\n    global index\n    if index is None:\n        return {"error": "Index not initialized"}\n    \n    query_array = np.array([query_vector], dtype=np.float32)\n    distances, indices = index.search(query_array, k)\n    \n    return {\n        "distances": distances[0].tolist(),\n        "indices": indices[0].tolist()\n    }\n\n@app.get("/info")\nasync def get_info():\n    global index, dimension\n    return {\n        "dimension": dimension,\n        "total_vectors": index.ntotal if index else 0,\n        "index_type": "IndexFlatL2"\n    }\n\nif __name__ == "__main__":\n    uvicorn.run(app, host="0.0.0.0", port=8000)' > faiss_service.py

EXPOSE 8000


# Run as non-root user
USER nobody

# Run as non-root user
USER nobody
CMD ["python", "faiss_service.py"]