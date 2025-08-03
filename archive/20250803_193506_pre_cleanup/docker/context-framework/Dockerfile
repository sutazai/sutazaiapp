FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/mihaicode/context-engineering-framework.git . || echo "Repository not found"

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    transformers \
    torch \
    numpy \
    pydantic \
    httpx

# Create a simple context framework service if repo doesn't exist
RUN if [ ! -f "main.py" ]; then \
    echo 'from fastapi import FastAPI, HTTPException\n\
from pydantic import BaseModel\n\
from typing import List, Dict, Any\n\
import numpy as np\n\
\n\
app = FastAPI(title="Context Engineering Framework")\n\
\n\
class ContextRequest(BaseModel):\n\
    text: str\n\
    context_window: int = 2048\n\
    overlap: int = 256\n\
\n\
class ContextResponse(BaseModel):\n\
    chunks: List[str]\n\
    embeddings: List[List[float]]\n\
    metadata: Dict[str, Any]\n\
\n\
@app.post("/process", response_model=ContextResponse)\n\
async def process_context(request: ContextRequest):\n\
    """Process text with context engineering"""\n\
    # Simple chunking implementation\n\
    text = request.text\n\
    window = request.context_window\n\
    overlap = request.overlap\n\
    \n\
    chunks = []\n\
    for i in range(0, len(text), window - overlap):\n\
        chunk = text[i:i + window]\n\
        if chunk:\n\
            chunks.append(chunk)\n\
    \n\
    # Simple mock embeddings\n\
    embeddings = [np.random.rand(768).tolist() for _ in chunks]\n\
    \n\
    return ContextResponse(\n\
        chunks=chunks,\n\
        embeddings=embeddings,\n\
        metadata={\n\
            "total_chunks": len(chunks),\n\
            "context_window": window,\n\
            "overlap": overlap\n\
        }\n\
    )\n\
\n\
@app.get("/health")\n\
async def health():\n\
    return {"status": "healthy", "service": "context-framework"}\n' > main.py; \
    fi

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]