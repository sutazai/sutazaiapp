FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install additional PyTorch ecosystem
RUN pip install --no-cache-dir \
    torchvision \
    torchaudio \
    transformers \
    datasets \
    accelerate \
    diffusers \
    tokenizers \
    safetensors \
    fastapi \
    uvicorn

# Create PyTorch API server
RUN echo 'import torch\nimport uvicorn\nfrom fastapi import FastAPI\nfrom transformers import pipeline\n\napp = FastAPI()\n\n# Initialize models\ndevice = "cuda" if torch.cuda.is_available() else "cpu"\nprint(f"Using device: {device}")\n\n@app.get("/health")\ndef health():\n    return {\n        "status": "healthy",\n        "device": device,\n        "cuda_available": torch.cuda.is_available(),\n        "torch_version": torch.__version__\n    }\n\n@app.post("/generate")\ndef generate(data: dict):\n    try:\n        task = data.get("task", "text-generation")\n        model = data.get("model", "gpt2")\n        text = data.get("text", "")\n        \n        generator = pipeline(task, model=model, device=0 if torch.cuda.is_available() else -1)\n        result = generator(text, max_length=100, num_return_sequences=1)\n        \n        return {"result": result}\n    except Exception as e:\n        return {"error": str(e)}\n\nif __name__ == "__main__":\n    uvicorn.run(app, host="0.0.0.0", port=8085)' > pytorch_server.py

# Create necessary directories
RUN mkdir -p /data /logs /models

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/data/transformers

# Expose port
EXPOSE 8085

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8085/health || exit 1

# Run PyTorch server
CMD ["python", "pytorch_server.py"]