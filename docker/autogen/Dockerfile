FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Clone AutoGen repository
RUN git clone https://github.com/ag2ai/ag2.git .

# Install dependencies
RUN pip install --no-cache-dir -e .[teachable,lmm,graph,websockets]

# Create necessary directories
RUN mkdir -p /data /logs /workspace

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV AUTOGEN_USE_DOCKER=False

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Create simple web server for AutoGen
RUN echo 'from autogen import ConversableAgent\nimport uvicorn\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get("/health")\ndef health():\n    return {"status": "healthy"}\n\n@app.post("/chat")\ndef chat(message: dict):\n    agent = ConversableAgent("assistant")\n    response = agent.generate_reply([message])\n    return {"response": response}\n\nif __name__ == "__main__":\n    uvicorn.run(app, host="0.0.0.0", port=8080)' > autogen_server.py

# Run AutoGen server
CMD ["python", "autogen_server.py"]