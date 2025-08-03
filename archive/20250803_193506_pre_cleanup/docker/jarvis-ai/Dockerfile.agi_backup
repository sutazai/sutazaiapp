FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    ffmpeg \
    espeak \
    espeak-data \
    libespeak1 \
    libespeak-dev \
    portaudio19-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for JARVIS
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    requests \
    aiohttp \
    websockets \
    speechrecognition \
    pyttsx3 \
    pyaudio \
    transformers \
    torch \
    numpy \
    scipy \
    scikit-learn \
    opencv-python \
    pillow \
    nltk \
    spacy \
    langchain \
    openai \
    anthropic \
    python-multipart \
    python-dotenv \
    psutil \
    schedule \
    asyncio \
    concurrent.futures

# Clone and setup JARVIS repositories
RUN git clone https://github.com/microsoft/JARVIS.git /app/microsoft-jarvis && \
    git clone https://github.com/Dipeshpal/Jarvis_AI.git /app/dipesh-jarvis && \
    git clone https://github.com/danilofalcao/jarvis.git /app/danilo-jarvis

# Copy our enhanced JARVIS service
COPY jarvis_service.py requirements.txt* ./
COPY jarvis_config.json ./

# Install additional requirements if they exist
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models /app/workspace

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV JARVIS_MODE=autonomous
ENV OLLAMA_BASE_URL=http://ollama:11434
ENV BACKEND_URL=http://backend-agi:8000

# Expose port
EXPOSE 8120

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8120/health || exit 1

# Start JARVIS service
CMD ["python", "jarvis_service.py"] 