FROM python:3.11-slim

WORKDIR /app

# Install essential system dependencies only
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    ffmpeg \
    portaudio19-dev \
    tesseract-ocr \
    libtesseract-dev \
    wget \
    && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/*

# Install core AI/ML packages efficiently in stages
# Stage 1: Core packages
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    websockets==11.0.2 \
    requests==2.31.0 \
    aiohttp==3.9.1 \
    python-dotenv==1.0.0 \
    pydantic==2.5.0

# Stage 2: AI/ML Core
RUN pip install --no-cache-dir \
    torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu \
    torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu \
    torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Stage 3: NLP and Transformers
RUN pip install --no-cache-dir \
    transformers==4.36.0 \
    sentence-transformers==2.2.2 \
    openai==1.3.0 \
    anthropic==0.7.0

# Stage 4: Audio/Speech
RUN pip install --no-cache-dir \
    openai-whisper==20231117 \
    SpeechRecognition==3.10.0 \
    pyttsx3==2.90

# Stage 5: Vision and Processing
RUN pip install --no-cache-dir \
    opencv-python-headless==4.8.1.78 \
    Pillow==10.1.0 \
    pytesseract==0.3.10

# Stage 6: Data Science Core
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.0

# Stage 7: Vector Databases (lightweight clients only)
RUN pip install --no-cache-dir \
    chromadb \
    faiss-cpu==1.7.4

# Stage 8: Web Interface
RUN pip install --no-cache-dir \
    gradio==4.7.1 \
    streamlit==1.28.1 \
    flask==2.3.3 \
    flask-socketio==5.3.6

# Stage 9: Language Processing
RUN pip install --no-cache-dir \
    langchain==0.0.340 \
    langchain-community==0.0.1 \
    spacy==3.7.2 \
    nltk==3.8.1

# Stage 10: Utilities and Extras
RUN pip install --no-cache-dir \
    beautifulsoup4==4.12.2 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    plotly==5.17.0 \
    typing-extensions==4.8.0

# Create workspace and required directories
RUN mkdir -p workspaces brain engine playground tools logs data models

# Copy our optimized super intelligent Jarvis system
COPY jarvis_super_system.py .
COPY jarvis_config.json .

# Set environment variables for optimal performance
ENV PYTHONPATH="/app"
ENV JARVIS_MODE="super_intelligent"
ENV JARVIS_WORKSPACE="/app/workspaces"
ENV JARVIS_MODELS_PATH="/app/models"
ENV JARVIS_DATA_PATH="/app/data"
ENV JARVIS_LOGS_PATH="/app/logs"
ENV TRANSFORMERS_CACHE="/app/models/transformers"
ENV HF_HOME="/app/models/huggingface"

# Expose ports for different services
EXPOSE 8080 8081 8082 8083 8084

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8084/health || exit 1

# Create optimized startup script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "🚀 Starting JARVIS Super Intelligence System v2.0..."\n\
echo "⚡ Optimized for CPU-only deployment with minimal footprint"\n\
echo "🧠 Advanced AI capabilities with efficient resource usage"\n\
echo "📡 Multi-service architecture: API, Web UI, Voice, Vision"\n\
echo "🔧 Integration with Ollama, ChromaDB, FAISS vector stores"\n\
echo ""\n\
exec python jarvis_super_system.py\n\
' > /app/start_jarvis.sh && chmod +x /app/start_jarvis.sh


# Run as non-root user
USER nobody

# Run as non-root user
USER nobody

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app && chown -R appuser:appuser /app
USER appuser

CMD ["/app/start_jarvis.sh"]