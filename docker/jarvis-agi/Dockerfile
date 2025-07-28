FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    ffmpeg \
    alsa-utils \
    portaudio19-dev \
    pulseaudio \
    nodejs \
    npm \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    pandoc \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Clone and setup Dipeshpal's Jarvis AI (Core Framework)
RUN git clone https://github.com/Dipeshpal/Jarvis_AI.git jarvis-core
WORKDIR /app/jarvis-core
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir jarvis-ai

# Clone Microsoft's JARVIS (Advanced Multi-Model Orchestration)
WORKDIR /app
RUN git clone https://github.com/microsoft/JARVIS.git microsoft-jarvis
WORKDIR /app/microsoft-jarvis
RUN pip install --no-cache-dir -r requirements.txt

# Clone DaniloFalcao's Jarvis (Web Interface & File Processing)
WORKDIR /app
RUN git clone https://github.com/danilofalcao/jarvis.git web-jarvis
WORKDIR /app/web-jarvis
RUN npm install
RUN pip install --no-cache-dir flask flask-socketio eventlet

# Clone SreejanPersonal's JARVIS-AGI (Multimodal AGI Features)
WORKDIR /app
RUN git clone https://github.com/SreejanPersonal/JARVIS-AGI.git jarvis-agi-core
WORKDIR /app/jarvis-agi-core
RUN pip install --no-cache-dir -r requirements.txt || echo "Requirements file not found, installing manually..."

# Install additional AI/ML packages for super intelligence
RUN pip install --no-cache-dir \
    transformers \
    torch \
    torchvision \
    torchaudio \
    whisper \
    speech-recognition \
    pyttsx3 \
    opencv-python \
    Pillow \
    numpy \
    pandas \
    scikit-learn \
    tensorflow \
    gradio \
    huggingface-hub \
    openai \
    anthropic \
    langchain \
    langchain-community \
    chromadb \
    faiss-cpu \
    sentence-transformers \
    spacy \
    nltk \
    beautifulsoup4 \
    requests \
    aiohttp \
    websockets \
    fastapi \
    uvicorn \
    streamlit \
    plotly \
    matplotlib \
    seaborn \
    jupyter \
    ipykernel \
    python-dotenv \
    pydantic \
    typing-extensions \
    asyncio \
    concurrent-futures \
    multiprocessing-logging

# Setup workspace directory
WORKDIR /app
RUN mkdir -p workspaces brain engine playground tools logs data models

# Copy our super intelligent Jarvis system
COPY jarvis_super_system.py .
COPY jarvis_config.json .
COPY requirements_super.txt .

# Install super system requirements
RUN pip install --no-cache-dir -r requirements_super.txt

# Set environment variables for optimal performance
ENV PYTHONPATH="/app:/app/jarvis-core:/app/microsoft-jarvis:/app/web-jarvis:/app/jarvis-agi-core"
ENV JARVIS_MODE="super_intelligent"
ENV JARVIS_WORKSPACE="/app/workspaces"
ENV JARVIS_MODELS_PATH="/app/models"
ENV JARVIS_DATA_PATH="/app/data"
ENV JARVIS_LOGS_PATH="/app/logs"

# Expose ports for different services
EXPOSE 8080 8081 8082 8083 8084

# Create startup script
RUN echo '#!/bin/bash\n\
echo "🚀 Starting JARVIS Super Intelligence System..."\n\
echo "📡 Initializing all AI subsystems..."\n\
echo "🧠 Core Framework: Dipeshpal Jarvis AI"\n\
echo "🤖 Multi-Model Orchestration: Microsoft JARVIS"\n\
echo "🌐 Web Interface: DaniloFalcao Jarvis"\n\
echo "🧬 AGI Core: SreejanPersonal JARVIS-AGI"\n\
echo "⚡ Super Intelligence: SutazAI Integration"\n\
echo ""\n\
echo "Starting services in parallel..."\n\
python jarvis_super_system.py\n\
' > /app/start_jarvis.sh && chmod +x /app/start_jarvis.sh

CMD ["/app/start_jarvis.sh"]