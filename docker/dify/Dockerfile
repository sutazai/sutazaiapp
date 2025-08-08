FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/*

# Clone Dify
RUN git clone https://github.com/langgenius/dify.git .

# Install Python dependencies
RUN pip install --no-cache-dir -r api/requirements.txt

# Install Node.js dependencies
RUN cd web && npm install

# Create startup script
RUN echo '#!/bin/bash\n\
cd api && python app.py &\n\
cd web && npm start &\n\
wait\n\
' > start.sh && chmod +x start.sh

EXPOSE 5001 3000


# Run as non-root user
USER nobody

# Run as non-root user
USER nobody

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app && chown -R appuser:appuser /app
USER appuser

CMD ["./start.sh"]
