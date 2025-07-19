FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone Aider
RUN git clone https://github.com/Aider-AI/aider.git .

# Install dependencies
RUN pip install --no-cache-dir -e .

# Create startup script
RUN echo '#!/bin/bash\n\
aider --api --host 0.0.0.0 --port 8080\n\
' > start.sh && chmod +x start.sh

EXPOSE 8080

CMD ["./start.sh"]
