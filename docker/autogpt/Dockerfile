FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone AutoGPT
RUN git clone https://github.com/Significant-Gravitas/AutoGPT.git .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create startup script
RUN echo '#!/bin/bash\n\
python -m autogpt.app.main --ai-settings ai_settings.yaml\n\
' > start.sh && chmod +x start.sh

EXPOSE 8080

CMD ["./start.sh"]
