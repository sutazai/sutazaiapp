FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone GPT-Engineer
RUN git clone https://github.com/AntonOsika/gpt-engineer.git .

# Install dependencies
RUN pip install --no-cache-dir -e .

# Create startup script
RUN echo '#!/bin/bash\n\
python -m gpt_engineer.server --host 0.0.0.0 --port 8080\n\
' > start.sh && chmod +x start.sh

EXPOSE 8080

CMD ["./start.sh"]
