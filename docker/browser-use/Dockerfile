FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/*

# Clone Browser-Use
RUN git clone https://github.com/browser-use/browser-use.git .

# Install Python dependencies from pyproject.toml
RUN pip install --no-cache-dir -e .

# Create startup script
RUN echo '#!/bin/bash\n\
python -m browser_use --host 0.0.0.0 --port 8080\n\
' > start.sh && chmod +x start.sh

EXPOSE 8080


# Run as non-root user
USER nobody

# Run as non-root user
USER nobody
CMD ["./start.sh"]
