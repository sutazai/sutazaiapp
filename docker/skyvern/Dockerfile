FROM mcr.microsoft.com/playwright/python:v1.46.0-jammy

WORKDIR /app

# Install minimal tools needed for cloning
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone Skyvern
RUN git clone https://github.com/Skyvern-AI/skyvern.git .

# Install project using PEP 517/518 (handles pyproject-based builds)
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir .

# Create startup script
RUN echo '#!/bin/bash\n\
python -m skyvern.app --host 0.0.0.0 --port 8080\n\
' > start.sh && chmod +x start.sh

EXPOSE 8080


# Run as non-root user provided by Playwright image
USER pwuser
CMD ["./start.sh"]
