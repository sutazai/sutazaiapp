FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/*

# Install LangFlow
RUN pip install langflow

# Create startup script
RUN echo '#!/bin/bash\n\
langflow run --host 0.0.0.0 --port 7860\n\
' > start.sh && chmod +x start.sh

EXPOSE 7860


# Run as non-root user
USER nobody

# Run as non-root user
USER nobody
CMD ["./start.sh"]
