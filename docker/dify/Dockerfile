FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

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

CMD ["./start.sh"]
