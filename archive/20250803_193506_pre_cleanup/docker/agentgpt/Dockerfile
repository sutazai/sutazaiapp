FROM node:18-alpine

WORKDIR /app

# Install git and build dependencies
RUN apk add --no-cache git python3 make g++

# Clone AgentGPT
RUN git clone https://github.com/reworkd/AgentGPT.git .

# Change to the next directory where package.json is located
WORKDIR /app/next

# Install dependencies
RUN npm install

# Set environment variables for local LLM with Ollama
ENV DATABASE_URL="postgresql://sutazai:OkI2_5Ob_L5bVwIiyyuSBuWQv@postgres:5432/sutazai"
ENV NEXTAUTH_SECRET="sutazai-nextauth-secret-2024-local-llm"
ENV NEXTAUTH_URL="http://localhost:3000"
ENV OPENAI_API_KEY="sk-local-ollama-key"
ENV OPENAI_API_HOST="http://ollama:11434"
ENV OLLAMA_API_BASE="http://ollama:11434/v1"
ENV OLLAMA_MODEL="llama2"

# Create custom configuration for local LLM
RUN echo '{"openai": {"apiKey": "sk-local-ollama-key", "baseURL": "http://ollama:11434/v1", "model": "llama2"}}' > /app/next/local-llm-config.json

# Build the application
RUN npm run build

EXPOSE 3000

# Create startup script that configures local LLM
RUN echo '#!/bin/sh\necho "🚀 Starting AgentGPT with Local LLM (Ollama)..."\necho "📡 Ollama URL: $OLLAMA_API_BASE"\necho "🤖 Model: $OLLAMA_MODEL"\nnpm start' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
