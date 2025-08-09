FROM ollama/ollama:latest

# Create non-root user for Ollama
# Using UID/GID 1002 to avoid conflicts
RUN groupadd -r ollama -g 1002 2>/dev/null || true && \
    useradd -r -g ollama -u 1002 -m -d /home/ollama ollama 2>/dev/null || true

# Create necessary directories with proper permissions
RUN mkdir -p /home/ollama/.ollama/models \
             /home/ollama/.ollama/logs \
             /tmp/ollama && \
    chown -R ollama:ollama /home/ollama /tmp/ollama

# Copy models if they exist (handled by volume mount usually)
# Set environment variables for ollama user
ENV HOME=/home/ollama \
    OLLAMA_MODELS=/home/ollama/.ollama/models \
    OLLAMA_TMPDIR=/tmp/ollama \
    OLLAMA_HOST=0.0.0.0

# Switch to ollama user
USER ollama

# Ollama default port
EXPOSE 11434

# Run ollama serve
CMD ["ollama", "serve"]