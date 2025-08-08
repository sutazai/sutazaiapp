# Multi-stage Ollama service optimized for CPU
# Stage 1: Build environment
FROM ollama/ollama:latest as base

# Stage 2: Optimized runtime
FROM ubuntu:22.04

# Security: Create ollama user
RUN groupadd -r ollama && useradd -r -g ollama -s /bin/false ollama

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/*

# Copy Ollama binary from base image
COPY --from=base /bin/ollama /bin/ollama

# Create directories with proper permissions
RUN mkdir -p /root/.ollama /app/models && \
    chown -R ollama:ollama /root/.ollama /app/models

# Environment variables for CPU optimization
ENV OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_MODELS=/app/models \
    OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_MAX_LOADED_MODELS=1 \
    OLLAMA_FLASH_ATTENTION=false \
    OLLAMA_LLM_LIBRARY=cpu

# Create configuration for CPU-optimized models
RUN echo '{\
  "num_ctx": 2048,\
  "num_batch": 1,\
  "num_gqa": 1,\
  "num_gpu": 0,\
  "main_gpu": 0,\
  "low_vram": true,\
  "f16_kv": false,\
  "logits_all": false,\
  "vocab_only": false,\
  "use_mmap": true,\
  "use_mlock": false,\
  "num_thread": 8\
}' > /root/.ollama/config.json && \
    chown ollama:ollama /root/.ollama/config.json

# Switch to non-root user
USER ollama

# Expose port
EXPOSE 11434

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:11434/api/tags || exit 1

# Volume for model persistence
VOLUME ["/app/models"]

# Run Ollama

# Run as non-root user
USER nobody

# Run as non-root user
USER nobody

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app && chown -R appuser:appuser /app
USER appuser

CMD ["ollama", "serve"]