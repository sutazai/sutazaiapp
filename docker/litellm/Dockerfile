FROM ghcr.io/berriai/litellm:main-latest

# Install additional dependencies
RUN pip install --no-cache-dir \
    psycopg2-binary \
    redis

# Create config directory
RUN mkdir -p /app/config

# Set environment variables
ENV LITELLM_MASTER_KEY=sk-local
ENV LITELLM_LOG=INFO
ENV DATABASE_URL=postgresql://sutazai:sutazai_password@postgres:5432/sutazai
ENV REDIS_URL=redis://:redis_password@redis:6379/0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:4000/health || exit 1

# Expose port
EXPOSE 4000

# Run LiteLLM proxy
CMD ["litellm", "--host", "0.0.0.0", "--port", "4000", "--detailed_debug"]