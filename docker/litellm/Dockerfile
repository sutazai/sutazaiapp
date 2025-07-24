FROM python:3.11-slim

WORKDIR /app

# Install LiteLLM with all features
RUN pip install --no-cache-dir 'litellm[proxy,extra_proxy_features]' redis

# Create a default config
RUN mkdir -p /app/config

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:4000/health').raise_for_status()"

EXPOSE 4000

# Run with config file if provided, otherwise with default settings
CMD ["litellm", "--host", "0.0.0.0", "--port", "4000", "--telemetry", "false"]