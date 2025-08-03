# Multi-stage Streamlit frontend with optimizations
# Stage 1: Build environment
FROM node:18-alpine as node-builder

WORKDIR /app

# Copy package files if they exist
COPY package*.json ./
RUN npm install --production 2>/dev/null || echo "No package.json found, skipping npm install"

# Stage 2: Python build environment
FROM python:3.11-slim as python-builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python requirements
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Stage 3: Runtime environment
FROM python:3.11-slim

# Security: Create non-root user
RUN groupadd -r streamlit && useradd -r -g streamlit -s /bin/false streamlit

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=python-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy Node.js artifacts if they exist
COPY --from=node-builder /app/node_modules ./node_modules 2>/dev/null || true

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_RUN_ON_SAVE=false

# Create app directories with proper permissions
RUN mkdir -p /app /app/data /app/.streamlit /app/static && \
    chown -R streamlit:streamlit /app

WORKDIR /app

# Copy application code
COPY --chown=streamlit:streamlit . .

# Create Streamlit config
RUN echo '[server]\n\
port = 8501\n\
address = "0.0.0.0"\n\
runOnSave = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
\n\
[theme]\n\
base = "light"\n\
primaryColor = "#FF6B6B"\n\
backgroundColor = "#FFFFFF"\n\
secondaryBackgroundColor = "#F0F2F6"\n\
textColor = "#262730"' > /app/.streamlit/config.toml

# Switch to non-root user
USER streamlit

# Expose port
EXPOSE 8501

# Health check with proper endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application

# Run as non-root user
USER nobody

# Run as non-root user
USER nobody
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]