# SutazAI Knowledge Graph Builder Dockerfile
# Multi-stage build for optimal image size and security

# Build stage
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /build

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Runtime stage
FROM python:3.11-slim AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/home/kgbuilder/.local/bin:$PATH \
    PYTHONPATH=/app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    graphviz \
    libgraphviz-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 kgbuilder && \
    mkdir -p /app /data /logs && \
    chown -R kgbuilder:kgbuilder /app /data /logs

# Switch to non-root user
USER kgbuilder

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder --chown=kgbuilder:kgbuilder /root/.local /home/kgbuilder/.local

# Copy application code
COPY --chown=kgbuilder:kgbuilder . .

# Create necessary directories
RUN mkdir -p /data/knowledge_graph \
             /data/exports \
             /data/backups \
             /logs/knowledge_graph \
             /app/static \
             /app/templates

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8048/api/health || exit 1

# Expose port
EXPOSE 8048

# Default command

# Add CLAUDE.md rules enforcement
COPY --from=builder /opt/sutazaiapp/CLAUDE.md /app/CLAUDE.md
COPY --from=builder /opt/sutazaiapp/.claude/agents/claude_rules_checker.py /app/claude_rules_checker.py

ENV CLAUDE_RULES_PATH=/app/CLAUDE.md
ENV ENFORCE_CLAUDE_RULES=true

# Add rules checking to Python path
ENV PYTHONPATH="/app:${PYTHONPATH}"

CMD ["python", "-m", "api", "--host", "0.0.0.0", "--port", "8048"]

# Labels for metadata
LABEL maintainer="SutazAI Team" \
      version="1.0.0" \
      description="SutazAI Knowledge Graph Builder for codebase hygiene standards" \
      org.opencontainers.image.title="SutazAI Knowledge Graph Builder" \
      org.opencontainers.image.description="Comprehensive knowledge graph system for SutazAI codebase standards" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="SutazAI" \
      org.opencontainers.image.licenses="MIT"