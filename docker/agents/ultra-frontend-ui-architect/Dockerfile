# Ultra Frontend UI Architect - ULTRAORGANIZE + ULTRAPROPERSTRUCTURE
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (Rule 11: Docker Excellence)
RUN useradd -m -u 1000 ultraarch && \
    chown -R ultraarch:ultraarch /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set proper permissions
RUN chown -R ultraarch:ultraarch /app

# Switch to non-root user
USER ultraarch

# Expose port for Ultra Frontend UI Architect
EXPOSE 11201

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:11201/health || exit 1

# Environment variables
ENV PYTHONPATH=/app
ENV ULTRA_MODE=production
ENV LOG_LEVEL=INFO

# Start Ultra Frontend UI Architect service
CMD ["python", "main.py"]