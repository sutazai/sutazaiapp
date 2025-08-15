# Ultra System Architect Docker Container
# Production-ready container for 500-agent coordination

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create non-root user for security
RUN useradd -m -u 1000 ultrauser && chown -R ultrauser:ultrauser /app

# Switch to non-root user
USER ultrauser

# Expose port (using 11200 range for ultra agents)
EXPOSE 11200

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:11200/health').raise_for_status()" || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=11200
ENV REDIS_URL=redis://redis:6379

# Run the application
CMD ["python", "app.py"]