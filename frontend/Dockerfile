FROM python:3.12.8-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Set environment variables
ENV AGENT_ID=senior-frontend-developer
ENV AGENT_NAME="Senior Frontend Developer"
ENV PORT=8542

# Expose port
EXPOSE 8542

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8542/health || exit 1

# Run the application

# Create and use non-root user
RUN adduser --disabled-password --gecos '' --uid 1001 frontend
USER frontend


# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app && chown -R appuser:appuser /app
USER appuser

CMD ["python", "app.py"]
