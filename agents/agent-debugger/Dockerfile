FROM python:3.12.8-slim-bookworm

# Security: Create non-root user
RUN groupadd -r debugger && useradd -r -g debugger debugger

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set ownership to non-root user
RUN chown -R debugger:debugger /app

# Switch to non-root user
USER debugger

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10205/debug/health || exit 1

# Expose port
EXPOSE 10205

# Start the debugger service
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10205"]