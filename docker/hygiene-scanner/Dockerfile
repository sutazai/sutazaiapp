FROM python:3.11-slim

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY docker/hygiene-scanner/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy scanner script
COPY docker/hygiene-scanner/hygiene_scanner.py .
COPY docker/hygiene-scanner/rule_definitions.yaml .

# Create directories
RUN mkdir -p /app/data /app/logs /app/reports

# Run as non-root user
RUN useradd -m -u 1000 scanner && chown -R scanner:scanner /app
USER scanner

ENTRYPOINT ["python", "hygiene_scanner.py"]