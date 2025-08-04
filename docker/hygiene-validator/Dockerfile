FROM python:3.11-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY docker/hygiene-validator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy validator script
COPY docker/hygiene-validator/rule_validator.py .

# Create directories
RUN mkdir -p /app/rules /app/reports

# Run as non-root
RUN useradd -m -u 1000 validator && chown -R validator:validator /app
USER validator

ENTRYPOINT ["python", "rule_validator.py"]