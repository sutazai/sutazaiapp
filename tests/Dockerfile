FROM python:3.11-slim

WORKDIR /tests

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python test dependencies
COPY requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt

# Copy test files
COPY . .

# Set Python path
ENV PYTHONPATH=/tests:$PYTHONPATH

# Default command
CMD ["pytest", "-v"]