# Dockerfile for GPT-Engineer Service
# ------------------------------------

# Use a slim Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install gpt-engineer fastapi uvicorn pydantic

# Copy the service application code
COPY ./gpt_engineer_service /app

# Expose port
EXPOSE 8080

# Set the entrypoint to run as service
ENTRYPOINT ["python", "main.py"]
