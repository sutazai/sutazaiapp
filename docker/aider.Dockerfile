# Dockerfile for Aider
# ----------------------

# Use a slim Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Aider
RUN pip install aider-chat

# Set up a non-root user
RUN useradd -ms /bin/bash aider
USER aider
WORKDIR /home/aider

# Set the entrypoint
ENTRYPOINT ["aider"]
