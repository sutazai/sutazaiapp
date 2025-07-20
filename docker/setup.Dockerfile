FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    unzip \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    nodejs \
    npm \
    postgresql-client \
    redis-tools \
    nginx \
    htop \
    iotop \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    jq \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Docker
RUN curl -fsSL https://get.docker.com -o get-docker.sh && \
    sh get-docker.sh && \
    rm get-docker.sh

# Install Docker Compose
RUN curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose && \
    chmod +x /usr/local/bin/docker-compose

# Create sutazai user
RUN useradd -m -s /bin/bash sutazai && \
    usermod -aG docker sutazai && \
    echo "sutazai ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set working directory
WORKDIR /opt/sutazaiapp

# Copy the entire project
COPY . .

# Change ownership to sutazai user
RUN chown -R sutazai:sutazai /opt/sutazaiapp

# Switch to sutazai user
USER sutazai

# Make setup script executable
RUN chmod +x setup_complete_agi_system.sh

# Default command
CMD ["./setup_complete_agi_system.sh"]