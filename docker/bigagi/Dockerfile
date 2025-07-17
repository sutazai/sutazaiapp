FROM node:18-alpine

# Install system dependencies
RUN apk add --no-cache \
    curl \
    wget \
    git \
    python3 \
    py3-pip \
    make \
    g++

# Set work directory
WORKDIR /app

# Clone and build BigAGI
RUN git clone https://github.com/enricoros/big-AGI.git . \
    && npm install \
    && npm run build

# Create necessary directories
RUN mkdir -p /workspace /logs

# Set environment variables
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || exit 1

# Run application
CMD ["npm", "start"]