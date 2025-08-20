FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package.json ./

# Install dependencies
RUN npm install

# Copy server code
COPY server.js ./

# Set environment variables
ENV MEMORY_DB_PATH=/data/memory

# Create data directory
RUN mkdir -p /data/memory

# Expose port for MCP protocol
EXPOSE 3009

# Start the server
CMD ["node", "server.js"]