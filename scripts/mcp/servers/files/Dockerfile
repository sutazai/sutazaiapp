FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package.json ./

# Install dependencies
RUN npm install

# Copy server code
COPY server.js ./

# Set environment variables
ENV PROJECT_DIR=/workspace

# Create workspace directory
RUN mkdir -p /workspace

# Expose port for MCP protocol
EXPOSE 3003

# Start the server
CMD ["node", "server.js"]