FROM node:18-alpine

WORKDIR /app

# Copy package and server files
COPY package.json ./
COPY server-http.js ./

# Install dependencies
RUN npm install

# Set environment variables
ENV CONTEXT_PATH=/data/context
ENV PORT=3004

# Create data directory
RUN mkdir -p /data/context

# Expose port
EXPOSE 3004

# Start the server
CMD ["node", "server-http.js"]