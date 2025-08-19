# Real MCP Server Implementation
FROM node:20-alpine

WORKDIR /app

# Install MCP SDK and dependencies
RUN npm install -g \
    @modelcontextprotocol/sdk@latest \
    @modelcontextprotocol/server-toolkit@latest \
    typescript \
    ts-node

# Copy server implementation
COPY server.ts ./
COPY tsconfig.json ./
COPY package.json ./

# Install dependencies
RUN npm install

# Build TypeScript
RUN npm run build

# Run the MCP server
CMD ["node", "dist/server.js"]