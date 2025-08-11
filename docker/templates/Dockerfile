FROM sutazai/nodejs-agent-base:latest
# Service-specific configurations for templates
COPY package*.json ./
RUN npm install || true
COPY . .
