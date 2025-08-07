FROM node:18-slim

WORKDIR /app

# Copy package files first to leverage Docker cache
COPY package*.json ./

RUN npm install

# Copy full codebase to /app inside container
COPY .. .

# Install Playwright browsers and dependencies
RUN npx playwright install && npx playwright install-deps

EXPOSE 3000

CMD ["node", "index.js"]
