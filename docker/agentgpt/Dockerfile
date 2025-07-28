FROM node:18-alpine

WORKDIR /app

# Install git and build dependencies
RUN apk add --no-cache git python3 make g++

# Clone AgentGPT
RUN git clone https://github.com/reworkd/AgentGPT.git .

# Change to the next directory where package.json is located
WORKDIR /app/next

# Install dependencies
RUN npm install

# Set default environment variables for build
ENV DATABASE_URL="sqlite:///tmp/agentgpt.db"
ENV NEXTAUTH_SECRET="default-secret-for-build"
ENV NEXTAUTH_URL="http://localhost:3000"

# Build the application
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
