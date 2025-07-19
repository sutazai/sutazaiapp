FROM node:18-alpine

WORKDIR /app

# Install git and build dependencies
RUN apk add --no-cache git python3 make g++

# Clone AgentGPT
RUN git clone https://github.com/reworkd/AgentGPT.git .

# Install dependencies
RUN npm install

# Build the application
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
EOF < /dev/null
