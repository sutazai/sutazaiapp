FROM node:18-alpine

WORKDIR /usr/src/app

# Install dependencies
RUN apk add --no-cache python3 py3-pip make g++ git

# Copy package files
COPY package.json ./

# Install npm dependencies
RUN npm install --production

# Copy service files
COPY flowise_service.js .

EXPOSE 3000

# Environment variables
ENV FLOWISE_USERNAME=admin
ENV FLOWISE_PASSWORD=sutazai123
ENV PORT=3000

CMD ["npm", "start"]