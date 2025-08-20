FROM node:18-alpine

WORKDIR /app

COPY package.json ./
COPY server-http.js ./

RUN npm install

ENV STATE_PATH=/data/claude-flow
ENV PORT=3001

RUN mkdir -p /data/claude-flow

EXPOSE 3001

CMD ["node", "server-http.js"]