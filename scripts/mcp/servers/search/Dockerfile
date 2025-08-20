FROM node:18-alpine

WORKDIR /app

COPY package.json ./
COPY server-http.js ./

RUN npm install

ENV INDEX_PATH=/data/search
ENV PORT=3006

RUN mkdir -p /data/search

EXPOSE 3006

CMD ["node", "server-http.js"]