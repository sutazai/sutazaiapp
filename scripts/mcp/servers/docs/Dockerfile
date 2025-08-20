FROM node:18-alpine

WORKDIR /app

COPY package.json ./
COPY server-http.js ./

RUN npm install

ENV DOCS_PATH=/data/docs
ENV PORT=3017

RUN mkdir -p /data/docs

EXPOSE 3017

CMD ["node", "server-http.js"]