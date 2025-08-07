FROM node:18-slim

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

RUN npx playwright install && npx playwright install-deps

EXPOSE 3000

CMD ["node", "index.js"]
