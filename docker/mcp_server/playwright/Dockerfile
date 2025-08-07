FROM mcr.microsoft.com/playwright:v1.54.2-jammy

WORKDIR /app
COPY . .

RUN npm install && npx playwright install --with-deps
CMD ["npx", "playwright", "test"]
