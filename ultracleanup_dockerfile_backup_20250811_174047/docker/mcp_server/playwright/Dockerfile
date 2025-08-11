FROM mcr.microsoft.com/playwright:v1.54.2-jammy

WORKDIR /app
COPY . .

RUN npm install && npx playwright install --with-deps

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app && chown -R appuser:appuser /app
USER appuser

CMD ["npx", "playwright", "test"]
