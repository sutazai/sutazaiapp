# syntax=docker/dockerfile:1.7

FROM node:20.16.0-alpine3.19 AS runner
WORKDIR /app
ENV CI=1

# Install pinned Playwright locally so imports resolve without a lockfile
RUN --mount=type=cache,target=/root/.npm npm install --no-save --ignore-scripts @playwright/test@1.45.2

# Install lightweight Docker CLI and common tools
RUN apk add --no-cache docker-cli bash coreutils

# Copy tests and config
COPY tests/playwright/playwright.mcp.config.ts ./playwright.mcp.config.ts
COPY tests/playwright/specs ./tests/playwright/specs


# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["node", "./node_modules/@playwright/test/cli.js", "test", "-c", "./playwright.mcp.config.ts"]
