FROM nginx:alpine

LABEL maintainer="Sutazai Infrastructure Team"
LABEL description="Reverse Proxy for Hygiene Monitoring System"

# Install curl for health checks
RUN apk add --no-cache curl

# Remove default nginx configuration
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx configuration
COPY docker/nginx/nginx.conf /etc/nginx/nginx.conf
COPY docker/nginx/upstream.conf /etc/nginx/conf.d/upstream.conf

# Create health check endpoint
RUN mkdir -p /usr/share/nginx/html/health
RUN echo '{"status":"healthy","service":"nginx-proxy","timestamp":"'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"}' > /usr/share/nginx/html/health/index.html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app && chown -R appuser:appuser /app
USER appuser

    CMD curl -f http://localhost/health/ || exit 1