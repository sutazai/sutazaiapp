FROM nginx:alpine

LABEL maintainer="Sutazai Frontend Team"
LABEL description="Hygiene Monitoring Dashboard"

# Install curl for health checks
RUN apk add --no-cache curl

# Copy dashboard files
COPY dashboard/hygiene-monitor/ /usr/share/nginx/html/

# Copy nginx configuration
COPY docker/hygiene-dashboard/nginx.conf /etc/nginx/nginx.conf
COPY docker/hygiene-dashboard/default.conf /etc/nginx/conf.d/default.conf

# Copy configuration script
COPY docker/hygiene-dashboard/configure-environment.sh /docker-entrypoint.d/30-configure-environment.sh
RUN chmod +x /docker-entrypoint.d/30-configure-environment.sh

# Create health check endpoint
RUN echo '{"status":"healthy","service":"hygiene-dashboard"}' > /usr/share/nginx/html/health

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1