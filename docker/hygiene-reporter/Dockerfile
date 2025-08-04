FROM nginx:alpine

# Copy custom nginx config
COPY docker/hygiene-reporter/nginx.conf /etc/nginx/conf.d/default.conf

# Create directory for reports
RUN mkdir -p /usr/share/nginx/html

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/health || exit 1

EXPOSE 80