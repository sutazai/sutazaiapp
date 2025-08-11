FROM redis:7.2-alpine

# Redis Alpine already includes a 'redis' user, but we need to ensure it's used

# Create redis group and user if not exists (Alpine specific)
RUN addgroup -g 999 -S redis 2>/dev/null || true && \
    adduser -u 999 -S -D -H -G redis redis 2>/dev/null || true

# Create data directory with proper permissions
RUN mkdir -p /data && \
    chown -R redis:redis /data && \
    chmod 770 /data

# Switch to redis user
USER redis

# Redis default port
EXPOSE 6379

# Override command to ensure proper permissions
CMD ["redis-server", "--save", "", "--appendonly", "no", "--maxmemory", "512mb", "--maxmemory-policy", "allkeys-lru"]