# SutazAI Database Services Base Configuration
# Consolidated database utilities and initialization
# Last Updated: 2025-08-20

FROM alpine:3.19

# Install database client tools
RUN apk add --no-cache \
    postgresql-client \
    redis \
    curl \
    bash \
    py3-pip \
    python3

# Install Python database drivers
RUN pip3 install --no-cache-dir \
    psycopg2-binary \
    redis \
    neo4j \
    pymongo

# Create non-root user
RUN addgroup -g 1000 dbuser && \
    adduser -D -u 1000 -G dbuser dbuser

# Create directories for scripts and data
RUN mkdir -p /scripts /data /migrations && \
    chown -R dbuser:dbuser /scripts /data /migrations

# Copy database initialization scripts
COPY --chown=dbuser:dbuser scripts/ /scripts/
COPY --chown=dbuser:dbuser migrations/ /migrations/

# Set working directory
WORKDIR /scripts

# Switch to non-root user
USER dbuser

# Health check script
RUN echo '#!/bin/sh' > /scripts/healthcheck.sh && \
    echo 'pg_isready -h ${POSTGRES_HOST:-postgres} -p ${POSTGRES_PORT:-5432} || exit 1' >> /scripts/healthcheck.sh && \
    echo 'redis-cli -h ${REDIS_HOST:-redis} -p ${REDIS_PORT:-6379} ping || exit 1' >> /scripts/healthcheck.sh && \
    chmod +x /scripts/healthcheck.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD /scripts/healthcheck.sh

# Default command - can be overridden
CMD ["sh", "-c", "echo 'Database utilities container ready'"]