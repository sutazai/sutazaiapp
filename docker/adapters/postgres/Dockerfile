FROM sutazaiapp/service-adapter:latest

# Install PostgreSQL client libraries
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy PostgreSQL-specific configuration
COPY postgres_adapter.py adapter.py
COPY postgres_config.yaml config.yaml

# PostgreSQL default port mapping
ENV TARGET_PORT=5432
ENV ADAPTER_PORT=10100

# Add connection pooling support
ENV PGPOOL_ENABLED=true
ENV PGPOOL_SIZE=20