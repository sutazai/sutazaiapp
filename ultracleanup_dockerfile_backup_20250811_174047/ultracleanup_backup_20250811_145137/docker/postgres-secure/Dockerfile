FROM postgres:16.3-alpine

# PostgreSQL Alpine already includes a non-root 'postgres' user (UID 70)
# We just need to ensure proper permissions

# Create necessary directories with correct ownership
RUN mkdir -p /var/lib/postgresql/data /var/run/postgresql && \
    chown -R postgres:postgres /var/lib/postgresql /var/run/postgresql && \
    chmod 700 /var/lib/postgresql/data

# Ensure postgres user is used
USER postgres

# PostgreSQL default port
EXPOSE 5432

# Use the default entrypoint and command from base image
# The base image already handles running as postgres user