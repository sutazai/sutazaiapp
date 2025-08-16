# PostgreSQL MCP Service
FROM sutazaiapp/mcp-base:latest

# Copy service-specific files
COPY . /opt/mcp/service/

# Install PostgreSQL client
USER root
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python dependencies
RUN pip install --no-cache-dir \
    psycopg2-binary \
    sqlalchemy

USER mcp

# Set service-specific environment
ENV MCP_SERVICE_NAME=postgres
ENV MCP_SERVICE_PORT=11100

WORKDIR /opt/mcp
CMD ["python3", "mcp_http_server.py"]