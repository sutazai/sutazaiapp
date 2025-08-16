# Files MCP Service
FROM sutazaiapp/mcp-base:latest

# Copy service-specific files
COPY . /opt/mcp/service/

# Set service-specific environment
ENV MCP_SERVICE_NAME=files
ENV MCP_SERVICE_PORT=11101

WORKDIR /opt/mcp
CMD ["python3", "mcp_http_server.py"]