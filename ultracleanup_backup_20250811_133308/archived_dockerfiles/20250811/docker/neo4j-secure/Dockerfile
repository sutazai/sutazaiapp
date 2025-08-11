FROM neo4j:5.13-community

# Neo4j image already has a neo4j user, we just need to ensure proper permissions
# and switch to non-root user

# Ensure neo4j user owns all necessary directories
RUN chown -R neo4j:neo4j /data /logs /conf /ssl /plugins /import /var/lib/neo4j 2>/dev/null || true && \
    chmod 755 /data /logs /conf /ssl /plugins /import 2>/dev/null || true

# Switch to neo4j user (non-root)
USER neo4j

# Neo4j default ports
EXPOSE 7474 7687

# Use the default entrypoint and command from base image