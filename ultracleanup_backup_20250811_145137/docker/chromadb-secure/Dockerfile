FROM chromadb/chroma:0.5.0

# Create non-root user for ChromaDB
# Using UID/GID 1003 to avoid conflicts
RUN groupadd -r chroma -g 1003 && \
    useradd -r -g chroma -u 1003 -m -d /home/chroma chroma

# Create necessary directories with proper permissions
RUN mkdir -p /chroma/data \
             /home/chroma/.cache \
             /home/chroma/.chroma && \
    chown -R chroma:chroma /chroma /home/chroma

# Set working directory
WORKDIR /chroma

# Switch to chroma user
USER chroma

# ChromaDB default port
EXPOSE 8000

# Environment variables for chroma user
ENV HOME=/home/chroma \
    CHROMA_SERVER_HOST=0.0.0.0 \
    CHROMA_SERVER_HTTP_PORT=8000 \
    PERSIST_DIRECTORY=/chroma/data

# Use default entrypoint from base image