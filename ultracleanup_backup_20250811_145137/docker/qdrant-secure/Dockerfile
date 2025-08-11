FROM qdrant/qdrant:v1.9.2

# Create non-root user for Qdrant
# Using UID/GID 1004 to avoid conflicts
RUN groupadd -r qdrant -g 1004 && \
    useradd -r -g qdrant -u 1004 -m -d /home/qdrant qdrant

# Create necessary directories with proper permissions
RUN mkdir -p /qdrant/storage \
             /qdrant/snapshots \
             /home/qdrant && \
    chown -R qdrant:qdrant /qdrant /home/qdrant

# Set working directory
WORKDIR /qdrant

# Switch to qdrant user
USER qdrant

# Qdrant ports
EXPOSE 6333 6334

# Environment variables
ENV HOME=/home/qdrant \
    QDRANT__SERVICE__HTTP_PORT=6333 \
    QDRANT__SERVICE__GRPC_PORT=6334 \
    QDRANT__LOG_LEVEL=INFO

# Use default entrypoint from base image