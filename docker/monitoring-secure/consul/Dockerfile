FROM consul:1.18

# Create consul user and group
RUN addgroup -g 1000 consul && \
    adduser -D -s /bin/sh -u 1000 -G consul consul

# Set up directories with proper permissions
RUN mkdir -p /consul/data /consul/config && \
    chown -R consul:consul /consul

# Switch to non-root user
USER consul

# Default command
CMD ["consul", "agent", "-dev", "-client", "0.0.0.0"]