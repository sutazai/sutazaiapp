FROM rabbitmq:3.12-management-alpine

# RabbitMQ image already has a rabbitmq user (UID 999)
# We need to ensure proper permissions for the Erlang cookie and data directories

# Create necessary directories and fix permissions as root first
RUN mkdir -p /var/lib/rabbitmq /var/log/rabbitmq /etc/rabbitmq && \
    chown -R rabbitmq:rabbitmq /var/lib/rabbitmq /var/log/rabbitmq /etc/rabbitmq && \
    chmod 755 /var/lib/rabbitmq /var/log/rabbitmq /etc/rabbitmq && \
    # Fix Erlang cookie permissions (must be 400 and owned by rabbitmq)
    touch /var/lib/rabbitmq/.erlang.cookie && \
    chown rabbitmq:rabbitmq /var/lib/rabbitmq/.erlang.cookie && \
    chmod 400 /var/lib/rabbitmq/.erlang.cookie

# Switch to rabbitmq user (non-root)
USER rabbitmq

# RabbitMQ default ports
EXPOSE 5672 15672

# Use the default entrypoint and command from base image