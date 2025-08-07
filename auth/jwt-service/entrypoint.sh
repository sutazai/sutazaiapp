#!/bin/bash

set -e

echo "Starting SutazAI JWT Authentication Service..."

# Wait for dependencies
echo "Waiting for database..."
while ! nc -z postgres 5432; do
  sleep 1
done
echo "Database is ready!"

echo "Waiting for Redis..."
while ! nc -z redis 6379; do
  sleep 1
done
echo "Redis is ready!"

echo "Waiting for Keycloak..."
while ! curl -f http://keycloak:8080/health/ready; do
  sleep 5
done
echo "Keycloak is ready!"

# Generate JWT keys if they don't exist
if [ ! -f "/app/keys/jwt_private.pem" ]; then
    echo "Generating JWT keys..."
    mkdir -p /app/keys
    openssl genrsa -out /app/keys/jwt_private.pem 2048
    openssl rsa -in /app/keys/jwt_private.pem -pubout -out /app/keys/jwt_public.pem
    chmod 600 /app/keys/jwt_private.pem
    chmod 644 /app/keys/jwt_public.pem
    echo "JWT keys generated!"
fi

# Start the application
echo "Starting JWT service..."
exec python main.py