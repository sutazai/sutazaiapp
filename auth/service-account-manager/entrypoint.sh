#!/bin/bash

set -e

echo "Starting SutazAI Service Account Manager..."

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

echo "Waiting for Vault..."
while ! curl -f http://vault:8200/v1/sys/health; do
  sleep 2
done
echo "Vault is ready!"

# Start the application
echo "Starting Service Account Manager..."
exec python main.py