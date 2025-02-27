#!/bin/bash

# Wait for database and redis to be ready
until nc -z -v -w30 postgres 5432; do
    echo "Waiting for PostgreSQL database connection..."
    sleep 5
done

until nc -z -v -w30 redis 6379; do
    echo "Waiting for Redis connection..."
    sleep 5
done

# Execute the main command
exec "$@" 