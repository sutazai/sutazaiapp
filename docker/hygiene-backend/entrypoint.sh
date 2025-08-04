#!/bin/bash
set -e

echo "🚀 Starting Hygiene Monitoring Backend..."

# Wait for database
echo "⏳ Waiting for PostgreSQL..."
while ! nc -z hygiene-postgres 5432; do
    sleep 1
done
echo "✅ PostgreSQL is ready"

# Wait for Redis
echo "⏳ Waiting for Redis..."
while ! nc -z hygiene-redis 6379; do
    sleep 1
done
echo "✅ Redis is ready"

# Run database migrations if needed
echo "🗄️ Running database setup..."

# Start the application
echo "🎯 Starting backend server..."
exec python app.py