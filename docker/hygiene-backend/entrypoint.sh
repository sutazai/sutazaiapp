#!/bin/bash
set -e

echo "🚀 Starting Hygiene Monitoring Backend..."

# Wait for database
echo "⏳ Waiting for PostgreSQL..."
while ! curl -s http://postgres:5432 > /dev/null 2>&1; do
    sleep 1
done
echo "✅ PostgreSQL is ready"

# Run database migrations if needed
echo "🗄️ Running database setup..."

# Start the application
echo "🎯 Starting backend server..."
exec python app.py