#!/bin/bash
set -e


# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

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