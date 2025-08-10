#!/bin/bash
# Container validation script
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

echo 'Stopping all services...'
docker-compose down

echo 'Testing postgres...'
docker-compose up -d postgres
sleep 10
docker-compose ps postgres | grep 'Up' || exit 1

echo 'Testing redis...'
docker-compose up -d redis
sleep 10
docker-compose ps redis | grep 'Up' || exit 1

echo 'Testing backend...'
docker-compose up -d backend
sleep 10
docker-compose ps backend | grep 'Up' || exit 1

echo 'Testing frontend...'
docker-compose up -d frontend
sleep 10
docker-compose ps frontend | grep 'Up' || exit 1

echo 'Testing ollama...'
docker-compose up -d ollama
sleep 10
docker-compose ps ollama | grep 'Up' || exit 1

echo 'Building autogpt...'
docker-compose build autogpt

echo 'Building crewai...'
docker-compose build crewai

echo 'Building letta...'
docker-compose build letta

echo 'Building aider...'
docker-compose build aider

echo 'All critical containers validated!'