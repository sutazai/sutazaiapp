#!/bin/bash
# Container validation script
set -e

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