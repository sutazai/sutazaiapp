#!/bin/bash

# Test script to check repository structures

echo "Checking AutoGPT repository structure..."
docker run --rm python:3.11-slim bash -c "
apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1
git clone --quiet https://github.com/Significant-Gravitas/AutoGPT.git /tmp/autogpt
echo 'AutoGPT files:'
ls -la /tmp/autogpt/ | head -15
echo ''
echo 'Checking for installation files:'
ls /tmp/autogpt/ | grep -E '(requirements|setup|pyproject)' || echo 'No standard install files found'
"

echo -e "\n\nChecking GPT-Engineer repository structure..."
docker run --rm python:3.11-slim bash -c "
apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1
git clone --quiet https://github.com/AntonOsika/gpt-engineer.git /tmp/gpt-engineer
echo 'GPT-Engineer files:'
ls -la /tmp/gpt-engineer/ | head -15
echo ''
echo 'Checking for installation files:'
ls /tmp/gpt-engineer/ | grep -E '(requirements|setup|pyproject)' || echo 'No standard install files found'
"