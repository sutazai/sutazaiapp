#!/bin/bash

# SutazAI Terminal Wrapper
# This script ensures proper execution environment for the terminal

# Set environment variables
export TERM=xterm-256color
export PYTHONPATH=$PYTHONPATH:/opt/sutazaiapp
export SUTAZAI_HOME=/opt/sutazaiapp
export PATH=/opt/sutazaiapp/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH

# Fix tty permissions if needed
if [ -t 0 ]; then
  TTY=$(tty)
  if [ -n "$TTY" ]; then
    chmod +rw "$TTY" 2>/dev/null || true
  fi
fi

# Reset terminal settings
stty sane

# Activate virtual environment if available
if [ -f "/opt/sutazaiapp/venv/bin/activate" ]; then
  source /opt/sutazaiapp/venv/bin/activate
fi

# Start a new shell
if [ -n "$1" ]; then
  exec "$@"
else
  exec $SHELL -l
fi
