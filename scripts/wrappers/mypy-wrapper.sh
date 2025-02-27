#!/bin/bash
# Wrapper for mypy to prevent terminal freezing
ulimit -t 60  # CPU time limit (seconds)
ulimit -v 2000000  # Virtual memory limit (KB)
exec mypy "$@" 2>&1
