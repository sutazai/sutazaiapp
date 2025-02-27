#!/bin/bash
# Wrapper for semgrep to prevent terminal freezing
ulimit -t 120  # CPU time limit (seconds)
ulimit -v 2000000  # Virtual memory limit (KB)
exec semgrep "$@" 2>&1
