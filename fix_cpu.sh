#!/bin/bash
ps -eo pid,pcpu,comm | grep python | sort -k2 -r | head -10
