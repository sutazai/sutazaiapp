#!/bin/bash
# Run load tests on the system
ab -n 1000 -c 100 http://localhost:8000/health
echo "Load test completed!" 