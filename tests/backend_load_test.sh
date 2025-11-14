#!/bin/bash
#
# Backend API Load Testing Script
# Tests various endpoints with ApacheBench
#

set -e

BACKEND_URL="http://localhost:10200"
CONCURRENCY=50
REQUESTS=1000

echo "================================================================================"
echo "🚀 BACKEND API LOAD TEST"
echo "================================================================================"
echo "Configuration:"
echo "  - Backend URL: $BACKEND_URL"
echo "  - Total requests: $REQUESTS"
echo "  - Concurrent connections: $CONCURRENCY"
echo "================================================================================"
echo ""

# Test 1: Health endpoint (GET, no auth)
echo "📊 Test 1: Health Endpoint (GET /health)"
echo "--------------------------------------------------------------------------------"
ab -n $REQUESTS -c $CONCURRENCY -k $BACKEND_URL/health 2>&1 | grep -E "(Complete|Failed|Time per|Requests per|50%|95%|99%)"
echo ""

# Test 2: Root endpoint (GET)
echo "📊 Test 2: Root Endpoint (GET /)"
echo "--------------------------------------------------------------------------------"
ab -n $REQUESTS -c $CONCURRENCY -k $BACKEND_URL/ 2>&1 | grep -E "(Complete|Failed|Time per|Requests per|50%|95%|99%)"
echo ""

# Test 3: Services endpoint (GET, may require auth)
echo "📊 Test 3: Services Endpoint (GET /services)"
echo "--------------------------------------------------------------------------------"
ab -n 500 -c 25 -k $BACKEND_URL/services 2>&1 | grep -E "(Complete|Failed|Time per|Requests per|50%|95%|99%)"
echo ""

# Test 4: System Info endpoint (GET)
echo "📊 Test 4: System Info Endpoint (GET /system/info)"
echo "--------------------------------------------------------------------------------"
ab -n 500 -c 25 -k $BACKEND_URL/system/info 2>&1 | grep -E "(Complete|Failed|Time per|Requests per|50%|95%|99%)"
echo ""

# Summary
echo "================================================================================"
echo "✅ LOAD TEST COMPLETE"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  - All endpoints tested with high concurrency"
echo "  - Check above results for p50/p95/p99 latencies"
echo "  - Failed requests should be 0 for production readiness"
echo ""
echo "💡 Recommendations:"
echo "  - p95 < 100ms: Excellent"
echo "  - p95 < 500ms: Good"
echo "  - p95 > 1000ms: Needs optimization"
echo ""
echo "================================================================================"
