#!/bin/bash
# Comprehensive deployment test

# Test hardware optimization
./test_hardware.sh || {
    echo "❌ Hardware test failed"
    exit 1
}

# Test service deployment
./test_services.sh || {
    echo "❌ Service test failed"
    exit 1
}

    exit 1
}

echo "✅ All tests passed successfully"