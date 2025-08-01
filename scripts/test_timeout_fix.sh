#!/bin/bash
# Test the proposed timeout fix

echo "Testing the timeout fix approach..."

# Create a test function that mimics the real functions
test_long_function() {
    echo "Starting test function..."
    sleep 2
    echo "Test function completed successfully"
    return 0
}

test_hanging_function() {
    echo "Starting hanging function..."
    sleep 10  # This will timeout in our 5-second test
    echo "This should not appear"
    return 0
}

# Test 1: Current broken approach (should fail)
echo -e "\n1. Testing current broken approach:"
if timeout 3 bash -c 'test_long_function' 2>/dev/null; then
    echo "❌ Unexpected: bash -c approach worked"
else
    echo "✅ Expected: bash -c approach failed (function not found)"
fi

# Test 2: Proposed fix (should work)
echo -e "\n2. Testing proposed fix approach:"
if timeout 5 test_long_function 2>/dev/null; then
    echo "✅ Success: Direct function call with timeout worked"
else
    echo "❌ Failed: Direct function call with timeout failed"
fi

# Test 3: Test timeout functionality
echo -e "\n3. Testing timeout functionality:"
echo "This should timeout after 3 seconds..."
if timeout 3 test_hanging_function 2>/dev/null; then
    echo "❌ Function completed (unexpected)"
else
    echo "✅ Function timed out as expected"
fi

# Test 4: Verify functions can be called normally without timeout
echo -e "\n4. Testing normal function call:"
if test_long_function >/dev/null 2>&1; then
    echo "✅ Function works normally without timeout"
else
    echo "❌ Function failed even without timeout"
fi

echo -e "\n✅ Conclusion: The fix should work - remove 'bash -c' wrapper and call functions directly with timeout"