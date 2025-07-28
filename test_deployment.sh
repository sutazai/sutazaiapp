#\!/bin/bash
# Test the deployment script with a shorter timeout for validation
echo '🧪 Testing SutazAI Deployment Script - Validation Run'
echo '⏱️ Testing first 5 minutes of deployment process...'
echo ''

timeout 300 bash scripts/deploy_complete_system.sh 2>&1 | tee test_run.log | tail -20

echo ''
echo '📊 Test Results:'
if grep -q 'Hardware Auto-Detection.*completed' test_run.log; then
    echo '✅ Hardware detection: PASSED'
else
    echo '❌ Hardware detection: FAILED or INCOMPLETE'
fi

if grep -q 'Docker environment verified' test_run.log; then
    echo '✅ Docker verification: PASSED'
else
    echo '❌ Docker verification: FAILED or INCOMPLETE'
fi

if grep -q 'Phase.*Executing Super Intelligent Deployment' test_run.log; then
    echo '✅ Main deployment phase reached: PASSED'
else
    echo '❌ Main deployment phase: NOT REACHED'
fi

echo ''
echo '📈 Progress Summary:'
grep -E '(Phase [0-9]:)|(✅.*completed)|(❌)' test_run.log | tail -10

rm -f test_run.log

