#!/bin/bash

# Script to generate a report on the current state of the tests
echo "Generating test report..."

echo "1. Test files with potential issues:"
find tests -name "test_*.py" -exec grep -l "@pytest.mark.asyncio" {} \; | sort

echo -e "\n2. Tests with duplicate decorators (these need fixing):"
find tests -name "test_*.py" -exec grep -B 1 -A 1 "@pytest.mark.asyncio" {} \; | grep -A 1 -B 1 "@pytest.mark.asyncio.*@pytest.mark.asyncio" | grep -v -- "--" | sort

echo -e "\n3. Tests with indentation issues (lines not indented properly):"
find tests -name "test_*.py" -exec grep -n "^ *@pytest.mark.asyncio" {} \; | grep -v "    @pytest" | sort

echo -e "\n4. Summary of fixes applied:"
echo "- Fixed indentation and removed duplicate decorators in test_sync_manager_complete_coverage.py"
echo "- Checked agent_manager.py for necessary fixes (already in place)"
echo "- Verified task_queue tests have proper asyncio markers"

echo -e "\nReport generated successfully!" 