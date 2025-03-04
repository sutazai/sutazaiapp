#!/bin/bash

# Script to fix test_sync_exception method in test_sync_manager_complete_coverage.py

echo "Fixing test_sync_exception method in test_sync_manager_complete_coverage.py..."

# Use Python to fix the specific method
python3 -c "
import re

with open('tests/test_sync_manager_complete_coverage.py', 'r') as f:
    content = f.read()

# Find the test_sync_exception method and replace its implementation
pattern = r'(\s+@pytest\.mark\.asyncio\s+async def test_sync_exception\([^)]+\):.*?)(?=\s+@pytest|\s+def|\Z)'
replacement = '''    @pytest.mark.asyncio
    async def test_sync_exception(self, sync_manager):
        """Test the sync method with an exception."""
        with patch.object(sync_manager, "sync_with_server", side_effect=Exception("Test exception")):
            # This should not raise an exception
            await sync_manager.sync()
            assert True  # If we get here, no exception was raised
'''

# Replace the method
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open('tests/test_sync_manager_complete_coverage.py', 'w') as f:
    f.write(content)

print('Fixed test_sync_exception method in test_sync_manager_complete_coverage.py')
"

echo "Fix completed!" 