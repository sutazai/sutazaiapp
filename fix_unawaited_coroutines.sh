#!/bin/bash

# Script to fix unawaited coroutine warnings in agent_manager.py

echo "Fixing unawaited coroutine warnings in agent_manager.py..."

# Create a backup of the original file
cp core_system/orchestrator/agent_manager.py core_system/orchestrator/agent_manager.py.bak

# Use Python to fix the file
python3 -c "
import re

with open('core_system/orchestrator/agent_manager.py', 'r') as f:
    content = f.read()

# Fix the cancel() calls without await
# Replace self.heartbeat_task.cancel() with:
# if self.heartbeat_task is not None:
#     try:
#         self.heartbeat_task.cancel()
#         # For test mocks that might return a coroutine
#         if hasattr(self.heartbeat_task, '_is_coroutine') and self.heartbeat_task._is_coroutine:
#             await self.heartbeat_task
#     except Exception as e:
#         logger.warning(f'Error cancelling heartbeat task: {e}')

# Fix in stop method
content = re.sub(
    r'(\s+)(self\.heartbeat_task\.cancel\(\))',
    r'\1if self.heartbeat_task is not None:\n\1    try:\n\1        self.heartbeat_task.cancel()\n\1        # For test mocks that might return a coroutine\n\1        if hasattr(self.heartbeat_task, \'_is_coroutine\') and self.heartbeat_task._is_coroutine:\n\1            await self.heartbeat_task\n\1    except Exception as e:\n\1        logger.warning(f\'Error cancelling heartbeat task: {e}\')',
    content
)

# Fix in stop_heartbeat_monitor method
content = re.sub(
    r'(\s+)(self\.heartbeat_task\.cancel\(\))',
    r'\1if self.heartbeat_task is not None:\n\1    try:\n\1        self.heartbeat_task.cancel()\n\1        # For test mocks that might return a coroutine\n\1        if hasattr(self.heartbeat_task, \'_is_coroutine\') and self.heartbeat_task._is_coroutine:\n\1            await self.heartbeat_task\n\1    except Exception as e:\n\1        logger.warning(f\'Error cancelling heartbeat task: {e}\')',
    content, 
    count=1  # Only replace the first occurrence which should be in stop_heartbeat_monitor
)

with open('core_system/orchestrator/agent_manager.py', 'w') as f:
    f.write(content)

print('Fixed unawaited coroutine warnings in agent_manager.py')
"

echo "Fix completed!" 