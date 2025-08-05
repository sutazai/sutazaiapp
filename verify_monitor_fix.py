#!/usr/bin/env python3
"""Verify the monitor fix is working"""

import sys
sys.path.insert(0, '/opt/sutazaiapp/scripts/monitoring')

from static_monitor import EnhancedMonitor

# Create monitor instance
monitor = EnhancedMonitor()

# Get AI agents status
agents, healthy, total = monitor.get_ai_agents_status()

print(f"Monitor Results:")
print(f"Total agents shown: {total}")
print(f"Healthy agents: {healthy}")
print(f"Agent status:")
print("-" * 60)

for i, agent_line in enumerate(agents[:20]):  # Show first 20
    print(f"  {i+1}. {agent_line}")