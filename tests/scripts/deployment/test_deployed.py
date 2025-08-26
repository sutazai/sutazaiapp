#!/usr/bin/env python3
"""Test why _is_agent_deployed returns False"""

import logging

logger = logging.getLogger(__name__)
import subprocess
import json
from pathlib import Path

def test_is_deployed(agent_id):
    """Test the exact logic from _is_agent_deployed"""
    logger.info(f"\nTesting {agent_id}:")
    
    # Method 1: Check if running as Python process
    agent_dir = Path('/opt/sutazaiapp/agents') / agent_id
    logger.info(f"  Agent dir exists: {agent_dir.exists()}")
    
    if agent_dir.exists():
        # Look for running Python processes with this agent's app.py
        ps_result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True, text=True, timeout=2
        )
        if ps_result.returncode == 0:
            processes = ps_result.stdout.strip().split('\n')
            for proc in processes:
                if 'python' in proc and agent_id in proc and 'app.py' in proc:
                    logger.info(f"  ✓ Found Python process")
                    return True
    
    # Method 2: Check if agent registry communication config exists
    comm_config_path = Path('/opt/sutazaiapp/agents/communication_config.json')
    logger.info(f"  Comm config exists: {comm_config_path.exists()}")
    
    # Method 3: Check if container exists
    name_patterns = [
        f'name={agent_id}',
        f'name=sutazai-{agent_id}',
        f'name=sutazaiapp-{agent_id}',
        f'name={agent_id}-1'
    ]
    
    for pattern in name_patterns:
        result = subprocess.run(
            ['docker', 'ps', '-a', '--filter', pattern, '--format', '{{.Names}}'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            logger.info(f"  ✓ Found container with pattern '{pattern}': {result.stdout.strip()}")
            return True
    
    logger.info(f"  ✗ Not deployed")
    return False

# Test the specific agents the monitor is looking for
test_agents = [
    'document-knowledge-manager',
    'ollama-integration-specialist', 
    'code-generation-improver',
    'semgrep-security-analyzer',
    'senior-ai-engineer',
    'hardware-resource-optimizer'
]

for agent in test_agents:
    test_is_deployed(agent)