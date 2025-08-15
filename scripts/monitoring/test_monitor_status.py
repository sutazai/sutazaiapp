#!/usr/bin/env python3
"""Test monitor agent status display"""

import logging

logger = logging.getLogger(__name__)
import sys
sys.path.append('/opt/sutazaiapp/scripts/monitoring')

from static_monitor import EnhancedMonitor

def test_monitor():
    """Test monitor agent status"""
    logger.info("=== Testing Monitor Agent Status ===\n")
    
    try:
        monitor = EnhancedMonitor()
        agents, healthy_count, total_count = monitor.get_ai_agents_status()
        
        logger.info(f"Total agents: {total_count}")
        logger.info(f"Healthy agents: {healthy_count}")
        logger.info()
        
        logger.info("Agent Status Display:")
        for i, agent_display in enumerate(agents):
            logger.info(f"  {i+1}. {agent_display}")
        
        logger.info()
        logger.info("=== Testing individual container detection ===")
        test_agents = [
            'document-knowledge-manager',
            'ollama-integration-specialist', 
            'code-generation-improver',
            'semgrep-security-analyzer',
            'senior-ai-engineer',
            'hardware-resource-optimizer'
        ]
        
        for agent_id in test_agents:
            is_deployed = monitor._is_agent_deployed(agent_id)
            container_info = monitor._get_container_info(agent_id)
            
            logger.info(f"{agent_id}:")
            logger.info(f"  Deployed: {is_deployed}")
            if container_info:
                logger.info(f"  Container: {container_info['name']} ({container_info['status']})")
            else:
                logger.info(f"  Container: None")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_monitor()