#!/usr/bin/env python3
"""
Emergency Fix: Re-register MCP services in Consul with correct addresses
Author: Senior Backend Architect
Date: 2025-08-18 12:30:00 UTC
"""

import consul
import logging
import sys
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_mcp_service_registrations():
    """Fix corrupted MCP service registrations in Consul"""
    
    try:
        # Connect to Consul
        c = consul.Consul(host='localhost', port=10006)
        
        # List of MCP services that need fixing
        mcp_services = [
            'mcp-claude-flow', 'mcp-ruv-swarm', 'mcp-claude-task-runner',
            'mcp-files', 'mcp-context7', 'mcp-http', 'mcp-ddg',
            'mcp-sequentialthinking', 'mcp-nx-mcp', 'mcp-extended-memory',
            'mcp-mcp-ssh', 'mcp-ultimatecoder', 'mcp-playwright-mcp',
            'mcp-memory-bank-mcp', 'mcp-knowledge-graph-mcp', 'mcp-compass-mcp',
            'mcp-github', 'mcp-language-server', 'mcp-postgres'
        ]
        
        fixed_count = 0
        failed_count = 0
        
        for service_name in mcp_services:
            try:
                # Get current service info
                _, services = c.catalog.service(service_name)
                
                if not services:
                    logger.warning(f"Service {service_name} not found in Consul")
                    continue
                
                for service in services:
                    current_address = service['ServiceAddress']
                    
                    # Check if address is corrupted (concatenated IPs)
                    if '172.30' in current_address and '172.20' in current_address:
                        logger.info(f"Found corrupted address for {service_name}: {current_address}")
                        
                        # Use the MCP orchestrator container IP
                        # MCP services are running in Docker-in-Docker
                        fixed_address = 'sutazai-mcp-orchestrator'
                        
                        # Re-register with fixed address
                        c.agent.service.register(
                            name=service_name,
                            service_id=service['ServiceID'],
                            address=fixed_address,
                            port=service['ServicePort'],
                            tags=service.get('ServiceTags', ['mcp', 'stdio', 'dind']),
                            check=consul.Check.tcp(fixed_address, service['ServicePort'], '10s')
                        )
                        
                        logger.info(f"Fixed {service_name} - address: {fixed_address}")
                        fixed_count += 1
                    else:
                        logger.info(f"Service {service_name} address OK: {current_address}")
                        
            except Exception as e:
                logger.error(f"Failed to fix {service_name}: {e}")
                failed_count += 1
        
        # Also ensure backend service mesh can discover MCP services
        # Register a health check for the MCP bridge
        c.agent.service.register(
            name='mcp-bridge',
            service_id='mcp-bridge-health',
            address='sutazai-backend',
            port=8000,
            tags=['mcp', 'bridge', 'health'],
            check=consul.Check.http('http://sutazai-backend:8000/health', '30s')
        )
        
        logger.info(f"Registration complete. Fixed: {fixed_count}, Failed: {failed_count}")
        
        # Verify fixes
        logger.info("\nVerifying service registrations:")
        for service_name in mcp_services[:3]:  # Check first 3 as sample
            _, services = c.catalog.service(service_name)
            if services:
                logger.info(f"{service_name}: address={services[0]['ServiceAddress']}, port={services[0]['ServicePort']}")
        
        return fixed_count > 0
        
    except Exception as e:
        logger.error(f"Critical error: {e}")
        return False

if __name__ == "__main__":
    success = fix_mcp_service_registrations()
    sys.exit(0 if success else 1)