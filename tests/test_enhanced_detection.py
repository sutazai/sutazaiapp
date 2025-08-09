#!/usr/bin/env python3
"""Test enhanced container detection"""

import subprocess
from typing import Dict, Any, Optional

def _parse_container_status(name: str, status_full: str, ports: str) -> Dict[str, Any]:
    """Parse container status information"""
    # Extract simple status from full status string
    if 'Up' in status_full:
        if 'unhealthy' in status_full:
            status = 'unhealthy'
        elif 'health: starting' in status_full:
            status = 'starting'
        elif 'healthy' in status_full:
            status = 'healthy'
        else:
            status = 'running'
    elif 'Restarting' in status_full:
        status = 'restarting'
    elif 'Exited' in status_full:
        status = 'exited'
    else:
        status = 'unknown'
    
    port_list = []
    if ports:
        # Split multiple port mappings
        port_mappings = ports.split(', ')
        port_list = [p.strip() for p in port_mappings if p.strip()]
    
    return {
        'name': name,
        'status': status,
        'status_full': status_full,
        'ports': port_list
    }

def _get_container_info_enhanced(agent_id: str) -> Optional[Dict[str, Any]]:
    """Get Docker container information for an agent with enhanced matching"""
    try:
        # Method 1: Try exact container name patterns
        exact_patterns = [
            f'sutazai-{agent_id}',
            f'{agent_id}',
            f'sutazaiapp-{agent_id}',
            f'{agent_id}-1'
        ]
        
        for container_name in exact_patterns:
            result = subprocess.run(
                ['docker', 'ps', '-a', '--filter', f'name=^/{container_name}$', 
                 '--format', '{{.Names}}\t{{.Status}}\t{{.Ports}}'],
                capture_output=True, text=True, timeout=3
            )
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            name = parts[0]
                            status_full = parts[1]
                            ports = parts[2] if len(parts) > 2 else ''
                            
                            return _parse_container_status(name, status_full, ports)
        
        # Method 2: Fuzzy matching for known mismatched cases
        fuzzy_mappings = {
            'ollama-integration-specialist': 'sutazai-ollama',
            'semgrep-security-analyzer': 'sutazai-security-pentesting-specialist',
            'senior-ai-engineer': 'sutazai-senior-engineer'
        }
        
        if agent_id in fuzzy_mappings:
            target_container = fuzzy_mappings[agent_id]
            result = subprocess.run(
                ['docker', 'ps', '-a', '--filter', f'name=^/{target_container}$', 
                 '--format', '{{.Names}}\t{{.Status}}\t{{.Ports}}'],
                capture_output=True, text=True, timeout=3
            )
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            name = parts[0]
                            status_full = parts[1]
                            ports = parts[2] if len(parts) > 2 else ''
                            
                            return _parse_container_status(name, status_full, ports)
        
        # Method 3: Intelligent fuzzy search for remaining cases
        agent_words = agent_id.replace('-', ' ').split()
        if len(agent_words) > 1:
            # Get all sutazai containers and find best match
            result = subprocess.run(
                ['docker', 'ps', '-a', '--filter', 'name=sutazai-', 
                 '--format', '{{.Names}}\t{{.Status}}\t{{.Ports}}'],
                capture_output=True, text=True, timeout=3
            )
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                best_match = None
                best_score = 0
                
                for line in lines:
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            container_name = parts[0]
                            container_clean = container_name.replace('sutazai-', '').replace('-', ' ')
                            
                            # Calculate match score
                            score = 0
                            for word in agent_words:
                                if len(word) > 3 and word in container_clean:
                                    score += len(word)
                            
                            # Require minimum meaningful match
                            if score > best_score and score >= 6:
                                best_score = score
                                best_match = parts
                
                if best_match:
                    name = best_match[0]
                    status_full = best_match[1]
                    ports = best_match[2] if len(best_match) > 2 else ''
                    
                    return _parse_container_status(name, status_full, ports)
        
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_enhanced_detection():
    """Test enhanced detection for problematic agents"""
    test_agents = [
        'document-knowledge-manager',
        'ollama-integration-specialist', 
        'code-generation-improver',
        'semgrep-security-analyzer',
        'senior-ai-engineer',
        'hardware-resource-optimizer'
    ]
    
    print("=== Enhanced Container Detection Test ===\n")
    
    for agent_id in test_agents:
        print(f"Testing: {agent_id}")
        container_info = _get_container_info_enhanced(agent_id)
        
        if container_info:
            print(f"  ✓ Found: {container_info['name']}")
            print(f"    Status: {container_info['status']} ({container_info['status_full']})")
            if container_info['ports']:
                print(f"    Ports: {container_info['ports']}")
        else:
            print(f"  ✗ Not found")
        print()

if __name__ == "__main__":
    test_enhanced_detection()