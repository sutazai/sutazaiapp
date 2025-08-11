"""
Core utilities shared across agents.
"""
import os


def get_agent_name() -> str:
    """Determine agent name from environment or container name.

    - Prefer `AGENT_NAME` env var
    - Else derive from `HOSTNAME` by stripping known prefixes/suffixes
    - Fallback to `base-agent`
    """
    agent_name = os.getenv('AGENT_NAME')
    if agent_name:
        return agent_name

    container_name = os.getenv('HOSTNAME', '')
    if container_name.startswith('sutazai-'):
        name = container_name.replace('sutazai-', '')
        for suffix in ('-phase1', '-phase2', '-phase3'):
            name = name.replace(suffix, '')
        return name

    return 'base-agent'

