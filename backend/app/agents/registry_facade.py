"""
Registry Facade

Provides a unified import point for registry functionality while preserving
the two existing roles:
- Process/Lifecycle: backend.app.services.agent_registry.AgentRegistry
- In-memory/API:    backend.app.agents.registry.AgentRegistry

New code should import from this module to select the appropriate registry.
This is non-destructive and maintains backwards compatibility.
"""
from __future__ import annotations

from typing import Optional

try:
    # Prefer explicit module paths
    from backend.app.services.agent_registry import AgentRegistry as ProcessRegistry
except Exception:  # pragma: no cover
    ProcessRegistry = None  # type: ignore

try:
    from backend.app.agents.registry import AgentRegistry as MemoryRegistry
except Exception:  # pragma: no cover
    MemoryRegistry = None  # type: ignore


def get_process_registry() -> Optional["ProcessRegistry"]:
    return ProcessRegistry() if ProcessRegistry else None


def get_memory_registry() -> Optional["MemoryRegistry"]:
    return MemoryRegistry() if MemoryRegistry else None


class RegistryFacade:
    """Facade that exposes a minimal common subset across both registries."""

    def __init__(self):
        self.process = get_process_registry()
        self.memory = get_memory_registry()

    # Common-like operations guarded by presence
    async def get_agent_status(self):
        if self.process and hasattr(self.process, "get_agent_status"):
            return await self.process.get_agent_status()
        if self.memory and hasattr(self.memory, "list_agents"):
            agents = self.memory.list_agents()
            return {
                "total_agents": len(agents),
                "status_breakdown": {},
                "agents": [a.__dict__ if hasattr(a, "__dict__") else a for a in agents],
            }
        return {"total_agents": 0, "status_breakdown": {}, "agents": []}

