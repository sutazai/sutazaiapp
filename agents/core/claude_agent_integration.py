"""
Claude Agent Integration System
Rule 14 Compliant - 231 Agent Integration Bridge

This module integrates all 231 Claude agents from /.claude/agents/ with the 
main orchestration system, providing unified access and management.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import hashlib
from datetime import datetime
import asyncio
import importlib.util
import inspect
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClaudeAgentMetadata:
    """Metadata for a Claude agent."""
    name: str
    file_path: str
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    specializations: List[str] = field(default_factory=list)
    system_prompt: str = ""
    examples: List[Dict[str, str]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    last_updated: datetime = field(default_factory=datetime.now)


class ClaudeAgentLoader:
    """
    Loader for Claude agents from markdown files.
    
    This class is responsible for discovering, loading, and parsing
    all 231 Claude agent definitions from the .claude/agents directory.
    """
    
    def __init__(self, agents_dir: str = "/.claude/agents"):
        self.agents_dir = Path(agents_dir)
        self.loaded_agents: Dict[str, ClaudeAgentMetadata] = {}
        self.agent_prompts: Dict[str, str] = {}
        self.agent_index: Dict[str, List[str]] = {}  # capability -> agent names
        
        # Load all agents on initialization
        self._load_all_agents()
    
    def _load_all_agents(self):
        """Load all Claude agents from the directory."""
        if not self.agents_dir.exists():
            # If the directory doesn't exist in the expected location, try the current project
            self.agents_dir = Path("/opt/sutazaiapp/.claude/agents")
            
        if not self.agents_dir.exists():
            logger.warning(f"Claude agents directory not found: {self.agents_dir}")
            return
        
        # Find all .md files in the agents directory
        agent_files = list(self.agents_dir.glob("*.md"))
        logger.info(f"Found {len(agent_files)} Claude agent files")
        
        for agent_file in agent_files:
            try:
                agent_metadata = self._parse_agent_file(agent_file)
                self.loaded_agents[agent_metadata.name] = agent_metadata
                
                # Index by capabilities
                for capability in agent_metadata.capabilities:
                    self.agent_index.setdefault(capability, []).append(agent_metadata.name)
                
                logger.debug(f"Loaded agent: {agent_metadata.name}")
                
            except Exception as e:
                logger.error(f"Failed to load agent from {agent_file}: {str(e)}")
    
    def _parse_agent_file(self, file_path: Path) -> ClaudeAgentMetadata:
        """Parse a Claude agent markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract agent name from filename
        agent_name = file_path.stem
        
        # Parse markdown sections
        metadata = ClaudeAgentMetadata(
            name=agent_name,
            file_path=str(file_path)
        )
        
        # Extract description (usually the first paragraph or header)
        description_match = re.search(r'^#\s+(.+?)$', content, re.MULTILINE)
        if description_match:
            metadata.description = description_match.group(1).strip()
        
        # Extract capabilities (look for lists or specific sections)
        capabilities_section = re.search(r'## Capabilities?\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
        if capabilities_section:
            capabilities = re.findall(r'[-*]\s+(.+?)$', capabilities_section.group(1), re.MULTILINE)
            metadata.capabilities = [cap.strip() for cap in capabilities]
        
        # Extract specializations
        specializations_section = re.search(r'## Specializations?\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
        if specializations_section:
            specializations = re.findall(r'[-*]\s+(.+?)$', specializations_section.group(1), re.MULTILINE)
            metadata.specializations = [spec.strip() for spec in specializations]
        
        # Extract system prompt
        system_prompt_section = re.search(r'## System Prompt\s*\n```\n(.*?)\n```', content, re.DOTALL)
        if system_prompt_section:
            metadata.system_prompt = system_prompt_section.group(1).strip()
        else:
            # If no explicit system prompt, use the entire content as prompt
            metadata.system_prompt = content
        
        # Store the full prompt for later use
        self.agent_prompts[agent_name] = metadata.system_prompt or content
        
        # Extract examples if present
        examples_section = re.search(r'## Examples?\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
        if examples_section:
            # Parse example blocks
            example_blocks = re.findall(r'### (.+?)\n(.*?)(?=\n###|\Z)', examples_section.group(1), re.DOTALL)
            for title, example_content in example_blocks:
                metadata.examples.append({
                    "title": title.strip(),
                    "content": example_content.strip()
                })
        
        # Extract tags (look for hashtags or tag section)
        tags = re.findall(r'#(\w+)', content)
        metadata.tags = list(set(tags))
        
        return metadata
    
    def get_agent(self, agent_name: str) -> Optional[ClaudeAgentMetadata]:
        """Get a specific agent by name."""
        return self.loaded_agents.get(agent_name)
    
    def get_agent_prompt(self, agent_name: str) -> Optional[str]:
        """Get the system prompt for an agent."""
        return self.agent_prompts.get(agent_name)
    
    def list_agents(self) -> List[str]:
        """List all available agent names."""
        return list(self.loaded_agents.keys())
    
    def find_agents_by_capability(self, capability: str) -> List[str]:
        """Find agents with a specific capability."""
        return self.agent_index.get(capability, [])
    
    def search_agents(self, query: str) -> List[ClaudeAgentMetadata]:
        """Search agents by query string."""
        query_lower = query.lower()
        results = []
        
        for agent_name, metadata in self.loaded_agents.items():
            # Search in name, description, capabilities, and tags
            if (query_lower in agent_name.lower() or
                query_lower in metadata.description.lower() or
                any(query_lower in cap.lower() for cap in metadata.capabilities) or
                any(query_lower in tag.lower() for tag in metadata.tags)):
                results.append(metadata)
        
        return results


class ClaudeAgentExecutor:
    """
    Executor for Claude agents.
    
    Provides the runtime environment for executing Claude agents
    with their specific prompts and capabilities.
    """
    
    def __init__(self, loader: ClaudeAgentLoader):
        self.loader = loader
        self.execution_history: List[Dict[str, Any]] = []
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def execute_agent(self, agent_name: str, task: Dict[str, Any], 
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a Claude agent with a specific task.
        
        Args:
            agent_name: Name of the Claude agent
            task: Task specification
            context: Optional execution context
            
        Returns:
            Execution result
        """
        # Get agent metadata
        agent_metadata = self.loader.get_agent(agent_name)
        if not agent_metadata:
            return {
                "status": "error",
                "error": f"Agent {agent_name} not found"
            }
        
        # Get agent prompt
        system_prompt = self.loader.get_agent_prompt(agent_name)
        
        # Create execution session
        session_id = hashlib.md5(f"{agent_name}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        session = {
            "id": session_id,
            "agent": agent_name,
            "task": task,
            "context": context or {},
            "start_time": datetime.now(),
            "status": "running"
        }
        self.active_sessions[session_id] = session
        
        try:
            # Execute the agent (simulated - replace with actual LLM call)
            result = await self._simulate_agent_execution(agent_metadata, task, system_prompt, context)
            
            # Update session
            session["status"] = "completed"
            session["result"] = result
            session["end_time"] = datetime.now()
            session["duration"] = (session["end_time"] - session["start_time"]).total_seconds()
            
            # Record in history
            self._record_execution(session)
            
            return {
                "status": "success",
                "session_id": session_id,
                "agent": agent_name,
                "result": result,
                "duration": session["duration"]
            }
            
        except Exception as e:
            session["status"] = "failed"
            session["error"] = str(e)
            session["end_time"] = datetime.now()
            
            return {
                "status": "error",
                "session_id": session_id,
                "agent": agent_name,
                "error": str(e)
            }
        
        finally:
            # Clean up active session
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
    
    async def _simulate_agent_execution(self, agent: ClaudeAgentMetadata, task: Dict[str, Any],
                                       system_prompt: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate agent execution (replace with actual LLM integration)."""
        # Simulate processing time
        await asyncio.sleep(1.0)
        
        # Generate simulated response based on agent capabilities
        response = {
            "task_id": task.get("id", "unknown"),
            "agent_name": agent.name,
            "agent_capabilities": agent.capabilities,
            "task_description": task.get("description", ""),
            "analysis": f"Task analyzed by {agent.name} with specializations in {', '.join(agent.specializations[:3])}",
            "recommendations": [
                f"Recommendation 1 from {agent.name}",
                f"Recommendation 2 based on {agent.name}'s expertise",
                f"Recommendation 3 utilizing {agent.name}'s capabilities"
            ],
            "confidence": 0.85,
            "notes": f"Executed with system prompt of {len(system_prompt)} characters"
        }
        
        # Add context-specific information
        if context:
            response["context_keys"] = list(context.keys())
        
        return response
    
    def _record_execution(self, session: Dict[str, Any]):
        """Record execution in history."""
        self.execution_history.append({
            "session_id": session["id"],
            "agent": session["agent"],
            "task_id": session["task"].get("id"),
            "status": session["status"],
            "duration": session.get("duration"),
            "timestamp": session["end_time"].isoformat() if "end_time" in session else None
        })
    
    def get_execution_history(self, agent_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get execution history, optionally filtered by agent."""
        if agent_name:
            return [h for h in self.execution_history if h["agent"] == agent_name]
        return self.execution_history


class ClaudeAgentRegistry:
    """
    Central registry for all Claude agents.
    
    Provides unified access to all 231 Claude agents and their capabilities,
    integrating with the main agent registry system.
    """
    
    def __init__(self):
        self.loader = ClaudeAgentLoader()
        self.executor = ClaudeAgentExecutor(self.loader)
        self.capability_map: Dict[str, List[str]] = {}
        self.specialization_map: Dict[str, List[str]] = {}
        
        # Build capability and specialization maps
        self._build_registry_maps()
    
    def _build_registry_maps(self):
        """Build capability and specialization maps for quick lookup."""
        for agent_name, metadata in self.loader.loaded_agents.items():
            # Map capabilities
            for capability in metadata.capabilities:
                self.capability_map.setdefault(capability, []).append(agent_name)
            
            # Map specializations
            for specialization in metadata.specializations:
                self.specialization_map.setdefault(specialization, []).append(agent_name)
    
    def get_all_agents(self) -> Dict[str, ClaudeAgentMetadata]:
        """Get all registered Claude agents."""
        return self.loader.loaded_agents
    
    def get_agent_count(self) -> int:
        """Get total number of registered agents."""
        return len(self.loader.loaded_agents)
    
    def get_capabilities_summary(self) -> Dict[str, int]:
        """Get summary of capabilities across all agents."""
        summary = {}
        for capability, agents in self.capability_map.items():
            summary[capability] = len(agents)
        return dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))
    
    def get_specializations_summary(self) -> Dict[str, int]:
        """Get summary of specializations across all agents."""
        summary = {}
        for specialization, agents in self.specialization_map.items():
            summary[specialization] = len(agents)
        return dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))
    
    def find_expert_agents(self, domain: str, min_count: int = 3) -> List[str]:
        """Find expert agents for a specific domain."""
        domain_lower = domain.lower()
        experts = []
        
        for agent_name, metadata in self.loader.loaded_agents.items():
            relevance_score = 0
            
            # Check specializations
            for spec in metadata.specializations:
                if domain_lower in spec.lower():
                    relevance_score += 2
            
            # Check capabilities
            for cap in metadata.capabilities:
                if domain_lower in cap.lower():
                    relevance_score += 1
            
            # Check description
            if domain_lower in metadata.description.lower():
                relevance_score += 1
            
            if relevance_score >= min_count:
                experts.append((agent_name, relevance_score))
        
        # Sort by relevance score
        experts.sort(key=lambda x: x[1], reverse=True)
        
        return [agent for agent, _ in experts]
    
    async def execute_agent_task(self, agent_name: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with a specific Claude agent."""
        return await self.executor.execute_agent(agent_name, task)
    
    def export_registry(self) -> Dict[str, Any]:
        """Export the complete agent registry as JSON-serializable dict."""
        registry = {
            "version": "1.0.0",
            "total_agents": self.get_agent_count(),
            "agents": {},
            "capabilities": self.get_capabilities_summary(),
            "specializations": self.get_specializations_summary(),
            "generated_at": datetime.now().isoformat()
        }
        
        for agent_name, metadata in self.loader.loaded_agents.items():
            registry["agents"][agent_name] = {
                "name": metadata.name,
                "description": metadata.description,
                "capabilities": metadata.capabilities,
                "specializations": metadata.specializations,
                "tags": metadata.tags,
                "file_path": metadata.file_path,
                "version": metadata.version
            }
        
        return registry


class ClaudeAgentIntegrationBridge:
    """
    Integration bridge between Claude agents and the main orchestration system.
    
    This class provides the bridge between the 231 Claude agents and the
    existing agent infrastructure, enabling seamless integration.
    """
    
    def __init__(self, main_registry_path: str = "/opt/sutazaiapp/agents/agent_registry.json"):
        self.claude_registry = ClaudeAgentRegistry()
        self.main_registry_path = Path(main_registry_path)
        self.unified_registry: Dict[str, Any] = {}
        
        # Load main registry
        self._load_main_registry()
        
        # Merge Claude agents into unified registry
        self._merge_registries()
    
    def _load_main_registry(self):
        """Load the main agent registry."""
        if self.main_registry_path.exists():
            with open(self.main_registry_path, 'r') as f:
                self.unified_registry = json.load(f)
        else:
            self.unified_registry = {
                "agents": {},
                "version": "1.0.0",
                "provider": "universal"
            }
    
    def _merge_registries(self):
        """Merge Claude agents into the unified registry."""
        claude_agents = self.claude_registry.get_all_agents()
        
        for agent_name, metadata in claude_agents.items():
            # Create unified agent entry
            unified_entry = {
                "name": agent_name,
                "type": "claude",
                "description": metadata.description,
                "capabilities": metadata.capabilities,
                "specializations": metadata.specializations,
                "tags": metadata.tags,
                "source": "claude_agents",
                "file_path": metadata.file_path,
                "version": metadata.version
            }
            
            # Add to unified registry
            self.unified_registry["agents"][f"claude_{agent_name}"] = unified_entry
        
        logger.info(f"Merged {len(claude_agents)} Claude agents into unified registry")
    
    def get_unified_registry(self) -> Dict[str, Any]:
        """Get the complete unified agent registry."""
        return self.unified_registry
    
    def save_unified_registry(self):
        """Save the unified registry to disk."""
        # Create backup
        if self.main_registry_path.exists():
            backup_path = self.main_registry_path.with_suffix('.backup.json')
            with open(backup_path, 'w') as f:
                with open(self.main_registry_path, 'r') as original:
                    f.write(original.read())
        
        # Save unified registry
        with open(self.main_registry_path, 'w') as f:
            json.dump(self.unified_registry, f, indent=2)
        
        logger.info(f"Saved unified registry with {len(self.unified_registry['agents'])} total agents")
    
    async def route_task_to_agent(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a task to the appropriate agent (Claude or main system).
        
        Args:
            task: Task specification
            
        Returns:
            Routing result with selected agent
        """
        # Analyze task to determine best agent type
        task_description = task.get("description", "")
        
        # Check if task explicitly requests Claude agent
        if "claude" in task_description.lower() or task.get("agent_type") == "claude":
            # Find best Claude agent
            claude_agents = self.claude_registry.loader.search_agents(task_description)
            if claude_agents:
                selected_agent = claude_agents[0].name
                return await self.claude_registry.execute_agent_task(selected_agent, task)
        
        # Default routing logic
        return {
            "status": "routed",
            "agent_type": "main_system",
            "task": task
        }
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get statistics about all agents in the system."""
        total_agents = len(self.unified_registry["agents"])
        claude_agents = sum(1 for a in self.unified_registry["agents"].values() if a.get("type") == "claude")
        main_agents = total_agents - claude_agents
        
        return {
            "total_agents": total_agents,
            "claude_agents": claude_agents,
            "main_system_agents": main_agents,
            "claude_capabilities": self.claude_registry.get_capabilities_summary(),
            "claude_specializations": self.claude_registry.get_specializations_summary()
        }
    
    def validate_integration(self) -> Dict[str, Any]:
        """Validate the integration between Claude and main system agents."""
        validation_results = {
            "claude_agents_loaded": self.claude_registry.get_agent_count(),
            "main_agents_loaded": len([a for a in self.unified_registry["agents"].values() if a.get("type") != "claude"]),
            "total_agents": len(self.unified_registry["agents"]),
            "integration_status": "success",
            "issues": []
        }
        
        # Check for duplicates
        agent_names = [a["name"] for a in self.unified_registry["agents"].values()]
        duplicates = [name for name in agent_names if agent_names.count(name) > 1]
        if duplicates:
            validation_results["issues"].append(f"Duplicate agent names found: {duplicates}")
            validation_results["integration_status"] = "warning"
        
        # Check for missing capabilities
        if self.claude_registry.get_agent_count() == 0:
            validation_results["issues"].append("No Claude agents loaded")
            validation_results["integration_status"] = "error"
        
        return validation_results


# Main integration function
async def integrate_claude_agents():
    """
    Main function to integrate all 231 Claude agents with the orchestration system.
    
    This function should be called during system initialization to ensure
    all Claude agents are properly integrated.
    """
    logger.info("Starting Claude agent integration...")
    
    # Create integration bridge
    bridge = ClaudeAgentIntegrationBridge()
    
    # Validate integration
    validation = bridge.validate_integration()
    logger.info(f"Integration validation: {validation}")
    
    if validation["integration_status"] == "error":
        logger.error("Integration failed validation")
        return False
    
    # Save unified registry
    bridge.save_unified_registry()
    
    # Get statistics
    stats = bridge.get_agent_statistics()
    logger.info(f"Integration complete: {stats}")
    
    return True


if __name__ == "__main__":
    # Test integration
    asyncio.run(integrate_claude_agents())