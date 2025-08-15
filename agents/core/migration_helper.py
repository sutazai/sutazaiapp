#!/usr/bin/env python3
"""
Migration Helper for SutazAI System Agent Enhancement
Provides backward compatibility and migration utilities for existing agents

Features:
- Automatic detection of old agent patterns
- Seamless migration to BaseAgentV2
- Compatibility shims for legacy methods
- Migration validation and reporting
"""

import os
import sys
import json
import logging
import asyncio
import inspect
from typing import Dict, Any, Optional, List, Type, Union
from pathlib import Path

# Import the canonical base class and logging
sys.path.append('/opt/sutazaiapp')
from agents.core.base_agent import BaseAgent
from backend.app.core.logging_config import get_logger

# Configure structured logging (Rule 8 compliance)
logger = get_logger(__name__)


class AgentMigrationError(Exception):
    """Raised when agent migration fails"""
    pass


class LegacyAgentWrapper(BaseAgent):
    """
    Wrapper class that provides backward compatibility for legacy agents
    
    This allows existing agents to work with the new infrastructure
    without requiring immediate code changes.
    """
    
    def __init__(self, legacy_agent_class: Type, *args, **kwargs):
        """
        Initialize wrapper with legacy agent class
        
        Args:
            legacy_agent_class: The original agent class inheriting from BaseAgent
            *args, **kwargs: Arguments to pass to both legacy and new init
        """
        self.legacy_agent_class = legacy_agent_class
        self.legacy_methods = {}
        
        # Initialize the new base agent
        super().__init__(*args, **kwargs)
        
        # Store legacy method implementations
        self._extract_legacy_methods()
        
        logger.info(f"Legacy agent wrapper initialized for {legacy_agent_class.__name__}")
    
    def _extract_legacy_methods(self):
        """Extract overridden methods from legacy agent class"""
        for name, method in inspect.getmembers(self.legacy_agent_class, inspect.isfunction):
            # Skip private methods and inherited base methods
            if not name.startswith('_') and hasattr(BaseAgent, name):
                # Check if method is overridden
                legacy_method = getattr(self.legacy_agent_class, name)
                base_method = getattr(BaseAgent, name)
                
                if legacy_method != base_method:
                    self.legacy_methods[name] = legacy_method
                    logger.debug(f"Captured legacy method: {name}")
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process task using legacy method if available
        
        This provides compatibility for agents that override process_task
        """
        if 'process_task' in self.legacy_methods:
            try:
                # Create a temporary instance for legacy method execution
                legacy_instance = self.legacy_agent_class.__new__(self.legacy_agent_class)
                
                # Copy relevant attributes
                legacy_instance.logger = self.logger
                legacy_instance.config = self.config
                legacy_instance.agent_name = self.agent_name
                legacy_instance.agent_type = self.agent_type
                legacy_instance.backend_url = self.backend_url
                legacy_instance.ollama_url = self.ollama_url
                
                # Execute legacy process_task method
                if inspect.iscoroutinefunction(self.legacy_methods['process_task']):
                    result = await self.legacy_methods['process_task'](legacy_instance, task)
                else:
                    result = self.legacy_methods['process_task'](legacy_instance, task)
                
                # Convert to new TaskResult format
                from agents.core.base_agent import TaskResult
                from datetime import datetime
                
                return TaskResult(
                    task_id=task.get("id", "unknown"),
                    status="completed",
                    result=result,
                    processing_time=0.1  # Placeholder
                )
                
            except Exception as e:
                logger.error(f"Legacy process_task failed: {e}")
                # Fall back to default implementation
                return await super().process_task(task)
        else:
            # Use new implementation
            return await super().process_task(task)
    
    def query_ollama_sync(self, prompt: str, model: str = None) -> Optional[str]:
        """
        Legacy sync Ollama query method
        
        Provided for backward compatibility with agents using the old sync interface
        """
        try:
            # Try to run async method in current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, 
                        self.query_ollama(prompt, model)
                    )
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(self.query_ollama(prompt, model))
        except Exception as e:
            logger.error(f"Legacy sync Ollama query failed: {e}")
            return None


class AgentMigrationValidator:
    """
    Validates agent migration and provides reports
    """
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_agent_directory(self, agent_path: Path) -> Dict[str, Any]:
        """
        Validate a single agent directory for migration readiness
        
        Args:
            agent_path: Path to agent directory
            
        Returns:
            Validation results dictionary
        """
        agent_name = agent_path.name
        validation = {
            "agent_name": agent_name,
            "path": str(agent_path),
            "status": "unknown",
            "issues": [],
            "recommendations": [],
            "migration_type": "none"
        }
        
        try:
            # Check for app.py
            app_file = agent_path / "app.py"
            if not app_file.exists():
                validation["issues"].append("Missing app.py file")
                validation["status"] = "error"
                return validation
            
            # Analyze app.py content
            app_content = app_file.read_text()
            
            # Check for base agent import
            if "from agents.core.base_agent import BaseAgent" in app_content:
                validation["migration_type"] = "canonical_agent"
                validation["status"] = "current"
            elif "from agents.core.base_agent import BaseAgentV2" in app_content:
                validation["migration_type"] = "canonical_agent"  
                validation["status"] = "current"
            elif any(legacy in app_content for legacy in ["from agent_base import BaseAgent", "from shared.agent_base import BaseAgent"]):
                validation["migration_type"] = "legacy_agent"
                validation["issues"].append("Using legacy BaseAgent import - needs consolidation")
            else:
                validation["issues"].append("No recognizable base agent import found")
            
            # Check for common patterns
            if "def process_task" in app_content:
                validation["recommendations"].append("Custom process_task method detected - ensure compatibility")
            
            if "query_ollama" in app_content and "await" not in app_content:
                validation["issues"].append("Sync Ollama usage detected - consider migrating to async")
            
            # Check for requirements.txt
            requirements_file = agent_path / "requirements.txt"
            if requirements_file.exists():
                validation["recommendations"].append("Review requirements.txt for compatibility")
            
            # Set overall status
            if not validation["issues"]:
                if validation["migration_type"] == "v2_agent":
                    validation["status"] = "current"
                else:
                    validation["status"] = "migration_ready"
            else:
                validation["status"] = "needs_attention"
            
        except Exception as e:
            validation["status"] = "error"
            validation["issues"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def validate_all_agents(self, agents_base_path: str = "/opt/sutazaiapp/agents") -> Dict[str, Any]:
        """
        Validate all agents in the agents directory
        
        Returns:
            Complete validation report
        """
        agents_path = Path(agents_base_path)
        report = {
            "timestamp": "2025-01-01T00:00:00Z",  # Will be updated
            "total_agents": 0,
            "v2_agents": 0,
            "v1_agents": 0,
            "error_agents": 0,
            "migration_ready": 0,
            "needs_attention": 0,
            "agents": {}
        }
        
        # Find all agent directories
        for agent_dir in agents_path.iterdir():
            if agent_dir.is_dir() and not agent_dir.name.startswith('.'):
                # Skip core and other utility directories
                if agent_dir.name in ['core', '__pycache__', 'shared']:
                    continue
                
                validation = self.validate_agent_directory(agent_dir)
                report["agents"][agent_dir.name] = validation
                report["total_agents"] += 1
                
                # Update counters
                if validation["status"] == "current":
                    report["v2_agents"] += 1
                elif validation["migration_type"] == "v1_agent":
                    report["v1_agents"] += 1
                elif validation["status"] == "error":
                    report["error_agents"] += 1
                elif validation["status"] == "migration_ready":
                    report["migration_ready"] += 1
                elif validation["status"] == "needs_attention":
                    report["needs_attention"] += 1
        
        return report
    
    def generate_migration_plan(self, validation_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate migration plan based on validation results
        
        Returns:
            List of migration steps
        """
        plan = []
        
        # Phase 1: Immediate compatibility using wrapper
        v1_agents = [
            name for name, validation in validation_report["agents"].items()
            if validation["migration_type"] == "v1_agent" and validation["status"] in ["migration_ready", "needs_attention"]
        ]
        
        if v1_agents:
            plan.append({
                "phase": 1,
                "title": "Immediate Compatibility (Legacy Wrapper)",
                "description": "Use LegacyAgentWrapper for existing v1 agents",
                "agents": v1_agents,
                "effort": "low",
                "priority": "high"
            })
        
        # Phase 2: Fix agents with issues
        problem_agents = [
            name for name, validation in validation_report["agents"].items()
            if validation["status"] == "error" or len(validation["issues"]) > 0
        ]
        
        if problem_agents:
            plan.append({
                "phase": 2,
                "title": "Fix Problem Agents",
                "description": "Address validation issues and errors",
                "agents": problem_agents,
                "effort": "medium",
                "priority": "high"
            })
        
        # Phase 3: Full migration to v2
        if v1_agents:
            plan.append({
                "phase": 3,
                "title": "Full Migration to BaseAgent",
                "description": "Migrate agents to use BaseAgent directly",
                "agents": v1_agents,
                "effort": "medium",
                "priority": "medium"
            })
        
        return plan


def create_agent_factory(agent_name: str, 
                        agent_path: str = "/opt/sutazaiapp/agents") -> Union[BaseAgent, LegacyAgentWrapper]:
    """
    Factory function to create appropriate agent instance
    
    Automatically detects agent type and creates compatible instance
    """
    agent_dir = Path(agent_path) / agent_name
    app_file = agent_dir / "app.py"
    
    if not app_file.exists():
        raise AgentMigrationError(f"Agent {agent_name} not found at {agent_dir}")
    
    try:
        # Dynamic import of agent
        sys.path.insert(0, str(agent_dir))
        
        try:
            app_module = __import__('app')
            
            # Look for agent class in module
            agent_class = None
            for attr_name in dir(app_module):
                attr = getattr(app_module, attr_name)
                if (inspect.isclass(attr) and 
                    issubclass(attr, BaseAgent) and 
                    attr != BaseAgent):
                    agent_class = attr
                    break
            
            if not agent_class:
                raise AgentMigrationError(f"No agent class found in {agent_name}/app.py")
            
            # Create BaseAgent instance directly
            if issubclass(agent_class, BaseAgent):
                logger.info(f"Creating BaseAgent instance for {agent_name}")
                return agent_class()
            else:
                logger.info(f"Creating legacy wrapper for {agent_name}")
                return LegacyAgentWrapper(agent_class)
                
        finally:
            sys.path.pop(0)
            
    except Exception as e:
        raise AgentMigrationError(f"Failed to create agent {agent_name}: {e}")


def migrate_agent_to_v2(agent_name: str, 
                       agent_path: str = "/opt/sutazaiapp/agents",
                       backup: bool = True) -> bool:
    """
    Migrate a specific agent to BaseAgent
    
    Args:
        agent_name: Name of agent to migrate
        agent_path: Base path to agents directory
        backup: Whether to create backup before migration
        
    Returns:
        True if migration successful
    """
    agent_dir = Path(agent_path) / agent_name
    app_file = agent_dir / "app.py"
    
    if not app_file.exists():
        raise AgentMigrationError(f"Agent {agent_name} not found")
    
    try:
        # Create backup if requested
        if backup:
            backup_file = app_file.with_suffix('.py.backup')
            backup_file.write_text(app_file.read_text())
            logger.info(f"Created backup: {backup_file}")
        
        # Read current content
        content = app_file.read_text()
        
        # Perform migration transformations
        migrations = [
            # Update import statement
            ("from agent_base import BaseAgent", "from agents.core.base_agent import BaseAgent"),
            ("from shared.agent_base import BaseAgent", "from agents.core.base_agent import BaseAgent"),
            ("class.*Agent.*BaseAgent", "class {agent_name}Agent(BaseAgent)"),
            # Update method signatures if needed
            ("def process_task(self, task)", "async def process_task(self, task)"),
            # Update Ollama calls
            ("self.query_ollama(", "await self.query_ollama("),
        ]
        
        modified_content = content
        for old_pattern, new_pattern in migrations:
            if old_pattern in modified_content:
                if "{agent_name}" in new_pattern:
                    new_pattern = new_pattern.format(agent_name=agent_name.replace('-', '_').title())
                modified_content = modified_content.replace(old_pattern, new_pattern)
                logger.debug(f"Applied migration: {old_pattern} -> {new_pattern}")
        
        # Write modified content
        app_file.write_text(modified_content)
        
        logger.info(f"Successfully migrated {agent_name} to BaseAgent")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed for {agent_name}: {e}")
        
        # Restore backup if migration failed
        if backup:
            backup_file = app_file.with_suffix('.py.backup')
            if backup_file.exists():
                app_file.write_text(backup_file.read_text())
                logger.info(f"Restored backup for {agent_name}")
        
        raise AgentMigrationError(f"Migration failed: {e}")


def run_migration_report():
    """Generate and display migration report for all agents"""
    validator = AgentMigrationValidator()
    report = validator.validate_all_agents()
    
    logger.info("\n=== SutazAI Agent Migration Report ===")
    logger.info(f"Total Agents: {report['total_agents']}")
    logger.info(f"Already V2: {report['v2_agents']}")
    logger.info(f"Legacy V1: {report['v1_agents']}")
    logger.info(f"Migration Ready: {report['migration_ready']}")
    logger.info(f"Need Attention: {report['needs_attention']}")
    logger.info(f"Errors: {report['error_agents']}")
    
    # Generate migration plan
    plan = validator.generate_migration_plan(report)
    
    logger.info("\n=== Migration Plan ===")
    for step in plan:
        logger.info(f"\nPhase {step['phase']}: {step['title']}")
        logger.info(f"Priority: {step['priority']}, Effort: {step['effort']}")
        logger.info(f"Agents: {len(step['agents'])}")
        logger.info(f"Description: {step['description']}")
    
    return report, plan


if __name__ == "__main__":
    # Run migration report when executed directly
    run_migration_report()