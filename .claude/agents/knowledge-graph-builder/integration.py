#!/usr/bin/env python3

# CLAUDE.md Rules Enforcement
import os
from pathlib import Path

CLAUDE_MD_PATH = "/opt/sutazaiapp/CLAUDE.md"

def check_claude_rules():
    """Check and load CLAUDE.md rules"""
    if os.path.exists(CLAUDE_MD_PATH):
        with open(CLAUDE_MD_PATH, 'r') as f:
            return f.read()
    return None

# Load rules at startup
CLAUDE_RULES = check_claude_rules()

"""
SutazAI Knowledge Graph Integration Module

This module provides integration capabilities with the existing SutazAI agent system,
health check infrastructure, and monitoring systems.
"""

import os
import json
import logging
import asyncio
import httpx
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import yaml

from knowledge_graph_builder import KnowledgeGraphBuilder, ComplianceCheck, EnforcementLevel
from api import app as fastapi_app

logger = logging.getLogger(__name__)

class SutazAIIntegration:
    """Integration manager for SutazAI agent system"""
    
    def __init__(self, kg_builder: KnowledgeGraphBuilder, config: Dict[str, Any]):
        self.kg_builder = kg_builder
        self.config = config
        self.agent_registry = {}
        self.health_checks = {}
        self.compliance_cache = {}
        self.monitoring_callbacks = []
        
    async def register_with_coordinator(self, coordinator_url: str = "http://sutazai-backend:8000"):
        """Register knowledge graph builder with SutazAI coordinator"""
        try:
            agent_info = {
                "agent_name": "knowledge-graph-builder",
                "agent_type": "knowledge-graph-builder",
                "version": "1.0.0",
                "capabilities": [
                    "knowledge_extraction",
                    "graph_construction", 
                    "compliance_validation",
                    "relationship_inference",
                    "graph_visualization",
                    "standards_monitoring"
                ],
                "endpoints": {
                    "health": "/api/health",
                    "status": "/api/stats",
                    "compliance": "/api/validate",
                    "query": "/api/query",
                    "visualize": "/api/visualize"
                },
                "api_url": "http://sutazai-knowledge-graph-builder:8048",
                "health_check_interval": 30,
                "resource_requirements": {
                    "cpu": "1-2 cores",
                    "memory": "2-4GB",
                    "storage": "10GB",
                    "database": "Neo4j"
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{coordinator_url}/api/agents/register",
                    json=agent_info,
                    timeout=10
                )
                
                if response.status_code == 200:
                    logger.info("Successfully registered with SutazAI coordinator")
                    return True
                else:
                    logger.warning(f"Failed to register with coordinator: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error registering with coordinator: {e}")
            return False
    
    async def discover_agents(self, coordinator_url: str = "http://sutazai-backend:8000") -> List[Dict[str, Any]]:
        """Discover other agents in the SutazAI system"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{coordinator_url}/api/agents",
                    timeout=10
                )
                
                if response.status_code == 200:
                    agents = response.json()
                    
                    # Update agent registry
                    for agent in agents:
                        agent_name = agent.get("agent_name", agent.get("name"))
                        if agent_name:
                            self.agent_registry[agent_name] = agent
                    
                    logger.info(f"Discovered {len(agents)} agents")
                    return agents
                else:
                    logger.warning(f"Failed to discover agents: {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error discovering agents: {e}")
            return []
    
    async def sync_agent_information(self):
        """Synchronize agent information with knowledge graph"""
        logger.info("Synchronizing agent information with knowledge graph")
        
        # Discover current agents
        agents = await self.discover_agents()
        
        for agent_info in agents:
            agent_name = agent_info.get("agent_name", agent_info.get("name"))
            if not agent_name:
                continue
            
            # Check if agent entity exists in knowledge graph
            agent_entity_id = f"agent_{agent_name.replace('-', '_')}"
            
            if agent_entity_id not in self.kg_builder.entities:
                # Create new agent entity
                from knowledge_graph_builder import Entity, EntityType, Relationship, RelationType
                
                agent_entity = Entity(
                    id=agent_entity_id,
                    type=EntityType.AGENT,
                    name=agent_name,
                    description=f"SutazAI agent: {agent_name}",
                    properties={
                        "agent_type": agent_info.get("agent_type", "unknown"),
                        "version": agent_info.get("version", "unknown"),
                        "capabilities": agent_info.get("capabilities", []),
                        "endpoints": agent_info.get("endpoints", {}),
                        "api_url": agent_info.get("api_url", ""),
                        "resource_requirements": agent_info.get("resource_requirements", {}),
                        "last_updated": datetime.now().isoformat()
                    }
                )
                
                # Add to knowledge graph
                self.kg_builder.entities[agent_entity_id] = agent_entity
                self.kg_builder.graph.add_node(
                    agent_entity_id,
                    type=agent_entity.type.value,
                    name=agent_entity.name,
                    description=agent_entity.description,
                    **agent_entity.properties
                )
                
                logger.info(f"Added agent {agent_name} to knowledge graph")
            else:
                # Update existing agent entity
                agent_entity = self.kg_builder.entities[agent_entity_id]
                agent_entity.properties.update({
                    "version": agent_info.get("version", agent_entity.properties.get("version")),
                    "capabilities": agent_info.get("capabilities", agent_entity.properties.get("capabilities")),
                    "endpoints": agent_info.get("endpoints", agent_entity.properties.get("endpoints")),
                    "last_updated": datetime.now().isoformat()
                })
                
                # Update graph node
                self.kg_builder.graph.nodes[agent_entity_id].update(agent_entity.properties)
                
                logger.debug(f"Updated agent {agent_name} in knowledge graph")
    
    async def validate_agent_compliance(self, agent_name: str) -> List[ComplianceCheck]:
        """Validate compliance for a specific agent"""
        agent_entity_id = f"agent_{agent_name.replace('-', '_')}"
        
        if agent_entity_id not in self.kg_builder.entities:
            logger.warning(f"Agent {agent_name} not found in knowledge graph")
            return []
        
        # Check cache first
        cache_key = f"compliance_{agent_entity_id}"
        if cache_key in self.compliance_cache:
            cached_time, cached_result = self.compliance_cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=5):
                return cached_result
        
        # Validate compliance
        compliance_checks = await self.kg_builder.validate_compliance(agent_entity_id)
        
        # Cache result
        self.compliance_cache[cache_key] = (datetime.now(), compliance_checks)
        
        return compliance_checks
    
    async def get_agent_health_status(self, agent_name: str) -> Dict[str, Any]:
        """Get health status for an agent"""
        if agent_name not in self.agent_registry:
            return {"status": "unknown", "error": "Agent not found in registry"}
        
        agent_info = self.agent_registry[agent_name]
        api_url = agent_info.get("api_url", "")
        
        if not api_url:
            return {"status": "unknown", "error": "No API URL available"}
        
        try:
            health_endpoint = agent_info.get("endpoints", {}).get("health", "/health")
            full_url = f"{api_url.rstrip('/')}{health_endpoint}"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(full_url, timeout=5)
                
                if response.status_code == 200:
                    health_data = response.json()
                    return {
                        "status": "healthy",
                        "response_time": response.elapsed.total_seconds(),
                        "data": health_data
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "status_code": response.status_code,
                        "error": f"HTTP {response.status_code}"
                    }
                    
        except Exception as e:
            return {
                "status": "unreachable",
                "error": str(e)
            }
    
    async def monitor_system_compliance(self) -> Dict[str, Any]:
        """Monitor compliance across the entire SutazAI system"""
        logger.info("Monitoring system-wide compliance")
        
        # Sync agent information first
        await self.sync_agent_information()
        
        # Check compliance for all agents
        compliance_summary = {
            "timestamp": datetime.now().isoformat(),
            "total_agents": 0,
            "compliant_agents": 0,
            "warning_agents": 0,
            "violation_agents": 0,
            "agent_details": {},
            "system_health": {},
            "blocking_violations": [],
            "warnings": [],
            "recommendations": []
        }
        
        for agent_name in self.agent_registry.keys():
            compliance_summary["total_agents"] += 1
            
            # Get compliance checks
            compliance_checks = await self.validate_agent_compliance(agent_name)
            
            # Get health status
            health_status = await self.get_agent_health_status(agent_name)
            
            # Analyze compliance
            has_blocking = any(c.severity == EnforcementLevel.BLOCKING for c in compliance_checks)
            has_warnings = any(c.severity == EnforcementLevel.WARNING for c in compliance_checks)
            
            if has_blocking:
                compliance_summary["violation_agents"] += 1
                status = "violation"
                # Add blocking violations
                blocking_checks = [c for c in compliance_checks if c.severity == EnforcementLevel.BLOCKING]
                compliance_summary["blocking_violations"].extend([
                    {
                        "agent": agent_name,
                        "standard": c.standard_id,
                        "message": c.message,
                        "evidence": c.evidence
                    }
                    for c in blocking_checks
                ])
            elif has_warnings:
                compliance_summary["warning_agents"] += 1
                status = "warning"
                # Add warnings
                warning_checks = [c for c in compliance_checks if c.severity == EnforcementLevel.WARNING]
                compliance_summary["warnings"].extend([
                    {
                        "agent": agent_name,
                        "standard": c.standard_id,
                        "message": c.message,
                        "evidence": c.evidence
                    }
                    for c in warning_checks
                ])
            else:
                compliance_summary["compliant_agents"] += 1
                status = "compliant"
            
            # Store agent details
            compliance_summary["agent_details"][agent_name] = {
                "compliance_status": status,
                "health_status": health_status["status"],
                "total_checks": len(compliance_checks),
                "violations": len([c for c in compliance_checks if c.severity == EnforcementLevel.BLOCKING]),
                "warnings": len([c for c in compliance_checks if c.severity == EnforcementLevel.WARNING]),
                "last_checked": datetime.now().isoformat()
            }
            
            compliance_summary["system_health"][agent_name] = health_status
        
        # Generate recommendations
        if compliance_summary["violation_agents"] > 0:
            compliance_summary["recommendations"].append(
                f"🚨 URGENT: {compliance_summary['violation_agents']} agents have blocking violations that prevent deployment"
            )
        
        if compliance_summary["warning_agents"] > 0:
            compliance_summary["recommendations"].append(
                f"⚠️ WARNING: {compliance_summary['warning_agents']} agents require review before deployment"
            )
        
        if compliance_summary["compliant_agents"] == compliance_summary["total_agents"]:
            compliance_summary["recommendations"].append(
                "✅ All agents are compliant with hygiene standards"
            )
        
        # Calculate compliance rate
        if compliance_summary["total_agents"] > 0:
            compliance_rate = compliance_summary["compliant_agents"] / compliance_summary["total_agents"]
            compliance_summary["compliance_rate"] = compliance_rate
            
            if compliance_rate < 0.8:
                compliance_summary["recommendations"].append(
                    f"📊 System compliance rate is {compliance_rate:.1%} - target is >80%"
                )
        
        return compliance_summary
    
    async def setup_health_check_integration(self):
        """Set up integration with existing health check infrastructure"""
        logger.info("Setting up health check integration")
        
        # Register health check endpoints
        @fastapi_app.get("/health")
        async def health_check():
            """Health check endpoint for integration with SutazAI monitoring"""
            try:
                # Check knowledge graph status
                kg_status = {
                    "nodes": self.kg_builder.graph.number_of_nodes(),
                    "edges": self.kg_builder.graph.number_of_edges(),
                    "entities": len(self.kg_builder.entities)
                }
                
                # Check Neo4j connection
                neo4j_status = "unknown"
                if self.kg_builder.neo4j_driver:
                    try:
                        with self.kg_builder.neo4j_driver.session() as session:
                            result = session.run("RETURN 1")
                            list(result)  # Consume result
                            neo4j_status = "connected"
                    except Exception:
                        neo4j_status = "disconnected"
                
                return {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "knowledge_graph": kg_status,
                    "database": neo4j_status,
                    "agent_registry": len(self.agent_registry)
                }
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        @fastapi_app.get("/ready")
        async def readiness_check():
            """Readiness check endpoint"""
            try:
                # Check if knowledge graph is ready
                if self.kg_builder.graph.number_of_nodes() == 0:
                    return {
                        "status": "not_ready",
                        "reason": "Knowledge graph not initialized",
                        "timestamp": datetime.now().isoformat()
                    }
                
                return {
                    "status": "ready",
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                return {
                    "status": "not_ready",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        @fastapi_app.get("/metrics")
        async def metrics_endpoint():
            """Prometheus metrics endpoint"""
            try:
                metrics = []
                
                # Knowledge graph metrics
                metrics.append(f"kg_nodes_total {self.kg_builder.graph.number_of_nodes()}")
                metrics.append(f"kg_edges_total {self.kg_builder.graph.number_of_edges()}")
                metrics.append(f"kg_entities_total {len(self.kg_builder.entities)}")
                
                # Entity type metrics
                entity_counts = {}
                for entity in self.kg_builder.entities.values():
                    entity_type = entity.type.value
                    entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
                
                for entity_type, count in entity_counts.items():
                    metrics.append(f'kg_entities_by_type{{type="{entity_type}"}} {count}')
                
                # Agent metrics
                metrics.append(f"kg_agents_registered {len(self.agent_registry)}")
                
                # Compliance metrics (from cache)
                compliant_count = 0
                warning_count = 0
                violation_count = 0
                
                for cache_key, (timestamp, checks) in self.compliance_cache.items():
                    if datetime.now() - timestamp < timedelta(minutes=10):  # Recent cache
                        has_blocking = any(c.severity == EnforcementLevel.BLOCKING for c in checks)
                        has_warnings = any(c.severity == EnforcementLevel.WARNING for c in checks)
                        
                        if has_blocking:
                            violation_count += 1
                        elif has_warnings:
                            warning_count += 1
                        else:
                            compliant_count += 1
                
                metrics.append(f"kg_agents_compliant {compliant_count}")
                metrics.append(f"kg_agents_warnings {warning_count}")
                metrics.append(f"kg_agents_violations {violation_count}")
                
                return "\n".join(metrics), {"Content-Type": "text/plain"}
                
            except Exception as e:
                logger.error(f"Metrics generation failed: {e}")
                return f"# Error generating metrics: {e}", {"Content-Type": "text/plain"}
    
    async def start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        logger.info("Starting monitoring tasks")
        
        async def periodic_sync():
            """Periodically sync agent information"""
            while True:
                try:
                    await self.sync_agent_information()
                    await asyncio.sleep(300)  # Every 5 minutes
                except Exception as e:
                    logger.error(f"Error in periodic sync: {e}")
                    await asyncio.sleep(60)  # Retry in 1 minute on error
        
        async def periodic_compliance_check():
            """Periodically check system compliance"""
            while True:
                try:
                    compliance_summary = await self.monitor_system_compliance()
                    
                    # Trigger alerts for blocking violations
                    if compliance_summary["violation_agents"] > 0:
                        await self.trigger_compliance_alert(compliance_summary)
                    
                    await asyncio.sleep(600)  # Every 10 minutes
                except Exception as e:
                    logger.error(f"Error in periodic compliance check: {e}")
                    await asyncio.sleep(120)  # Retry in 2 minutes on error
        
        # Start background tasks
        asyncio.create_task(periodic_sync())
        asyncio.create_task(periodic_compliance_check())
    
    async def trigger_compliance_alert(self, compliance_summary: Dict[str, Any]):
        """Trigger alert for compliance violations"""
        logger.warning(f"Compliance violations detected: {compliance_summary['violation_agents']} agents")
        
        # Here you would integrate with alerting systems (Slack, email, etc.)
        # For now, just log the violations
        for violation in compliance_summary["blocking_violations"]:
            logger.error(f"BLOCKING VIOLATION - Agent: {violation['agent']}, Standard: {violation['standard']}, Message: {violation['message']}")
    
    def add_monitoring_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for monitoring events"""
        self.monitoring_callbacks.append(callback)
    
    async def initialize_integration(self):
        """Initialize all integration components"""
        logger.info("Initializing SutazAI integration")
        
        # Register with coordinator
        await self.register_with_coordinator()
        
        # Discover and sync agents
        await self.discover_agents()
        await self.sync_agent_information()
        
        # Set up health check endpoints
        await self.setup_health_check_integration()
        
        # Start monitoring tasks
        await self.start_monitoring_tasks()
        
        logger.info("SutazAI integration initialized successfully")

# Integration factory function
async def create_sutazai_integration(kg_builder: KnowledgeGraphBuilder) -> SutazAIIntegration:
    """Create and initialize SutazAI integration"""
    
    # Load integration config
    config_path = Path(__file__).parent / "knowledge_graph_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create integration instance
    integration = SutazAIIntegration(kg_builder, config)
    
    # Initialize integration
    await integration.initialize_integration()
    
    return integration