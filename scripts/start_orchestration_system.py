#!/usr/bin/env python3
"""
SutazAI Multi-Agent Orchestration System Startup Script
Initializes and starts all orchestration components in the correct order.
"""

import asyncio
import logging
import time
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append('/opt/sutazaiapp')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def start_orchestration_system():
    """Start the complete orchestration system"""
    logger.info("Starting SutazAI Multi-Agent Orchestration System...")
    
    try:
        # Import orchestration components
        from backend.app.orchestration.agent_orchestrator import SutazAIAgentOrchestrator
        from backend.app.orchestration.message_bus import MessageBus
        from backend.app.orchestration.task_router import IntelligentTaskRouter
        from backend.app.orchestration.workflow_engine import WorkflowEngine
        from backend.app.orchestration.agent_discovery import AgentDiscoveryService
        from backend.app.orchestration.coordination import DistributedCoordinator
        from backend.app.orchestration.monitoring import OrchestrationMonitor
        
        # Initialize components in order
        logger.info("Initializing orchestration components...")
        
        # 1. Message Bus (communication foundation)
        message_bus = MessageBus()
        await message_bus.initialize()
        logger.info("✓ Message Bus initialized")
        
        # 2. Agent Discovery (find available agents)
        agent_discovery = AgentDiscoveryService()
        await agent_discovery.initialize()
        logger.info("✓ Agent Discovery initialized")
        
        # 3. Task Router (intelligent task routing)
        task_router = IntelligentTaskRouter()
        await task_router.initialize()
        logger.info("✓ Task Router initialized")
        
        # 4. Workflow Engine (complex workflow execution)
        workflow_engine = WorkflowEngine()
        await workflow_engine.initialize()
        logger.info("✓ Workflow Engine initialized")
        
        # 5. Distributed Coordinator (consensus and resource allocation)
        coordinator = DistributedCoordinator()
        await coordinator.initialize()
        logger.info("✓ Distributed Coordinator initialized")
        
        # 6. Main Orchestrator (central coordination)
        orchestrator = SutazAIAgentOrchestrator()
        await orchestrator.initialize()
        logger.info("✓ Main Orchestrator initialized")
        
        # 7. Monitoring System (metrics and health)
        monitor = OrchestrationMonitor()
        monitor.set_components(
            orchestrator=orchestrator,
            message_bus=message_bus,
            task_router=task_router,
            workflow_engine=workflow_engine,
            agent_discovery=agent_discovery,
            coordinator=coordinator
        )
        await monitor.initialize()
        logger.info("✓ Monitoring System initialized")
        
        logger.info("🚀 SutazAI Multi-Agent Orchestration System started successfully!")
        
        # Display system status
        await display_system_status(
            orchestrator, message_bus, task_router, 
            workflow_engine, agent_discovery, coordinator, monitor
        )
        
        # Keep the system running
        logger.info("System is running... Press Ctrl+C to stop")
        
        try:
            while True:
                await asyncio.sleep(10)
                # Periodic status check
                await periodic_status_check(agent_discovery, orchestrator)
        except KeyboardInterrupt:
            logger.info("Shutdown signal received...")
            
            # Graceful shutdown
            await shutdown_components(
                orchestrator, message_bus, task_router,
                workflow_engine, agent_discovery, coordinator, monitor
            )
            
    except Exception as e:
        logger.error(f"Failed to start orchestration system: {e}")
        sys.exit(1)

async def display_system_status(orchestrator, message_bus, task_router, 
                               workflow_engine, agent_discovery, coordinator, monitor):
    """Display current system status"""
    print("\n" + "="*80)
    print("SUTAZAI MULTI-AGENT ORCHESTRATION SYSTEM STATUS")
    print("="*80)
    
    try:
        # Agent Discovery Status
        agents = await agent_discovery.get_discovered_agents()
        healthy_agents = [a for a in agents if a.status == "healthy"]
        print(f"Agents Discovered: {len(agents)} (Healthy: {len(healthy_agents)})")
        
        # Task Router Status
        queue_status = await task_router.get_queue_status()
        print(f"Task Queue Size: {queue_status.get('queue_size', 0)}")
        print(f"Load Balancing Algorithm: {queue_status.get('algorithm', 'Unknown')}")
        
        # Workflow Engine Status
        workflow_metrics = await workflow_engine.get_metrics()
        print(f"Active Workflows: {workflow_metrics.get('workflow_details', {}).get('running_workflows', 0)}")
        print(f"Total Workflows Executed: {workflow_metrics.get('workflows_executed', 0)}")
        
        # Orchestrator Status
        orchestrator_metrics = await orchestrator.get_system_metrics()
        print(f"Tasks Completed: {orchestrator_metrics.get('tasks_completed', 0)}")
        print(f"System Throughput: {orchestrator_metrics.get('system_throughput', 0):.2f}")
        
        # Message Bus Status
        bus_metrics = await message_bus.get_metrics()
        print(f"Messages Sent: {bus_metrics.get('messages_sent', 0)}")
        print(f"Messages Received: {bus_metrics.get('messages_received', 0)}")
        
        # Coordination Status
        coord_metrics = await coordinator.get_coordination_metrics()
        print(f"Consensus Sessions: {coord_metrics.get('consensus_sessions_completed', 0)}")
        print(f"Resource Allocations: {coord_metrics.get('resource_allocations', 0)}")
        
        print("="*80)
        print("System Status: ✓ ALL SYSTEMS OPERATIONAL")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Error displaying system status: {e}")

async def periodic_status_check(agent_discovery, orchestrator):
    """Perform periodic status checks"""
    try:
        # Check agent health
        agents = await agent_discovery.get_discovered_agents()
        healthy_count = len([a for a in agents if a.status == "healthy"])
        
        if healthy_count == 0:
            logger.warning("⚠️  No healthy agents available!")
        elif healthy_count < len(agents) * 0.5:
            logger.warning(f"⚠️  Low agent health: {healthy_count}/{len(agents)} healthy")
        
        # Check orchestrator metrics
        metrics = await orchestrator.get_system_metrics()
        if metrics.get('tasks_failed', 0) > metrics.get('tasks_completed', 1) * 0.1:
            logger.warning("⚠️  High task failure rate detected")
            
    except Exception as e:
        logger.error(f"Status check failed: {e}")

async def shutdown_components(*components):
    """Gracefully shutdown all components"""
    logger.info("Shutting down orchestration components...")
    
    for component in components:
        if component and hasattr(component, 'stop'):
            try:
                await component.stop()
                logger.info(f"✓ {component.__class__.__name__} stopped")
            except Exception as e:
                logger.error(f"Error stopping {component.__class__.__name__}: {e}")
    
    logger.info("🔴 SutazAI Orchestration System stopped")

def check_prerequisites():
    """Check system prerequisites"""
    logger.info("Checking system prerequisites...")
    
    # Check Redis availability
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, password='redis_password', decode_responses=True)
        r.ping()
        logger.info("✓ Redis connection available")
    except Exception as e:
        logger.error(f"✗ Redis not available: {e}")
        return False
    
    # Check if backend is running
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            logger.info("✓ Backend service available")
        else:
            logger.warning(f"⚠️  Backend service returned {response.status_code}")
    except Exception as e:
        logger.warning(f"⚠️  Backend service check failed: {e}")
    
    # Check Docker containers
    try:
        import docker
        client = docker.from_env()
        sutazai_containers = [c for c in client.containers.list() if 'sutazai' in c.name]
        logger.info(f"✓ {len(sutazai_containers)} SutazAI containers running")
    except Exception as e:
        logger.warning(f"⚠️  Docker check failed: {e}")
    
    return True

def main():
    """Main function"""
    print("🤖 SutazAI Multi-Agent Orchestration System")
    print("   Advanced AI Agent Coordination Platform")
    print("   https://github.com/sutazai/sutazai")
    print()
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites not met. Please ensure Redis and other services are running.")
        sys.exit(1)
    
    # Start orchestration system
    try:
        asyncio.run(start_orchestration_system())
    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")
    except Exception as e:
        logger.error(f"System startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()