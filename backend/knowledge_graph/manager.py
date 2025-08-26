"""
Knowledge Graph Manager
======================

Central manager for the SutazAI knowledge graph system.
Coordinates all components including Neo4j, query engine,
reasoning engine, visualization, and real-time updates.

Features:
- Unified initialization and configuration
- Component lifecycle management
- Health monitoring and recovery
- Performance metrics collection
- Configuration management
- Error handling and logging
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .neo4j_manager import Neo4jManager, set_neo4j_manager
from .query_engine import QueryEngine
from .reasoning_engine import ReasoningEngine, set_reasoning_engine
from .visualization import VisualizationManager
from .real_time_updater import RealTimeUpdater, set_real_time_updater
from .graph_builder import KnowledgeGraphBuilder
from .schema import KnowledgeGraphSchema
from ..ai_agents.core.agent_registry import get_agent_registry


class KnowledgeGraphConfig:
    """Configuration class for the knowledge graph system"""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        config = config_dict or {}
        
        # Neo4j configuration
        self.neo4j_uri = config.get("neo4j_uri", "bolt://localhost:7687")
        self.neo4j_username = config.get("neo4j_username", "neo4j")
        self.neo4j_password = config.get("neo4j_password", "password")
        self.neo4j_database = config.get("neo4j_database", "sutazai")
        
        # System paths
        self.base_path = config.get("base_path", "/opt/sutazaiapp/backend")
        self.output_path = config.get("output_path", "/opt/sutazaiapp/backend/knowledge_graph/output")
        
        # Feature flags
        self.enable_real_time_updates = config.get("enable_real_time_updates", True)
        self.enable_reasoning = config.get("enable_reasoning", True)
        self.enable_visualization = config.get("enable_visualization", True)
        
        # Performance settings
        self.batch_size = config.get("batch_size", 1000)
        self.max_retries = config.get("max_retries", 3)
        self.query_timeout = config.get("query_timeout", 120)
        
        # Real-time update settings
        self.update_batch_size = config.get("update_batch_size", 10)
        self.update_batch_timeout = config.get("update_batch_timeout", 30.0)
        
        # Reasoning settings
        self.reasoning_cycle_interval = config.get("reasoning_cycle_interval", 3600)  # 1 hour
        self.enable_auto_inference = config.get("enable_auto_inference", False)
        
        # Logging settings
        self.log_level = config.get("log_level", "INFO")
        self.log_file = config.get("log_file", "knowledge_graph.log")
    
    @classmethod
    def from_env(cls) -> 'KnowledgeGraphConfig':
        """Create configuration from environment variables"""
        config = {
            "neo4j_uri": os.getenv("KG_NEO4J_URI", "bolt://localhost:7687"),
            "neo4j_username": os.getenv("KG_NEO4J_USERNAME", "neo4j"),
            "neo4j_password": os.getenv("KG_NEO4J_PASSWORD", "password"),
            "neo4j_database": os.getenv("KG_NEO4J_DATABASE", "sutazai"),
            "base_path": os.getenv("KG_BASE_PATH", "/opt/sutazaiapp/backend"),
            "output_path": os.getenv("KG_OUTPUT_PATH", "/opt/sutazaiapp/backend/knowledge_graph/output"),
            "enable_real_time_updates": os.getenv("KG_ENABLE_REAL_TIME", "true").lower() == "true",
            "enable_reasoning": os.getenv("KG_ENABLE_REASONING", "true").lower() == "true",
            "enable_visualization": os.getenv("KG_ENABLE_VISUALIZATION", "true").lower() == "true",
            "log_level": os.getenv("KG_LOG_LEVEL", "INFO"),
            "log_file": os.getenv("KG_LOG_FILE", "knowledge_graph.log")
        }
        
        return cls(config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "neo4j_uri": self.neo4j_uri,
            "neo4j_username": self.neo4j_username,
            "neo4j_database": self.neo4j_database,
            "base_path": self.base_path,
            "output_path": self.output_path,
            "enable_real_time_updates": self.enable_real_time_updates,
            "enable_reasoning": self.enable_reasoning,
            "enable_visualization": self.enable_visualization,
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "query_timeout": self.query_timeout,
            "update_batch_size": self.update_batch_size,
            "update_batch_timeout": self.update_batch_timeout,
            "reasoning_cycle_interval": self.reasoning_cycle_interval,
            "enable_auto_inference": self.enable_auto_inference,
            "log_level": self.log_level,
            "log_file": self.log_file
        }


class KnowledgeGraphManager:
    """
    Central manager for the SutazAI knowledge graph system
    """
    
    def __init__(self, config: Optional[KnowledgeGraphConfig] = None):
        self.config = config or KnowledgeGraphConfig.from_env()
        
        # Component instances
        self.neo4j_manager: Optional[Neo4jManager] = None
        self.query_engine: Optional[QueryEngine] = None
        self.reasoning_engine: Optional[ReasoningEngine] = None
        self.visualization_manager: Optional[VisualizationManager] = None
        self.real_time_updater: Optional[RealTimeUpdater] = None
        self.graph_builder: Optional[KnowledgeGraphBuilder] = None
        
        # State management
        self.is_initialized = False
        self.is_running = False
        self.initialization_time: Optional[datetime] = None
        
        # Statistics and monitoring
        self.stats = {
            "initialization_time": None,
            "total_queries": 0,
            "total_reasoning_cycles": 0,
            "total_visualizations": 0,
            "total_updates": 0,
            "errors": 0,
            "last_health_check": None,
            "component_status": {}
        }
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger("kg_manager")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.config.log_file)
            ]
        )
    
    async def initialize(self) -> bool:
        """Initialize all knowledge graph components"""
        try:
            self.logger.info("Initializing Knowledge Graph Manager")
            start_time = datetime.utcnow()
            
            # Ensure output directory exists
            Path(self.config.output_path).mkdir(parents=True, exist_ok=True)
            
            # Initialize Neo4j manager
            success = await self._initialize_neo4j()
            if not success:
                return False
            
            # Initialize query engine
            self._initialize_query_engine()
            
            # Initialize reasoning engine (if enabled)
            if self.config.enable_reasoning:
                self._initialize_reasoning_engine()
            
            # Initialize visualization manager (if enabled)
            if self.config.enable_visualization:
                self._initialize_visualization_manager()
            
            # Initialize real-time updater (if enabled)
            if self.config.enable_real_time_updates:
                success = await self._initialize_real_time_updater()
                if not success:
                    self.logger.warning("Real-time updater initialization failed, continuing without it")
            
            # Initialize graph builder
            self._initialize_graph_builder()
            
            # Start background tasks
            self._start_background_tasks()
            
            # Update state
            self.is_initialized = True
            self.is_running = True
            self.initialization_time = datetime.utcnow()
            
            initialization_duration = (self.initialization_time - start_time).total_seconds()
            self.stats["initialization_time"] = initialization_duration
            
            self.logger.info(f"Knowledge Graph Manager initialized successfully in {initialization_duration:.2f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Knowledge Graph Manager: {e}")
            return False
    
    async def _initialize_neo4j(self) -> bool:
        """Initialize Neo4j manager"""
        try:
            self.neo4j_manager = Neo4jManager(
                uri=self.config.neo4j_uri,
                username=self.config.neo4j_username,
                password=self.config.neo4j_password,
                database=self.config.neo4j_database
            )
            
            success = await self.neo4j_manager.initialize()
            if success:
                set_neo4j_manager(self.neo4j_manager)
                self.stats["component_status"]["neo4j"] = "healthy"
                self.logger.info("Neo4j manager initialized successfully")
            else:
                self.stats["component_status"]["neo4j"] = "failed"
                
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Neo4j manager: {e}")
            self.stats["component_status"]["neo4j"] = "error"
            return False
    
    def _initialize_query_engine(self):
        """Initialize query engine"""
        try:
            if not self.neo4j_manager:
                raise ValueError("Neo4j manager not initialized")
                
            self.query_engine = QueryEngine(self.neo4j_manager)
            self.stats["component_status"]["query_engine"] = "healthy"
            self.logger.info("Query engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize query engine: {e}")
            self.stats["component_status"]["query_engine"] = "error"
    
    def _initialize_reasoning_engine(self):
        """Initialize reasoning engine"""
        try:
            if not self.query_engine:
                raise ValueError("Query engine not initialized")
                
            self.reasoning_engine = ReasoningEngine(self.query_engine)
            set_reasoning_engine(self.reasoning_engine)
            self.stats["component_status"]["reasoning_engine"] = "healthy"
            self.logger.info("Reasoning engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize reasoning engine: {e}")
            self.stats["component_status"]["reasoning_engine"] = "error"
    
    def _initialize_visualization_manager(self):
        """Initialize visualization manager"""
        try:
            if not self.query_engine:
                raise ValueError("Query engine not initialized")
                
            self.visualization_manager = VisualizationManager(self.query_engine)
            self.stats["component_status"]["visualization_manager"] = "healthy"
            self.logger.info("Visualization manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize visualization manager: {e}")
            self.stats["component_status"]["visualization_manager"] = "error"
    
    async def _initialize_real_time_updater(self) -> bool:
        """Initialize real-time updater"""
        try:
            if not self.neo4j_manager:
                raise ValueError("Neo4j manager not initialized")
                
            # Get agent registry if available
            agent_registry = get_agent_registry()
            
            self.real_time_updater = RealTimeUpdater(
                neo4j_manager=self.neo4j_manager,
                agent_registry=agent_registry,
                base_path=self.config.base_path
            )
            
            success = await self.real_time_updater.start()
            if success:
                set_real_time_updater(self.real_time_updater)
                self.stats["component_status"]["real_time_updater"] = "healthy"
                self.logger.info("Real-time updater initialized successfully")
            else:
                self.stats["component_status"]["real_time_updater"] = "failed"
                
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to initialize real-time updater: {e}")
            self.stats["component_status"]["real_time_updater"] = "error"
            return False
    
    def _initialize_graph_builder(self):
        """Initialize graph builder"""
        try:
            if not self.neo4j_manager:
                raise ValueError("Neo4j manager not initialized")
                
            self.graph_builder = KnowledgeGraphBuilder(
                base_path=self.config.base_path,
                neo4j_manager=self.neo4j_manager
            )
            
            self.stats["component_status"]["graph_builder"] = "healthy"
            self.logger.info("Graph builder initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize graph builder: {e}")
            self.stats["component_status"]["graph_builder"] = "error"
    
    def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        tasks = [
            self._health_monitor(),
            self._stats_collector()
        ]
        
        # Add reasoning cycle task if reasoning is enabled
        if self.config.enable_reasoning and self.reasoning_engine:
            tasks.append(self._reasoning_cycle_task())
        
        for coro in tasks:
            task = asyncio.create_task(coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
    
    async def _health_monitor(self):
        """Monitor component health"""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_health_check()
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(300)
    
    async def _stats_collector(self):
        """Collect system statistics"""
        while not self._shutdown_event.is_set():
            try:
                await self._collect_statistics()
                await asyncio.sleep(600)  # Collect every 10 minutes
                
            except Exception as e:
                self.logger.error(f"Error in stats collector: {e}")
                await asyncio.sleep(600)
    
    async def _reasoning_cycle_task(self):
        """Background reasoning cycle task"""
        while not self._shutdown_event.is_set():
            try:
                if self.reasoning_engine:
                    self.logger.info("Starting background reasoning cycle")
                    results = await self.reasoning_engine.perform_reasoning_cycle()
                    self.stats["total_reasoning_cycles"] += 1
                    self.logger.info(f"Reasoning cycle completed: {results.get('performance', {})}")
                
                await asyncio.sleep(self.config.reasoning_cycle_interval)
                
            except Exception as e:
                self.logger.error(f"Error in reasoning cycle: {e}")
                await asyncio.sleep(self.config.reasoning_cycle_interval)
    
    async def _perform_health_check(self):
        """Perform health check on all components"""
        try:
            health_status = {}
            
            # Check Neo4j connection
            if self.neo4j_manager:
                try:
                    stats = await self.neo4j_manager.get_graph_statistics()
                    health_status["neo4j"] = "healthy" if "error" not in stats else "unhealthy"
                except Exception as e:
                    health_status["neo4j"] = "unhealthy"
                    self.logger.warning(f"Neo4j health check failed: {e}")
            
            # Check real-time updater
            if self.real_time_updater:
                health_status["real_time_updater"] = "healthy" if self.real_time_updater.is_running else "unhealthy"
            
            # Check reasoning engine
            if self.reasoning_engine:
                try:
                    stats = self.reasoning_engine.get_reasoning_statistics()
                    health_status["reasoning_engine"] = "healthy"
                except Exception as e:
                    health_status["reasoning_engine"] = "unhealthy"
                    self.logger.warning(f"Reasoning engine health check failed: {e}")
            
            self.stats["component_status"].update(health_status)
            self.stats["last_health_check"] = datetime.utcnow().isoformat()
            
        except Exception as e:
            self.logger.error(f"Error performing health check: {e}")
    
    async def _collect_statistics(self):
        """Collect system statistics"""
        try:
            if self.neo4j_manager:
                graph_stats = await self.neo4j_manager.get_graph_statistics()
                self.stats.update({
                    "total_nodes": graph_stats.get("total_nodes", 0),
                    "total_relationships": graph_stats.get("total_relationships", 0),
                    "neo4j_stats": graph_stats.get("manager_stats", {})
                })
            
            if self.real_time_updater:
                updater_stats = self.real_time_updater.get_stats()
                self.stats["real_time_updater_stats"] = updater_stats
            
            if self.reasoning_engine:
                reasoning_stats = self.reasoning_engine.get_reasoning_statistics()
                self.stats["reasoning_stats"] = reasoning_stats
            
        except Exception as e:
            self.logger.error(f"Error collecting statistics: {e}")
    
    # Public API methods
    
    async def build_initial_graph(self) -> Dict[str, Any]:
        """Build the initial knowledge graph"""
        try:
            if not self.graph_builder:
                raise ValueError("Graph builder not initialized")
            
            self.logger.info("Building initial knowledge graph")
            result = await self.graph_builder.build_knowledge_graph()
            
            self.logger.info(f"Initial graph build completed: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error building initial graph: {e}")
            self.stats["errors"] += 1
            return {"error": str(e)}
    
    async def rebuild_graph(self) -> Dict[str, Any]:
        """Rebuild the entire knowledge graph"""
        try:
            if not self.neo4j_manager or not self.graph_builder:
                raise ValueError("Required components not initialized")
            
            self.logger.info("Rebuilding knowledge graph")
            
            # Clear existing graph
            await self.neo4j_manager.clear_graph()
            
            # Rebuild from scratch
            result = await self.graph_builder.build_knowledge_graph()
            
            self.logger.info(f"Graph rebuild completed: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error rebuilding graph: {e}")
            self.stats["errors"] += 1
            return {"error": str(e)}
    
    def get_query_engine(self) -> Optional[QueryEngine]:
        """Get the query engine instance"""
        return self.query_engine
    
    def get_reasoning_engine(self) -> Optional[ReasoningEngine]:
        """Get the reasoning engine instance"""
        return self.reasoning_engine
    
    def get_visualization_manager(self) -> Optional[VisualizationManager]:
        """Get the visualization manager instance"""
        return self.visualization_manager
    
    def get_neo4j_manager(self) -> Optional[Neo4jManager]:
        """Get the Neo4j manager instance"""
        return self.neo4j_manager
    
    def get_real_time_updater(self) -> Optional[RealTimeUpdater]:
        """Get the real-time updater instance"""
        return self.real_time_updater
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            **self.stats,
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "initialization_time": self.initialization_time.isoformat() if self.initialization_time else None,
            "uptime_seconds": (datetime.utcnow() - self.initialization_time).total_seconds() if self.initialization_time else 0,
            "configuration": self.config.to_dict(),
            "background_tasks": len(self._background_tasks)
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        overall_healthy = all(
            status == "healthy" 
            for status in self.stats["component_status"].values()
        )
        
        return {
            "overall_status": "healthy" if overall_healthy else "degraded",
            "components": self.stats["component_status"],
            "is_running": self.is_running,
            "last_health_check": self.stats["last_health_check"]
        }
    
    async def shutdown(self):
        """Shutdown the knowledge graph manager"""
        try:
            self.logger.info("Shutting down Knowledge Graph Manager")
            
            self.is_running = False
            
            # Signal shutdown to background tasks
            self._shutdown_event.set()
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Shutdown components
            if self.real_time_updater:
                await self.real_time_updater.shutdown()
            
            if self.neo4j_manager:
                await self.neo4j_manager.shutdown()
            
            self.logger.info("Knowledge Graph Manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Global manager instance
_knowledge_graph_manager: Optional[KnowledgeGraphManager] = None


def get_knowledge_graph_manager() -> Optional[KnowledgeGraphManager]:
    """Get the global knowledge graph manager instance"""
    return _knowledge_graph_manager


def set_knowledge_graph_manager(manager: KnowledgeGraphManager):
    """Set the global knowledge graph manager instance"""
    global _knowledge_graph_manager
    _knowledge_graph_manager = manager


async def initialize_knowledge_graph_system(config: Optional[KnowledgeGraphConfig] = None) -> bool:
    """Initialize the knowledge graph system with optional configuration"""
    try:
        manager = KnowledgeGraphManager(config)
        success = await manager.initialize()
        
        if success:
            set_knowledge_graph_manager(manager)
            
            # Build initial graph if it doesn't exist
            stats = await manager.neo4j_manager.get_graph_statistics()
            if stats.get("total_nodes", 0) == 0:
                await manager.build_initial_graph()
        
        return success
        
    except Exception as e:
        logging.error(f"Failed to initialize knowledge graph system: {e}")
        return False


async def shutdown_knowledge_graph_system():
    """Shutdown the knowledge graph system"""
    manager = get_knowledge_graph_manager()
    if manager:
        await manager.shutdown()
        set_knowledge_graph_manager(None)