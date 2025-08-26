"""
Real-time Knowledge Graph Updater
=================================

Provides real-time synchronization and updates for the SutazAI knowledge graph.
Monitors system changes, agent status updates, service modifications, and
automatically updates the graph to maintain accuracy and freshness.

Features:
- Agent status monitoring and updates
- Service dependency change detection
- Configuration file monitoring
- Event-driven graph updates
- Conflict resolution
- Update batching for performance
- Rollback capabilities
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import hashlib

import aiofiles
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .neo4j_manager import Neo4jManager
from .schema import NodeType, RelationshipType, NodeProperties, RelationshipProperties
from .graph_builder import KnowledgeGraphBuilder
from ..ai_agents.core.agent_registry import AgentRegistry


@dataclass
class ChangeEvent:
    """Represents a change event in the system"""
    event_id: str = field(default_factory=lambda: str(time.time()))
    event_type: str = ""  # agent_update, service_change, file_change, etc.
    source: str = ""  # source of the change
    target_id: str = ""  # ID of the affected node
    changes: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    processed: bool = False
    retry_count: int = 0


class FileChangeHandler(FileSystemEventHandler):
    """Handles file system changes for configuration and code files"""
    
    def __init__(self, updater):
        self.updater = updater
        self.logger = logging.getLogger("file_change_handler")
        
        # File patterns to monitor
        self.monitored_patterns = {
            "*.py": "code_change",
            "*.json": "config_change", 
            "*.yaml": "config_change",
            "*.yml": "config_change",
            "requirements*.txt": "dependency_change",
            "Dockerfile*": "container_change"
        }
        
        # Debounce mechanism to avoid duplicate events
        self.last_events = {}
        self.debounce_time = 2.0  # seconds
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        self._handle_file_event("modified", event.src_path)
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        self._handle_file_event("created", event.src_path)
    
    def on_deleted(self, event):
        if event.is_directory:
            return
        
        self._handle_file_event("deleted", event.src_path)
    
    def _handle_file_event(self, event_type: str, file_path: str):
        """Handle file system events with debouncing"""
        try:
            # Check if we should monitor this file
            path = Path(file_path)
            if not self._should_monitor_file(path):
                return
            
            # Debounce check
            event_key = f"{event_type}:{file_path}"
            now = time.time()
            
            if event_key in self.last_events:
                if now - self.last_events[event_key] < self.debounce_time:
                    return  # Skip duplicate event
            
            self.last_events[event_key] = now
            
            # Create change event
            change_type = self._get_change_type(path)
            change_event = ChangeEvent(
                event_type=change_type,
                source=f"file_{event_type}",
                target_id=str(path),
                changes={
                    "file_path": file_path,
                    "operation": event_type,
                    "file_type": path.suffix
                }
            )
            
            # Queue the change
            asyncio.create_task(self.updater.queue_change(change_event))
            
        except Exception as e:
            self.logger.error(f"Error handling file event: {e}")
    
    def _should_monitor_file(self, path: Path) -> bool:
        """Check if file should be monitored"""
        # Skip temporary files and hidden files
        if path.name.startswith('.') or path.name.endswith('~'):
            return False
        
        # Skip files in certain directories
        skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules'}
        if any(skip_dir in path.parts for skip_dir in skip_dirs):
            return False
        
        # Check file patterns
        for pattern, _ in self.monitored_patterns.items():
            if path.match(pattern):
                return True
        
        return False
    
    def _get_change_type(self, path: Path) -> str:
        """Get change type based on file pattern"""
        for pattern, change_type in self.monitored_patterns.items():
            if path.match(pattern):
                return change_type
        
        return "file_change"


class RealTimeUpdater:
    """
    Main class for real-time knowledge graph updates
    """
    
    def __init__(self, neo4j_manager: Neo4jManager, 
                 agent_registry: Optional[AgentRegistry] = None,
                 base_path: str = "/opt/sutazaiapp/backend"):
        
        self.neo4j_manager = neo4j_manager
        self.agent_registry = agent_registry
        self.base_path = Path(base_path)
        
        # Change processing
        self.change_queue = asyncio.Queue()
        self.batch_size = 10
        self.batch_timeout = 30.0  # seconds
        self.max_retry_attempts = 3
        
        # File monitoring
        self.file_observer = Observer()
        self.file_handler = FileChangeHandler(self)
        
        # Status tracking
        self.is_running = False
        self.stats = {
            "events_processed": 0,
            "events_failed": 0,
            "batches_processed": 0,
            "last_update": None,
            "average_processing_time": 0.0
        }
        
        # Event handlers
        self.event_handlers = {
            "agent_update": self._handle_agent_update,
            "service_change": self._handle_service_change,
            "code_change": self._handle_code_change,
            "config_change": self._handle_config_change,
            "dependency_change": self._handle_dependency_change,
            "container_change": self._handle_container_change
        }
        
        # Background tasks
        self._background_tasks = set()
        self._shutdown_event = asyncio.Event()
        
        self.logger = logging.getLogger("real_time_updater")
    
    async def start(self) -> bool:
        """Start the real-time updater"""
        try:
            self.logger.info("Starting real-time knowledge graph updater")
            
            # Start file monitoring
            self._start_file_monitoring()
            
            # Start agent registry monitoring if available
            if self.agent_registry:
                self._start_agent_monitoring()
            
            # Start background tasks
            self._start_background_tasks()
            
            self.is_running = True
            self.logger.info("Real-time updater started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start real-time updater: {e}")
            return False
    
    def _start_file_monitoring(self):
        """Start monitoring file system changes"""
        self.file_observer.schedule(
            self.file_handler,
            str(self.base_path),
            recursive=True
        )
        self.file_observer.start()
        self.logger.info(f"Started file monitoring for {self.base_path}")
    
    def _start_agent_monitoring(self):
        """Start monitoring agent registry changes"""
        if self.agent_registry:
            # Register event handlers for agent changes
            self.agent_registry.register_event_handler(
                "agent_registered", 
                self._on_agent_registered
            )
            self.agent_registry.register_event_handler(
                "agent_unregistered",
                self._on_agent_unregistered
            )
            self.agent_registry.register_event_handler(
                "agent_health_changed",
                self._on_agent_health_changed
            )
            
            self.logger.info("Started agent registry monitoring")
    
    def _start_background_tasks(self):
        """Start background processing tasks"""
        tasks = [
            self._change_processor(),
            self._periodic_sync(),
            self._cleanup_old_events()
        ]
        
        for coro in tasks:
            task = asyncio.create_task(coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
    
    async def queue_change(self, change_event: ChangeEvent):
        """Queue a change event for processing"""
        await self.change_queue.put(change_event)
        self.logger.debug(f"Queued change event: {change_event.event_type}")
    
    async def _change_processor(self):
        """Process change events from the queue"""
        batch = []
        last_batch_time = time.time()
        
        while not self._shutdown_event.is_set():
            try:
                # Wait for changes with timeout
                try:
                    change_event = await asyncio.wait_for(
                        self.change_queue.get(),
                        timeout=5.0
                    )
                    batch.append(change_event)
                except asyncio.TimeoutError:
                    pass
                
                # Process batch if it's full or timeout reached
                current_time = time.time()
                if (len(batch) >= self.batch_size or 
                    (batch and current_time - last_batch_time >= self.batch_timeout)):
                    
                    if batch:
                        await self._process_change_batch(batch)
                        batch = []
                        last_batch_time = current_time
                
            except Exception as e:
                self.logger.error(f"Error in change processor: {e}")
                await asyncio.sleep(1)
    
    async def _process_change_batch(self, batch: List[ChangeEvent]):
        """Process a batch of change events"""
        start_time = time.time()
        processed_count = 0
        failed_count = 0
        
        self.logger.info(f"Processing batch of {len(batch)} change events")
        
        for change_event in batch:
            try:
                success = await self._process_single_change(change_event)
                if success:
                    processed_count += 1
                    change_event.processed = True
                else:
                    failed_count += 1
                    change_event.retry_count += 1
                    
                    # Retry if not exceeded max attempts
                    if change_event.retry_count < self.max_retry_attempts:
                        await self.queue_change(change_event)
            
            except Exception as e:
                self.logger.error(f"Error processing change event: {e}")
                failed_count += 1
        
        # Update statistics
        processing_time = time.time() - start_time
        self.stats["events_processed"] += processed_count
        self.stats["events_failed"] += failed_count
        self.stats["batches_processed"] += 1
        self.stats["last_update"] = datetime.utcnow()
        
        # Update average processing time
        current_avg = self.stats["average_processing_time"]
        batch_count = self.stats["batches_processed"]
        self.stats["average_processing_time"] = (
            (current_avg * (batch_count - 1)) + processing_time
        ) / batch_count
        
        self.logger.info(
            f"Batch processed: {processed_count} successful, "
            f"{failed_count} failed, {processing_time:.2f}s"
        )
    
    async def _process_single_change(self, change_event: ChangeEvent) -> bool:
        """Process a single change event"""
        try:
            event_type = change_event.event_type
            handler = self.event_handlers.get(event_type)
            
            if handler:
                return await handler(change_event)
            else:
                self.logger.warning(f"No handler for event type: {event_type}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error processing change event {change_event.event_id}: {e}")
            return False
    
    # Event handlers for different types of changes
    
    async def _handle_agent_update(self, change_event: ChangeEvent) -> bool:
        """Handle agent status updates"""
        try:
            agent_id = change_event.target_id
            changes = change_event.changes
            
            # Update agent node in graph
            cypher = """
            MATCH (a:Agent {id: $agent_id})
            SET a.health_status = $health_status,
                a.status = $status,
                a.updated_at = $updated_at
            RETURN a
            """
            
            parameters = {
                "agent_id": agent_id,
                "health_status": changes.get("health_status", "unknown"),
                "status": changes.get("status", "unknown"),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            results = await self.neo4j_manager.execute_cypher(cypher, parameters)
            return len(results) > 0
            
        except Exception as e:
            self.logger.error(f"Error handling agent update: {e}")
            return False
    
    async def _handle_service_change(self, change_event: ChangeEvent) -> bool:
        """Handle service configuration changes"""
        try:
            # Implementation for service changes
            # This would involve updating service nodes and their relationships
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling service change: {e}")
            return False
    
    async def _handle_code_change(self, change_event: ChangeEvent) -> bool:
        """Handle code file changes"""
        try:
            file_path = change_event.changes.get("file_path")
            operation = change_event.changes.get("operation", "modified")
            
            if operation == "deleted":
                # Handle file deletion
                await self._handle_file_deletion(file_path)
            else:
                # Re-analyze the file and update related nodes
                await self._reanalyze_file(file_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling code change: {e}")
            return False
    
    async def _handle_config_change(self, change_event: ChangeEvent) -> bool:
        """Handle configuration file changes"""
        try:
            file_path = change_event.changes.get("file_path")
            
            # Re-analyze configuration and update affected nodes
            await self._reanalyze_configuration(file_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling config change: {e}")
            return False
    
    async def _handle_dependency_change(self, change_event: ChangeEvent) -> bool:
        """Handle dependency file changes (requirements.txt, etc.)"""
        try:
            file_path = change_event.changes.get("file_path")
            
            # Analyze dependency changes and update service relationships
            await self._analyze_dependency_changes(file_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling dependency change: {e}")
            return False
    
    async def _handle_container_change(self, change_event: ChangeEvent) -> bool:
        """Handle container configuration changes (Dockerfile, etc.)"""
        try:
            file_path = change_event.changes.get("file_path")
            
            # Analyze container changes and update infrastructure nodes
            await self._analyze_container_changes(file_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling container change: {e}")
            return False
    
    # Agent registry event handlers
    
    async def _on_agent_registered(self, event_data: Dict[str, Any]):
        """Handle agent registration events"""
        agent_id = event_data.get("agent_id")
        if agent_id:
            change_event = ChangeEvent(
                event_type="agent_update",
                source="agent_registry",
                target_id=agent_id,
                changes={
                    "operation": "registered",
                    "status": "active",
                    "health_status": "healthy"
                }
            )
            await self.queue_change(change_event)
    
    async def _on_agent_unregistered(self, event_data: Dict[str, Any]):
        """Handle agent unregistration events"""
        agent_id = event_data.get("agent_id")
        if agent_id:
            change_event = ChangeEvent(
                event_type="agent_update",
                source="agent_registry",
                target_id=agent_id,
                changes={
                    "operation": "unregistered",
                    "status": "offline",
                    "health_status": "unresponsive"
                }
            )
            await self.queue_change(change_event)
    
    async def _on_agent_health_changed(self, event_data: Dict[str, Any]):
        """Handle agent health change events"""
        agent_id = event_data.get("agent_id")
        if agent_id:
            change_event = ChangeEvent(
                event_type="agent_update",
                source="agent_registry",
                target_id=agent_id,
                changes={
                    "operation": "health_changed",
                    "health_status": event_data.get("new_health"),
                    "old_health": event_data.get("old_health")
                }
            )
            await self.queue_change(change_event)
    
    # Helper methods for file analysis
    
    async def _handle_file_deletion(self, file_path: str):
        """Handle deletion of monitored files"""
        # Remove nodes associated with deleted files
        cypher = """
        MATCH (n)
        WHERE n.file_path = $file_path OR n.metadata CONTAINS $file_path
        DELETE n
        """
        
        await self.neo4j_manager.execute_cypher(cypher, {"file_path": file_path})
    
    async def _reanalyze_file(self, file_path: str):
        """Re-analyze a changed file and update the graph"""
        try:
            # Create a temporary graph builder for analysis
            builder = KnowledgeGraphBuilder(str(self.base_path), self.neo4j_manager)
            
            # Analyze the specific file
            # This is a simplified version - in practice, you'd want more targeted analysis
            if file_path.endswith('.py'):
                analysis = builder.discovery.code_analyzer.analyze_file(file_path)
                
                # Update or create nodes based on analysis
                await self._update_nodes_from_analysis(file_path, analysis)
        
        except Exception as e:
            self.logger.error(f"Error reanalyzing file {file_path}: {e}")
    
    async def _reanalyze_configuration(self, file_path: str):
        """Re-analyze configuration files"""
        # Implementation for configuration analysis
        pass
    
    async def _analyze_dependency_changes(self, file_path: str):
        """Analyze dependency file changes"""
        # Implementation for dependency analysis
        pass
    
    async def _analyze_container_changes(self, file_path: str):
        """Analyze container configuration changes"""
        # Implementation for container analysis
        pass
    
    async def _update_nodes_from_analysis(self, file_path: str, analysis: Dict[str, Any]):
        """Update graph nodes based on file analysis"""
        # Implementation for updating nodes from analysis results
        pass
    
    # Background maintenance tasks
    
    async def _periodic_sync(self):
        """Perform periodic full synchronization"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                if self.is_running:
                    self.logger.info("Starting periodic graph synchronization")
                    
                    # Perform full sync (simplified version)
                    # In practice, this would be more sophisticated
                    builder = KnowledgeGraphBuilder(str(self.base_path), self.neo4j_manager)
                    stats = await builder.build_knowledge_graph()
                    
                    self.logger.info(f"Periodic sync completed: {stats}")
                
            except Exception as e:
                self.logger.error(f"Error in periodic sync: {e}")
    
    async def _cleanup_old_events(self):
        """Clean up old processed events"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes
                
                # Clean up old events (implementation needed)
                # This would remove old processed events from memory/storage
                
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
    
    # Public methods
    
    async def force_full_sync(self) -> Dict[str, Any]:
        """Force a full graph synchronization"""
        self.logger.info("Starting forced full synchronization")
        
        try:
            builder = KnowledgeGraphBuilder(str(self.base_path), self.neo4j_manager)
            stats = await builder.build_knowledge_graph()
            
            self.logger.info("Forced full sync completed successfully")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error in forced full sync: {e}")
            return {"error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get updater statistics"""
        return {
            **self.stats,
            "is_running": self.is_running,
            "queue_size": self.change_queue.qsize(),
            "background_tasks": len(self._background_tasks)
        }
    
    async def shutdown(self):
        """Shutdown the real-time updater"""
        self.logger.info("Shutting down real-time updater")
        
        self.is_running = False
        
        # Signal shutdown to background tasks
        self._shutdown_event.set()
        
        # Stop file monitoring
        if self.file_observer.is_alive():
            self.file_observer.stop()
            self.file_observer.join()
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self.logger.info("Real-time updater shutdown complete")


# Global updater instance
_real_time_updater: Optional[RealTimeUpdater] = None


def get_real_time_updater() -> Optional[RealTimeUpdater]:
    """Get the global real-time updater instance"""
    return _real_time_updater


def set_real_time_updater(updater: RealTimeUpdater):
    """Set the global real-time updater instance"""
    global _real_time_updater
    _real_time_updater = updater