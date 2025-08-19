"""
Data Lineage Tracker
===================

Tracks data lineage across the entire SutazAI system, providing
comprehensive visibility into data flows, transformations, and dependencies.
"""

import asyncio
import logging
import json
from enum import Enum
from dataclasses import dataclass, field
import hashlib

from ..knowledge_graph.neo4j_manager import Neo4jManager


class LineageEventType(Enum):
    """Types of lineage events"""
    DATA_CREATION = "data_creation"
    DATA_READ = "data_read"
    DATA_TRANSFORMATION = "data_transformation"
    DATA_AGGREGATION = "data_aggregation"
    DATA_DELETION = "data_deletion"
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    DATA_COPY = "data_copy"
    SCHEMA_CHANGE = "schema_change"


class LineageNodeType(Enum):
    """Types of nodes in the lineage graph"""
    DATASET = "dataset"
    TABLE = "table"
    COLUMN = "column"
    FILE = "file"
    API_ENDPOINT = "api_endpoint"
    SERVICE = "service"
    PROCESS = "process"
    USER = "user"
    SYSTEM = "system"
    AI_MODEL = "ai_model"


@dataclass
class LineageNode:
    """Represents a node in the data lineage graph"""
    id: str
    name: str
    type: LineageNodeType
    
    # Location information
    source_system: str = ""
    database_name: Optional[str] = None
    schema_name: Optional[str] = None
    table_name: Optional[str] = None
    
    # Metadata
    description: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Quality metrics
    quality_score: Optional[float] = None
    data_volume: Optional[int] = None
    
    def __post_init__(self):
        """Generate ID if not provided"""
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique ID for the node"""
        components = [
            self.source_system,
            self.database_name or "",
            self.schema_name or "", 
            self.table_name or "",
            self.name
        ]
        content = "|".join(str(c) for c in components)
        return hashlib.md5(content.encode()).hexdigest()[:16]


@dataclass
class LineageEdge:
    """Represents a relationship/edge in the data lineage graph"""
    id: str
    source_node_id: str
    target_node_id: str
    event_type: LineageEventType
    
    # Process information
    process_name: Optional[str] = None
    process_id: Optional[str] = None
    transformation_type: Optional[str] = None
    
    # Timing information
    timestamp: datetime = field(default_factory=datetime.utcnow)
    execution_time_ms: Optional[int] = None
    
    # Metadata
    properties: Dict[str, Any] = field(default_factory=dict)
    transformation_logic: Optional[str] = None
    
    # User/system context
    user_id: Optional[str] = None
    service_name: Optional[str] = None
    
    def __post_init__(self):
        """Generate ID if not provided"""
        if not self.id:
            self.id = f"{self.source_node_id}_{self.target_node_id}_{self.event_type.value}_{int(self.timestamp.timestamp())}"


@dataclass
class LineagePath:
    """Represents a complete lineage path from source to target"""
    source_node_id: str
    target_node_id: str
    path: List[str]  # List of node IDs in the path
    edges: List[LineageEdge]
    
    # Path metrics
    path_length: int = 0
    total_transformations: int = 0
    path_confidence: float = 1.0
    
    # Timing information
    earliest_timestamp: Optional[datetime] = None
    latest_timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Calculate derived properties"""
        self.path_length = len(self.path)
        self.total_transformations = len([e for e in self.edges 
                                        if e.event_type == LineageEventType.DATA_TRANSFORMATION])
        
        if self.edges:
            timestamps = [e.timestamp for e in self.edges]
            self.earliest_timestamp = min(timestamps)
            self.latest_timestamp = max(timestamps)


class DataLineageTracker:
    """
    Comprehensive data lineage tracking system that captures and maintains
    data relationships and transformations across the SutazAI system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("lineage_tracker")
        
        # Neo4j integration for lineage graph storage
        self.neo4j_config = self.config.get('neo4j', {})
        self.neo4j_manager: Optional[Neo4jManager] = None
        
        # In-memory caches
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: Dict[str, LineageEdge] = {}
        
        # Configuration
        self.max_lineage_depth = self.config.get('max_lineage_depth', 20)
        self.batch_size = self.config.get('batch_size', 1000)
        self.cache_ttl_hours = self.config.get('cache_ttl_hours', 24)
        
        # Processing queues
        self.node_queue: asyncio.Queue = asyncio.Queue()
        self.edge_queue: asyncio.Queue = asyncio.Queue()
        
        # Statistics
        self.stats = {
            "nodes_tracked": 0,
            "relationships_tracked": 0,
            "lineage_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def initialize(self) -> bool:
        """Initialize the lineage tracker"""
        try:
            self.logger.info("Initializing data lineage tracker")
            
            # Initialize Neo4j connection if configured
            if self.neo4j_config:
                self.neo4j_manager = Neo4jManager(
                    uri=self.neo4j_config.get('uri', 'bolt://localhost:7687'),
                    username=self.neo4j_config.get('username', 'neo4j'),
                    password=self.neo4j_config.get('password', 'password'),
                    database=self.neo4j_config.get('database', 'lineage')
                )
                
                success = await self.neo4j_manager.initialize()
                if not success:
                    self.logger.warning("Failed to initialize Neo4j for lineage tracking")
                    self.neo4j_manager = None
            
            # Start background processing
            await self._start_background_tasks()
            
            self.logger.info("Data lineage tracker initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize lineage tracker: {e}")
            return False
    
    async def register_node(self, node: LineageNode) -> bool:
        """Register a new node in the lineage graph"""
        try:
            # Update timestamps
            node.updated_at = datetime.utcnow()
            
            # Store in cache
            self.nodes[node.id] = node
            
            # Queue for persistent storage
            await self.node_queue.put(node)
            
            self.stats["nodes_tracked"] += 1
            self.logger.debug(f"Registered lineage node: {node.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register lineage node {node.id}: {e}")
            return False
    
    async def record_data_flow(self, source_id: str, target_id: str, 
                             event_type: LineageEventType,
                             process_name: Optional[str] = None,
                             transformation_logic: Optional[str] = None,
                             properties: Optional[Dict[str, Any]] = None,
                             user_id: Optional[str] = None) -> bool:
        """Record a data flow relationship between two nodes"""
        try:
            edge = LineageEdge(
                id="",  # Will be auto-generated
                source_node_id=source_id,
                target_node_id=target_id,
                event_type=event_type,
                process_name=process_name,
                transformation_logic=transformation_logic,
                properties=properties or {},
                user_id=user_id
            )
            
            # Store in cache
            self.edges[edge.id] = edge
            
            # Queue for persistent storage
            await self.edge_queue.put(edge)
            
            self.stats["relationships_tracked"] += 1
            self.logger.debug(f"Recorded data flow: {source_id} -> {target_id} ({event_type.value})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record data flow {source_id} -> {target_id}: {e}")
            return False
    
    async def get_upstream_lineage(self, node_id: str, max_depth: Optional[int] = None) -> List[LineagePath]:
        """Get upstream lineage (sources) for a node"""
        return await self._trace_lineage(node_id, direction="upstream", max_depth=max_depth)
    
    async def get_downstream_lineage(self, node_id: str, max_depth: Optional[int] = None) -> List[LineagePath]:
        """Get downstream lineage (targets) for a node"""
        return await self._trace_lineage(node_id, direction="downstream", max_depth=max_depth)
    
    async def get_full_lineage(self, node_id: str, max_depth: Optional[int] = None) -> Dict[str, List[LineagePath]]:
        """Get both upstream and downstream lineage for a node"""
        upstream = await self.get_upstream_lineage(node_id, max_depth)
        downstream = await self.get_downstream_lineage(node_id, max_depth)
        
        return {
            "upstream": upstream,
            "downstream": downstream,
            "node_id": node_id,
            "query_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _trace_lineage(self, node_id: str, direction: str, max_depth: Optional[int] = None) -> List[LineagePath]:
        """Trace lineage in specified direction"""
        if max_depth is None:
            max_depth = self.max_lineage_depth
        
        self.stats["lineage_queries"] += 1
        
        try:
            # Use Neo4j if available for complex graph queries
            if self.neo4j_manager:
                return await self._trace_lineage_neo4j(node_id, direction, max_depth)
            else:
                return await self._trace_lineage_memory(node_id, direction, max_depth)
                
        except Exception as e:
            self.logger.error(f"Error tracing {direction} lineage for {node_id}: {e}")
            # Return empty list on lineage tracing error
            return []  # Valid empty list: Lineage tracing failed, no paths found
    
    async def _trace_lineage_neo4j(self, node_id: str, direction: str, max_depth: int) -> List[LineagePath]:
        """Trace lineage using Neo4j graph database"""
        paths = []
        
        try:
            # Build Cypher query based on direction
            if direction == "upstream":
                cypher = f"""
                MATCH path = (target {{id: $node_id}})<-[*1..{max_depth}]-(source)
                WHERE NOT (source)<-[]-()
                RETURN path, length(path) as path_length
                ORDER BY path_length
                LIMIT 1000
                """
            else:  # downstream
                cypher = f"""
                MATCH path = (source {{id: $node_id}})-[*1..{max_depth}]->(target)
                WHERE NOT (target)-[]->()
                RETURN path, length(path) as path_length
                ORDER BY path_length
                LIMIT 1000
                """
            
            results = await self.neo4j_manager.execute_cypher(cypher, {"node_id": node_id})
            
            for result in results:
                path_data = result.get("path", {})
                if path_data:
                    lineage_path = self._convert_neo4j_path_to_lineage_path(path_data)
                    if lineage_path:
                        paths.append(lineage_path)
            
            self.stats["cache_misses"] += 1
            
        except Exception as e:
            self.logger.error(f"Error in Neo4j lineage query: {e}")
        
        return paths
    
    async def _trace_lineage_memory(self, node_id: str, direction: str, max_depth: int) -> List[LineagePath]:
        """Trace lineage using in-memory data structures"""
        paths = []
        visited = set()
        
        def find_paths(current_node: str, current_path: List[str], current_edges: List[LineageEdge], depth: int):
            if depth >= max_depth or current_node in visited:
                return
            
            visited.add(current_node)
            
            # Find connected edges
            connected_edges = []
            for edge in self.edges.values():
                if direction == "upstream" and edge.target_node_id == current_node:
                    connected_edges.append((edge.source_node_id, edge))
                elif direction == "downstream" and edge.source_node_id == current_node:
                    connected_edges.append((edge.target_node_id, edge))
            
            if not connected_edges:
                # End of path - create LineagePath
                if len(current_path) > 1:
                    if direction == "upstream":
                        source_id = current_path[-1]
                        target_id = current_path[0]
                    else:
                        source_id = current_path[0]
                        target_id = current_path[-1]
                    
                    path = LineagePath(
                        source_node_id=source_id,
                        target_node_id=target_id,
                        path=current_path.copy(),
                        edges=current_edges.copy()
                    )
                    paths.append(path)
                return
            
            # Continue tracing
            for next_node, edge in connected_edges:
                if next_node not in current_path:  # Avoid cycles
                    find_paths(
                        next_node,
                        current_path + [next_node],
                        current_edges + [edge],
                        depth + 1
                    )
            
            visited.remove(current_node)
        
        # Start tracing
        find_paths(node_id, [node_id], [], 0)
        
        self.stats["cache_hits"] += 1
        return paths
    
    def _convert_neo4j_path_to_lineage_path(self, path_data: Dict[str, Any]) -> Optional[LineagePath]:
        """Convert Neo4j path result to LineagePath object"""
        try:
            # This would extract nodes and relationships from Neo4j path
            # Simplified implementation
            nodes = path_data.get("nodes", [])
            relationships = path_data.get("relationships", [])
            
            if len(nodes) < 2:
                return None
            
            # Extract node IDs
            node_ids = [node.get("id") for node in nodes]
            
            # Convert relationships to LineageEdge objects
            edges = []
            for rel in relationships:
                edge = LineageEdge(
                    id=rel.get("id", ""),
                    source_node_id=rel.get("source_id", ""),
                    target_node_id=rel.get("target_id", ""),
                    event_type=LineageEventType(rel.get("event_type", "data_read")),
                    timestamp=datetime.fromisoformat(rel.get("timestamp", datetime.utcnow().isoformat()))
                )
                edges.append(edge)
            
            return LineagePath(
                source_node_id=node_ids[0],
                target_node_id=node_ids[-1],
                path=node_ids,
                edges=edges
            )
            
        except Exception as e:
            self.logger.error(f"Error converting Neo4j path: {e}")
            return None
    
    async def find_data_dependencies(self, node_id: str) -> Dict[str, Any]:
        """Find all data dependencies for a node"""
        try:
            upstream = await self.get_upstream_lineage(node_id)
            downstream = await self.get_downstream_lineage(node_id)
            
            # Extract unique dependencies
            upstream_nodes = set()
            downstream_nodes = set()
            
            for path in upstream:
                upstream_nodes.update(path.path[:-1])  # Exclude the target node itself
            
            for path in downstream:
                downstream_nodes.update(path.path[1:])  # Exclude the source node itself
            
            # Get node details
            upstream_details = [self.nodes.get(node_id, {"id": node_id, "name": "Unknown"}) 
                             for node_id in upstream_nodes]
            downstream_details = [self.nodes.get(node_id, {"id": node_id, "name": "Unknown"}) 
                                for node_id in downstream_nodes]
            
            return {
                "node_id": node_id,
                "upstream_dependencies": {
                    "count": len(upstream_nodes),
                    "nodes": upstream_details
                },
                "downstream_dependencies": {
                    "count": len(downstream_nodes),
                    "nodes": downstream_details
                },
                "total_dependencies": len(upstream_nodes) + len(downstream_nodes),
                "query_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error finding dependencies for {node_id}: {e}")
            return {"error": str(e)}
    
    async def analyze_data_impact(self, node_id: str, change_type: str = "modification") -> Dict[str, Any]:
        """Analyze the potential impact of changes to a data node"""
        try:
            downstream = await self.get_downstream_lineage(node_id)
            
            impact_analysis = {
                "node_id": node_id,
                "change_type": change_type,
                "potentially_affected_nodes": set(),
                "affected_systems": set(),
                "affected_processes": set(),
                "risk_level": "low",
                "recommendations": []
            }
            
            # Analyze downstream impacts
            for path in downstream:
                impact_analysis["potentially_affected_nodes"].update(path.path)
                
                # Extract affected systems and processes
                for edge in path.edges:
                    if edge.process_name:
                        impact_analysis["affected_processes"].add(edge.process_name)
                    if edge.service_name:
                        impact_analysis["affected_systems"].add(edge.service_name)
            
            # Determine risk level
            affected_count = len(impact_analysis["potentially_affected_nodes"])
            if affected_count > 50:
                impact_analysis["risk_level"] = "critical"
            elif affected_count > 20:
                impact_analysis["risk_level"] = "high"
            elif affected_count > 5:
                impact_analysis["risk_level"] = "medium"
            
            # Generate recommendations
            if impact_analysis["risk_level"] in ["high", "critical"]:
                impact_analysis["recommendations"].extend([
                    "Perform comprehensive testing before implementing changes",
                    "Notify downstream system owners",
                    "Consider staged rollout"
                ])
            
            # Convert sets to lists for JSON serialization
            impact_analysis["potentially_affected_nodes"] = list(impact_analysis["potentially_affected_nodes"])
            impact_analysis["affected_systems"] = list(impact_analysis["affected_systems"])
            impact_analysis["affected_processes"] = list(impact_analysis["affected_processes"])
            
            impact_analysis["analysis_timestamp"] = datetime.utcnow().isoformat()
            
            return impact_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing impact for {node_id}: {e}")
            return {"error": str(e)}
    
    async def get_lineage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive lineage statistics"""
        stats = {
            "total_nodes": len(self.nodes),
            "total_relationships": len(self.edges),
            "nodes_by_type": {},
            "relationships_by_type": {},
            "most_connected_nodes": [],
            "orphaned_nodes": [],
            "system_statistics": self.stats.copy(),
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Count nodes by type
        for node in self.nodes.values():
            node_type = node.type.value
            stats["nodes_by_type"][node_type] = stats["nodes_by_type"].get(node_type, 0) + 1
        
        # Count relationships by type
        for edge in self.edges.values():
            event_type = edge.event_type.value
            stats["relationships_by_type"][event_type] = stats["relationships_by_type"].get(event_type, 0) + 1
        
        # Find most connected nodes
        node_connections = {}
        for edge in self.edges.values():
            node_connections[edge.source_node_id] = node_connections.get(edge.source_node_id, 0) + 1
            node_connections[edge.target_node_id] = node_connections.get(edge.target_node_id, 0) + 1
        
        most_connected = sorted(node_connections.items(), key=lambda x: x[1], reverse=True)[:10]
        stats["most_connected_nodes"] = [
            {"node_id": node_id, "connection_count": count, 
             "node_name": self.nodes.get(node_id, {}).get("name", "Unknown")}
            for node_id, count in most_connected
        ]
        
        # Find orphaned nodes (no connections)
        connected_nodes = set(node_connections.keys())
        all_nodes = set(self.nodes.keys())
        orphaned = all_nodes - connected_nodes
        
        stats["orphaned_nodes"] = [
            {"node_id": node_id, "node_name": self.nodes[node_id].name}
            for node_id in orphaned
        ]
        
        return stats
    
    async def _start_background_tasks(self):
        """Start background processing tasks"""
        
        async def process_node_queue():
            """Process queued nodes for persistent storage"""
            while True:
                try:
                    node = await self.node_queue.get()
                    
                    # Store in Neo4j if available
                    if self.neo4j_manager:
                        await self._store_node_in_neo4j(node)
                    
                    self.node_queue.task_done()
                    
                except Exception as e:
                    self.logger.error(f"Error processing node queue: {e}")
        
        async def process_edge_queue():
            """Process queued edges for persistent storage"""
            while True:
                try:
                    edge = await self.edge_queue.get()
                    
                    # Store in Neo4j if available
                    if self.neo4j_manager:
                        await self._store_edge_in_neo4j(edge)
                    
                    self.edge_queue.task_done()
                    
                except Exception as e:
                    self.logger.error(f"Error processing edge queue: {e}")
        
        # Start background tasks
        asyncio.create_task(process_node_queue())
        asyncio.create_task(process_edge_queue())
        
        self.logger.info("Started lineage tracker background tasks")
    
    async def _store_node_in_neo4j(self, node: LineageNode):
        """Store a node in Neo4j"""
        try:
            cypher = """
            MERGE (n:LineageNode {id: $id})
            SET n.name = $name,
                n.type = $type,
                n.source_system = $source_system,
                n.database_name = $database_name,
                n.schema_name = $schema_name,
                n.table_name = $table_name,
                n.description = $description,
                n.created_at = $created_at,
                n.updated_at = $updated_at,
                n.quality_score = $quality_score,
                n.data_volume = $data_volume,
                n.properties = $properties,
                n.tags = $tags
            """
            
            params = {
                "id": node.id,
                "name": node.name,
                "type": node.type.value,
                "source_system": node.source_system,
                "database_name": node.database_name,
                "schema_name": node.schema_name,
                "table_name": node.table_name,
                "description": node.description,
                "created_at": node.created_at.isoformat(),
                "updated_at": node.updated_at.isoformat(),
                "quality_score": node.quality_score,
                "data_volume": node.data_volume,
                "properties": json.dumps(node.properties),
                "tags": node.tags
            }
            
            await self.neo4j_manager.execute_cypher(cypher, params)
            
        except Exception as e:
            self.logger.error(f"Error storing node {node.id} in Neo4j: {e}")
    
    async def _store_edge_in_neo4j(self, edge: LineageEdge):
        """Store an edge in Neo4j"""
        try:
            cypher = """
            MATCH (source:LineageNode {id: $source_id})
            MATCH (target:LineageNode {id: $target_id})
            CREATE (source)-[r:DATA_FLOW {
                id: $id,
                event_type: $event_type,
                process_name: $process_name,
                process_id: $process_id,
                transformation_type: $transformation_type,
                timestamp: $timestamp,
                execution_time_ms: $execution_time_ms,
                transformation_logic: $transformation_logic,
                user_id: $user_id,
                service_name: $service_name,
                properties: $properties
            }]->(target)
            """
            
            params = {
                "source_id": edge.source_node_id,
                "target_id": edge.target_node_id,
                "id": edge.id,
                "event_type": edge.event_type.value,
                "process_name": edge.process_name,
                "process_id": edge.process_id,
                "transformation_type": edge.transformation_type,
                "timestamp": edge.timestamp.isoformat(),
                "execution_time_ms": edge.execution_time_ms,
                "transformation_logic": edge.transformation_logic,
                "user_id": edge.user_id,
                "service_name": edge.service_name,
                "properties": json.dumps(edge.properties)
            }
            
            await self.neo4j_manager.execute_cypher(cypher, params)
            
        except Exception as e:
            self.logger.error(f"Error storing edge {edge.id} in Neo4j: {e}")
    
    async def shutdown(self):
        """Shutdown the lineage tracker"""
        try:
            self.logger.info("Shutting down data lineage tracker")
            
            # Wait for queues to empty
            await self.node_queue.join()
            await self.edge_queue.join()
            
            # Shutdown Neo4j connection
            if self.neo4j_manager:
                await self.neo4j_manager.shutdown()
            
            self.logger.info("Data lineage tracker shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during lineage tracker shutdown: {e}")
    
    # Convenience methods for common use cases
    
    async def track_database_table(self, database: str, schema: str, table: str,
                                 source_system: str = "sutazai") -> str:
        """Register a database table as a lineage node"""
        node = LineageNode(
            id="",  # Will be auto-generated
            name=f"{schema}.{table}",
            type=LineageNodeType.TABLE,
            source_system=source_system,
            database_name=database,
            schema_name=schema,
            table_name=table
        )
        
        await self.register_node(node)
        return node.id
    
    async def track_ai_model(self, model_name: str, version: str,
                           training_data_ids: List[str]) -> str:
        """Register an AI model and its training data dependencies"""
        # Register model node
        model_node = LineageNode(
            id="",  # Will be auto-generated
            name=f"{model_name}_v{version}",
            type=LineageNodeType.AI_MODEL,
            source_system="sutazai",
            properties={"version": version, "model_type": "ai_model"}
        )
        
        await self.register_node(model_node)
        
        # Record dependencies on training data
        for data_id in training_data_ids:
            await self.record_data_flow(
                source_id=data_id,
                target_id=model_node.id,
                event_type=LineageEventType.DATA_TRANSFORMATION,
                process_name="model_training",
                transformation_logic=f"Training {model_name} model"
            )
        
        return model_node.id
    
    async def track_api_endpoint(self, endpoint: str, service_name: str,
                               data_sources: List[str]) -> str:
        """Register an API endpoint and its data dependencies"""
        node = LineageNode(
            id="",  # Will be auto-generated
            name=endpoint,
            type=LineageNodeType.API_ENDPOINT,
            source_system=service_name,
            properties={"endpoint": endpoint, "service": service_name}
        )
        
        await self.register_node(node)
        
        # Record data dependencies
        for source_id in data_sources:
            await self.record_data_flow(
                source_id=source_id,
                target_id=node.id,
                event_type=LineageEventType.DATA_READ,
                process_name="api_response_generation",
                service_name=service_name
            )
        
        return node.id