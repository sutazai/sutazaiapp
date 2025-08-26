"""
Knowledge Graph API
==================

FastAPI endpoints for the SutazAI knowledge graph system.
Provides REST API access to graph querying, visualization,
reasoning, and real-time updates.

Endpoints:
- /kg/nodes - Node operations
- /kg/relationships - Relationship operations  
- /kg/query - Graph querying
- /kg/reasoning - Reasoning and inference
- /kg/visualization - Graph visualization
- /kg/stats - System statistics
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query as QueryParam
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

from .neo4j_manager import Neo4jManager, get_neo4j_manager
from .query_engine import QueryEngine, QueryResult
from .reasoning_engine import ReasoningEngine, get_reasoning_engine
from .visualization import VisualizationManager
from .real_time_updater import RealTimeUpdater, get_real_time_updater
from .graph_builder import KnowledgeGraphBuilder
from .schema import NodeType, RelationshipType


# Pydantic models for API

class NodeQuery(BaseModel):
    node_type: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    limit: int = Field(default=100, ge=1, le=1000)


class RelationshipQuery(BaseModel):
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    relationship_type: Optional[str] = None
    limit: int = Field(default=100, ge=1, le=1000)


class CypherQuery(BaseModel):
    cypher: str
    parameters: Optional[Dict[str, Any]] = None


class NaturalLanguageQuery(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None


class AgentCapabilityQuery(BaseModel):
    capabilities: List[str]
    min_match: int = Field(default=1, ge=1)
    include_health: bool = True


class TaskAssignmentRequest(BaseModel):
    task_description: str
    required_capabilities: List[str]
    preferred_agent_type: Optional[str] = None


class VisualizationRequest(BaseModel):
    type: str = Field(..., description="Type of visualization: agent_capabilities, service_dependencies, system_architecture, data_flow")
    source_id: Optional[str] = None
    output_format: str = Field(default="html", description="Output format: html, json")


# Router setup
router = APIRouter(prefix="/kg", tags=["knowledge-graph"])
logger = logging.getLogger("kg_api")


# Dependency functions
async def get_query_engine() -> QueryEngine:
    """Get query engine dependency"""
    neo4j_manager = get_neo4j_manager()
    if not neo4j_manager:
        raise HTTPException(status_code=503, detail="Neo4j manager not initialized")
    return QueryEngine(neo4j_manager)


async def get_visualization_manager() -> VisualizationManager:
    """Get visualization manager dependency"""
    query_engine = await get_query_engine()
    return VisualizationManager(query_engine)


# Node operations

@router.get("/nodes", response_model=Dict[str, Any])
async def find_nodes(
    node_type: Optional[str] = QueryParam(None, description="Filter by node type"),
    name: Optional[str] = QueryParam(None, description="Filter by name (partial match)"),
    limit: int = QueryParam(100, ge=1, le=1000, description="Maximum number of nodes to return"),
    query_engine: QueryEngine = Depends(get_query_engine)
):
    """Find nodes in the knowledge graph"""
    try:
        filters = {}
        if name:
            # Use Cypher CONTAINS for partial matching
            cypher = f"""
            MATCH (n{':' + node_type.title() if node_type else ''})
            WHERE toLower(n.name) CONTAINS toLower($name)
            RETURN n
            LIMIT {limit}
            """
            results = await query_engine.neo4j_manager.execute_cypher(
                cypher, {"name": name}
            )
            nodes = [record["n"] for record in results]
        else:
            node_type_enum = None
            if node_type:
                try:
                    node_type_enum = NodeType(node_type.lower())
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid node type: {node_type}")
            
            nodes = await query_engine.find_nodes(node_type_enum, filters, limit)
        
        return {
            "nodes": nodes,
            "count": len(nodes),
            "query_params": {
                "node_type": node_type,
                "name": name,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Error finding nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nodes/{node_id}", response_model=Dict[str, Any])
async def get_node(
    node_id: str,
    include_relationships: bool = QueryParam(False, description="Include related nodes and relationships"),
    query_engine: QueryEngine = Depends(get_query_engine)
):
    """Get a specific node by ID"""
    try:
        cypher = "MATCH (n {id: $node_id}) RETURN n"
        results = await query_engine.neo4j_manager.execute_cypher(
            cypher, {"node_id": node_id}
        )
        
        if not results:
            raise HTTPException(status_code=404, detail="Node not found")
        
        node = results[0]["n"]
        response = {"node": node}
        
        if include_relationships:
            # Get related nodes and relationships
            rel_cypher = """
            MATCH (n {id: $node_id})-[r]-(related)
            RETURN r, related, type(r) as rel_type
            LIMIT 50
            """
            rel_results = await query_engine.neo4j_manager.execute_cypher(
                rel_cypher, {"node_id": node_id}
            )
            
            relationships = []
            related_nodes = []
            
            for record in rel_results:
                relationships.append({
                    "relationship": record["r"],
                    "type": record["rel_type"]
                })
                related_nodes.append(record["related"])
            
            response["relationships"] = relationships
            response["related_nodes"] = related_nodes
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting node {node_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Relationship operations

@router.get("/relationships", response_model=Dict[str, Any])
async def find_relationships(
    source_id: Optional[str] = QueryParam(None, description="Source node ID"),
    target_id: Optional[str] = QueryParam(None, description="Target node ID"),
    relationship_type: Optional[str] = QueryParam(None, description="Relationship type"),
    limit: int = QueryParam(100, ge=1, le=1000),
    query_engine: QueryEngine = Depends(get_query_engine)
):
    """Find relationships in the knowledge graph"""
    try:
        rel_type_enum = None
        if relationship_type:
            try:
                rel_type_enum = RelationshipType(relationship_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid relationship type: {relationship_type}")
        
        relationships = await query_engine.find_relationships(
            source_id, target_id, rel_type_enum, limit
        )
        
        return {
            "relationships": relationships,
            "count": len(relationships),
            "query_params": {
                "source_id": source_id,
                "target_id": target_id,
                "relationship_type": relationship_type,
                "limit": limit
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Query operations

@router.post("/query/cypher", response_model=Dict[str, Any])
async def execute_cypher_query(
    query_request: CypherQuery,
    query_engine: QueryEngine = Depends(get_query_engine)
):
    """Execute a custom Cypher query"""
    try:
        # Basic security check - prevent destructive operations in production
        dangerous_keywords = ["DELETE", "REMOVE", "DROP", "CREATE", "SET", "MERGE"]
        cypher_upper = query_request.cypher.upper()
        
        if any(keyword in cypher_upper for keyword in dangerous_keywords):
            # In production, you might want to allow these only for admin users
            logger.warning(f"Potentially destructive Cypher query attempted: {query_request.cypher}")
        
        result = await query_engine.execute_custom_query(
            query_request.cypher,
            query_request.parameters
        )
        
        return {
            "result": result.nodes + result.relationships,
            "metadata": result.metadata,
            "query_time_ms": result.query_time_ms,
            "total_results": result.total_results
        }
        
    except Exception as e:
        logger.error(f"Error executing Cypher query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/natural", response_model=Dict[str, Any])
async def execute_natural_language_query(
    query_request: NaturalLanguageQuery,
    query_engine: QueryEngine = Depends(get_query_engine)
):
    """Execute a natural language query"""
    try:
        result = await query_engine.process_natural_language_query(query_request.query)
        
        return {
            "result": result.nodes + result.relationships,
            "metadata": result.metadata,
            "query_time_ms": result.query_time_ms,
            "total_results": result.total_results,
            "interpretation": {
                "intent": result.metadata.get("detected_intent"),
                "entities": result.metadata.get("extracted_entities"),
                "generated_cypher": result.metadata.get("generated_cypher")
            }
        }
        
    except Exception as e:
        logger.error(f"Error executing natural language query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query/suggestions", response_model=List[str])
async def get_query_suggestions(
    partial_query: str = QueryParam(..., description="Partial query for suggestions"),
    query_engine: QueryEngine = Depends(get_query_engine)
):
    """Get query suggestions based on partial input"""
    try:
        suggestions = query_engine.get_query_suggestions(partial_query)
        return suggestions
        
    except Exception as e:
        logger.error(f"Error getting query suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Agent-specific operations

@router.post("/agents/find-by-capability", response_model=Dict[str, Any])
async def find_agents_by_capability(
    query_request: AgentCapabilityQuery,
    query_engine: QueryEngine = Depends(get_query_engine)
):
    """Find agents with specific capabilities"""
    try:
        result = await query_engine.find_agents_by_capability(
            query_request.capabilities,
            query_request.min_match,
            query_request.include_health
        )
        
        return {
            "agents": result.nodes,
            "metadata": result.metadata,
            "query_time_ms": result.query_time_ms,
            "total_results": result.total_results
        }
        
    except Exception as e:
        logger.error(f"Error finding agents by capability: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/optimal-for-task", response_model=Dict[str, Any])
async def find_optimal_agent_for_task(
    request: TaskAssignmentRequest,
    query_engine: QueryEngine = Depends(get_query_engine)
):
    """Find optimal agent for a specific task"""
    try:
        result = await query_engine.find_optimal_agent_for_task(
            request.required_capabilities,
            request.preferred_agent_type
        )
        
        return {
            "optimal_agents": result.nodes,
            "task_description": request.task_description,
            "required_capabilities": request.required_capabilities,
            "metadata": result.metadata,
            "query_time_ms": result.query_time_ms
        }
        
    except Exception as e:
        logger.error(f"Error finding optimal agent for task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/network", response_model=Dict[str, Any])
async def get_agent_network(
    agent_id: str,
    relationship_types: Optional[List[str]] = QueryParam(None, description="Filter by relationship types"),
    max_depth: int = QueryParam(2, ge=1, le=5, description="Maximum relationship depth"),
    query_engine: QueryEngine = Depends(get_query_engine)
):
    """Get the network of relationships around an agent"""
    try:
        result = await query_engine.get_agent_network(
            agent_id,
            relationship_types,
            max_depth
        )
        
        return {
            "network_nodes": result.nodes,
            "network_relationships": result.relationships,
            "metadata": result.metadata,
            "query_time_ms": result.query_time_ms
        }
        
    except Exception as e:
        logger.error(f"Error getting agent network: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Service operations

@router.get("/services/{service_name}/dependencies", response_model=Dict[str, Any])
async def analyze_service_dependencies(
    service_name: str,
    max_depth: int = QueryParam(3, ge=1, le=5, description="Maximum dependency depth"),
    query_engine: QueryEngine = Depends(get_query_engine)
):
    """Analyze service dependencies and impact"""
    try:
        result = await query_engine.analyze_service_dependencies(service_name, max_depth)
        
        return {
            "dependencies": result.nodes,
            "impact_analysis": result.metadata.get("impact_analysis", []),
            "metadata": result.metadata,
            "query_time_ms": result.query_time_ms
        }
        
    except Exception as e:
        logger.error(f"Error analyzing service dependencies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/services/{source_name}/data-flow", response_model=Dict[str, Any])
async def trace_data_flow(
    source_name: str,
    max_hops: int = QueryParam(5, ge=1, le=10, description="Maximum data flow hops"),
    query_engine: QueryEngine = Depends(get_query_engine)
):
    """Trace data flow from a source component"""
    try:
        result = await query_engine.trace_data_flow(source_name, max_hops)
        
        return {
            "flow_paths": result.metadata.get("flow_paths", []),
            "metadata": result.metadata,
            "query_time_ms": result.query_time_ms
        }
        
    except Exception as e:
        logger.error(f"Error tracing data flow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Reasoning operations

@router.post("/reasoning/cycle", response_model=Dict[str, Any])
async def execute_reasoning_cycle(
    reasoning_engine: ReasoningEngine = Depends(get_reasoning_engine)
):
    """Execute a complete reasoning cycle"""
    try:
        if not reasoning_engine:
            raise HTTPException(status_code=503, detail="Reasoning engine not initialized")
        
        results = await reasoning_engine.perform_reasoning_cycle()
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing reasoning cycle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reasoning/task-assignment", response_model=Dict[str, Any])
async def get_task_assignment_recommendation(
    request: TaskAssignmentRequest,
    reasoning_engine: ReasoningEngine = Depends(get_reasoning_engine)
):
    """Get intelligent task assignment recommendation"""
    try:
        if not reasoning_engine:
            raise HTTPException(status_code=503, detail="Reasoning engine not initialized")
        
        result = await reasoning_engine.find_optimal_agent_for_task(
            request.task_description,
            request.required_capabilities
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task assignment recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reasoning/stats", response_model=Dict[str, Any])
async def get_reasoning_statistics(
    reasoning_engine: ReasoningEngine = Depends(get_reasoning_engine)
):
    """Get reasoning engine statistics"""
    try:
        if not reasoning_engine:
            raise HTTPException(status_code=503, detail="Reasoning engine not initialized")
        
        stats = reasoning_engine.get_reasoning_statistics()
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting reasoning statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Visualization operations

@router.post("/visualization/create", response_model=Dict[str, Any])
async def create_visualization(
    request: VisualizationRequest,
    background_tasks: BackgroundTasks,
    viz_manager: VisualizationManager = Depends(get_visualization_manager)
):
    """Create a graph visualization"""
    try:
        output_file = f"{request.type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        if request.type == "agent_capabilities":
            file_path = await viz_manager.create_agent_capability_view(output_file)
        elif request.type == "service_dependencies":
            file_path = await viz_manager.create_service_dependency_view(output_file)
        elif request.type == "system_architecture":
            file_path = await viz_manager.create_system_architecture_view(output_file)
        elif request.type == "data_flow":
            file_path = await viz_manager.create_data_flow_view(request.source_id, output_file)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid visualization type: {request.type}")
        
        return {
            "visualization_type": request.type,
            "file_path": file_path,
            "output_format": request.output_format,
            "created_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualization/{filename}", response_class=HTMLResponse)
async def serve_visualization(filename: str):
    """Serve a generated visualization file"""
    try:
        file_path = Path(filename)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Visualization file not found")
        
        return FileResponse(str(file_path), media_type="text/html")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# System operations

@router.get("/system/overview", response_model=Dict[str, Any])
async def get_system_overview(
    query_engine: QueryEngine = Depends(get_query_engine)
):
    """Get comprehensive system overview"""
    try:
        result = await query_engine.get_system_overview()
        
        return {
            "overview": result.metadata,
            "query_time_ms": result.query_time_ms,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/stats", response_model=Dict[str, Any])
async def get_system_statistics(
    neo4j_manager: Neo4jManager = Depends(get_neo4j_manager)
):
    """Get comprehensive system statistics"""
    try:
        graph_stats = await neo4j_manager.get_graph_statistics()
        
        # Get real-time updater stats if available
        updater = get_real_time_updater()
        updater_stats = updater.get_stats() if updater else {}
        
        # Get reasoning engine stats if available
        reasoning_engine = get_reasoning_engine()
        reasoning_stats = reasoning_engine.get_reasoning_statistics() if reasoning_engine else {}
        
        return {
            "graph_statistics": graph_stats,
            "real_time_updater": updater_stats,
            "reasoning_engine": reasoning_stats,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/rebuild", response_model=Dict[str, Any])
async def rebuild_knowledge_graph(
    background_tasks: BackgroundTasks,
    neo4j_manager: Neo4jManager = Depends(get_neo4j_manager)
):
    """Rebuild the knowledge graph from scratch"""
    try:
        # This is a potentially expensive operation, so run it in the background
        def rebuild_task():
            try:
                builder = KnowledgeGraphBuilder("/opt/sutazaiapp/backend", neo4j_manager)
                # Run the async operation in a new event loop
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(builder.build_knowledge_graph())
                loop.close()
                logger.info(f"Knowledge graph rebuild completed: {result}")
            except Exception as e:
                logger.error(f"Error rebuilding knowledge graph: {e}")
        
        background_tasks.add_task(rebuild_task)
        
        return {
            "message": "Knowledge graph rebuild started in background",
            "started_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting knowledge graph rebuild: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/sync", response_model=Dict[str, Any])
async def force_sync(
    updater: RealTimeUpdater = Depends(get_real_time_updater)
):
    """Force a full system synchronization"""
    try:
        if not updater:
            raise HTTPException(status_code=503, detail="Real-time updater not initialized")
        
        result = await updater.force_full_sync()
        return {
            "sync_result": result,
            "synced_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error forcing sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check

@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint"""
    try:
        neo4j_manager = get_neo4j_manager()
        neo4j_healthy = neo4j_manager is not None
        
        updater = get_real_time_updater()
        updater_healthy = updater is not None and updater.is_running
        
        reasoning_engine = get_reasoning_engine()
        reasoning_healthy = reasoning_engine is not None
        
        overall_healthy = neo4j_healthy and updater_healthy and reasoning_healthy
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "components": {
                "neo4j_manager": "healthy" if neo4j_healthy else "unhealthy",
                "real_time_updater": "healthy" if updater_healthy else "unhealthy", 
                "reasoning_engine": "healthy" if reasoning_healthy else "unhealthy"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }