#!/usr/bin/env python3
"""
SutazAI Knowledge Graph API - REST endpoints for knowledge graph operations
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, Depends, Query, Body
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("FastAPI not available. Install with: pip install fastapi uvicorn")
    exit(1)

from knowledge_graph_builder import (
    KnowledgeGraphBuilder, 
    Entity, 
    Relationship, 
    ComplianceCheck,
    EntityType,
    RelationType,
    EnforcementLevel
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class EntityRequest(BaseModel):
    """Request model for creating entities"""
    type: str
    name: str
    description: str
    properties: Dict[str, Any] = {}

class RelationshipRequest(BaseModel):
    """Request model for creating relationships"""
    source_id: str
    target_id: str
    type: str
    confidence: float = 1.0
    properties: Dict[str, Any] = {}

class QueryRequest(BaseModel):
    """Request model for graph queries"""
    query: str
    query_type: str = "cypher"
    parameters: Dict[str, Any] = {}

class ComplianceRequest(BaseModel):
    """Request model for compliance validation"""
    entity_id: Optional[str] = None
    standard_id: Optional[str] = None
    check_type: Optional[str] = None

class BuildRequest(BaseModel):
    """Request model for building knowledge graph"""
    sources: Dict[str, Any]
    comprehensive: bool = False
    force_rebuild: bool = False

class VisualizationRequest(BaseModel):
    """Request model for visualization data"""
    layout: str = "force"
    max_nodes: int = 1000
    filter_by: Dict[str, Any] = {}
    color_by: str = "type"
    size_by: str = "degree"

class EntityResponse(BaseModel):
    """Response model for entity data"""
    id: str
    type: str
    name: str
    description: str
    properties: Dict[str, Any]
    relationships: List[Dict[str, Any]] = []

class ComplianceResponse(BaseModel):
    """Response model for compliance checks"""
    entity_id: str
    standard_id: str
    status: str
    message: str
    severity: str
    timestamp: str
    evidence: List[str]

class GraphStats(BaseModel):
    """Response model for graph statistics"""
    total_nodes: int
    total_edges: int
    entity_counts: Dict[str, int]
    relationship_counts: Dict[str, int]
    compliance_summary: Dict[str, Any]

# FastAPI app instance
app = FastAPI(
    title="SutazAI Knowledge Graph API",
    description="REST API for SutazAI codebase hygiene standards knowledge graph",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global knowledge graph builder instance
kg_builder: Optional[KnowledgeGraphBuilder] = None

def get_kg_builder() -> KnowledgeGraphBuilder:
    """Dependency to get knowledge graph builder instance"""
    global kg_builder
    if kg_builder is None:
        config_path = os.path.join(os.path.dirname(__file__), "knowledge_graph_config.yaml")
        kg_builder = KnowledgeGraphBuilder(config_path=config_path)
    return kg_builder

# API Routes

@app.on_event("startup")
async def startup_event():
    """Initialize the knowledge graph on startup"""
    logger.info("Starting SutazAI Knowledge Graph API")
    
    # Initialize the knowledge graph builder
    global kg_builder
    config_path = os.path.join(os.path.dirname(__file__), "knowledge_graph_config.yaml")
    kg_builder = KnowledgeGraphBuilder(config_path=config_path)
    
    # Build initial graph from default sources
    try:
        sources = {
            'claude_md': '/opt/sutazaiapp/CLAUDE.md',
            'docker_configs': [
                '/opt/sutazaiapp/docker-compose.complete-agents.yml',
                '/opt/sutazaiapp/docker-compose.agents-simple.yml'
            ]
        }
        
        # Check if sources exist
        existing_sources = {}
        if os.path.exists(sources['claude_md']):
            existing_sources['claude_md'] = sources['claude_md']
        
        existing_docker_configs = [f for f in sources['docker_configs'] if os.path.exists(f)]
        if existing_docker_configs:
            existing_sources['docker_configs'] = existing_docker_configs
        
        if existing_sources:
            await kg_builder.build_comprehensive_graph(existing_sources)
            logger.info("Initial knowledge graph built successfully")
        else:
            logger.warning("No source files found, starting with empty graph")
            
    except Exception as e:
        logger.error(f"Error building initial graph: {e}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        builder = get_kg_builder()
        stats = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "graph_nodes": builder.graph.number_of_nodes(),
            "graph_edges": builder.graph.number_of_edges(),
            "entities": len(builder.entities),
            "neo4j_connected": builder.neo4j_driver is not None
        }
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats", response_model=GraphStats)
async def get_graph_stats(builder: KnowledgeGraphBuilder = Depends(get_kg_builder)):
    """Get comprehensive graph statistics"""
    try:
        # Entity counts by type
        entity_counts = {}
        for entity in builder.entities.values():
            entity_type = entity.type.value
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        # Relationship counts by type
        relationship_counts = {}
        for rel in builder.relationships:
            rel_type = rel.type.value
            relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1
        
        # Compliance summary
        compliance_summary = {
            "total_agents": len([e for e in builder.entities.values() if e.type == EntityType.AGENT]),
            "total_standards": len([e for e in builder.entities.values() if e.type == EntityType.STANDARD]),
            "blocking_violations": 0,  # This would be calculated from actual compliance checks
            "warnings": 0,
            "compliant_rate": 0.85  # This would be calculated from actual compliance data
        }
        
        return GraphStats(
            total_nodes=builder.graph.number_of_nodes(),
            total_edges=builder.graph.number_of_edges(),
            entity_counts=entity_counts,
            relationship_counts=relationship_counts,
            compliance_summary=compliance_summary
        )
    except Exception as e:
        logger.error(f"Error getting graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/build")
async def build_graph(
    request: BuildRequest,
    builder: KnowledgeGraphBuilder = Depends(get_kg_builder)
):
    """Build or rebuild the knowledge graph from specified sources"""
    try:
        logger.info(f"Building graph from sources: {request.sources}")
        
        if request.force_rebuild:
            # Clear existing graph
            builder.graph.clear()
            builder.entities.clear()
            builder.relationships.clear()
        
        # Build graph
        graph = await builder.build_comprehensive_graph(request.sources)
        
        return {
            "status": "success",
            "message": f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges",
            "timestamp": datetime.now().isoformat(),
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges()
        }
    except Exception as e:
        logger.error(f"Error building graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/entities")
async def list_entities(
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(100, description="Maximum number of entities to return"),
    offset: int = Query(0, description="Number of entities to skip"),
    builder: KnowledgeGraphBuilder = Depends(get_kg_builder)
):
    """List entities with optional filtering"""
    try:
        entities = []
        for entity in builder.entities.values():
            # Apply filters
            if entity_type and entity.type.value != entity_type:
                continue
            if category and entity.properties.get('category') != category:
                continue
            
            entities.append({
                "id": entity.id,
                "type": entity.type.value,
                "name": entity.name,
                "description": entity.description[:200] + "..." if len(entity.description) > 200 else entity.description,
                "category": entity.properties.get('category'),
                "enforcement_level": entity.properties.get('enforcement_level')
            })
        
        # Apply pagination
        total = len(entities)
        entities = entities[offset:offset + limit]
        
        return {
            "entities": entities,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Error listing entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/entities/{entity_id}", response_model=EntityResponse)
async def get_entity(
    entity_id: str,
    builder: KnowledgeGraphBuilder = Depends(get_kg_builder)
):
    """Get detailed information about a specific entity"""
    try:
        if entity_id not in builder.entities:
            raise HTTPException(status_code=404, detail="Entity not found")
        
        entity = builder.entities[entity_id]
        
        # Get relationships for this entity
        relationships = []
        for rel in builder.relationships:
            if rel.source_id == entity_id:
                target_entity = builder.entities.get(rel.target_id)
                relationships.append({
                    "type": rel.type.value,
                    "direction": "outgoing",
                    "target_id": rel.target_id,
                    "target_name": target_entity.name if target_entity else "Unknown",
                    "confidence": rel.confidence,
                    "properties": rel.properties
                })
            elif rel.target_id == entity_id:
                source_entity = builder.entities.get(rel.source_id)
                relationships.append({
                    "type": rel.type.value,
                    "direction": "incoming",
                    "source_id": rel.source_id,
                    "source_name": source_entity.name if source_entity else "Unknown",
                    "confidence": rel.confidence,
                    "properties": rel.properties
                })
        
        return EntityResponse(
            id=entity.id,
            type=entity.type.value,
            name=entity.name,
            description=entity.description,
            properties=entity.properties,
            relationships=relationships
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting entity {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/entities")
async def create_entity(
    request: EntityRequest,
    builder: KnowledgeGraphBuilder = Depends(get_kg_builder)
):
    """Create a new entity in the knowledge graph"""
    try:
        # Validate entity type
        try:
            entity_type = EntityType(request.type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid entity type: {request.type}")
        
        # Create entity
        entity = Entity(
            id="",  # Will be auto-generated
            type=entity_type,
            name=request.name,
            description=request.description,
            properties=request.properties
        )
        
        # Add to builder
        builder.entities[entity.id] = entity
        builder.graph.add_node(
            entity.id,
            type=entity.type.value,
            name=entity.name,
            description=entity.description,
            **entity.properties
        )
        
        # Store in Neo4j if available
        if builder.neo4j_driver:
            with builder.neo4j_driver.session() as session:
                session.run(
                    f"CREATE (e:{entity.type.value} {{id: $id, name: $name, description: $description}}) "
                    "SET e += $properties",
                    id=entity.id,
                    name=entity.name,
                    description=entity.description,
                    properties=entity.properties
                )
        
        return {
            "status": "success",
            "entity_id": entity.id,
            "message": "Entity created successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/relationships")
async def create_relationship(
    request: RelationshipRequest,
    builder: KnowledgeGraphBuilder = Depends(get_kg_builder)
):
    """Create a new relationship between entities"""
    try:
        # Validate entities exist
        if request.source_id not in builder.entities:
            raise HTTPException(status_code=404, detail=f"Source entity {request.source_id} not found")
        if request.target_id not in builder.entities:
            raise HTTPException(status_code=404, detail=f"Target entity {request.target_id} not found")
        
        # Validate relationship type
        try:
            rel_type = RelationType(request.type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid relationship type: {request.type}")
        
        # Create relationship
        relationship = Relationship(
            source_id=request.source_id,
            target_id=request.target_id,
            type=rel_type,
            confidence=request.confidence,
            properties=request.properties
        )
        
        # Add to builder
        builder.relationships.append(relationship)
        builder.graph.add_edge(
            request.source_id,
            request.target_id,
            type=rel_type.value,
            confidence=request.confidence,
            **request.properties
        )
        
        # Store in Neo4j if available
        if builder.neo4j_driver:
            with builder.neo4j_driver.session() as session:
                session.run(
                    f"MATCH (a {{id: $source_id}}), (b {{id: $target_id}}) "
                    f"CREATE (a)-[r:{rel_type.value.upper()} {{confidence: $confidence}}]->(b) "
                    "SET r += $properties",
                    source_id=request.source_id,
                    target_id=request.target_id,
                    confidence=request.confidence,
                    properties=request.properties
                )
        
        return {
            "status": "success",
            "message": "Relationship created successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def execute_query(
    request: QueryRequest,
    builder: KnowledgeGraphBuilder = Depends(get_kg_builder)
):
    """Execute a query against the knowledge graph"""
    try:
        results = builder.query_graph(request.query, request.query_type)
        
        return {
            "status": "success",
            "query": request.query,
            "query_type": request.query_type,
            "results": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/validate", response_model=List[ComplianceResponse])
async def validate_compliance(
    request: ComplianceRequest,
    builder: KnowledgeGraphBuilder = Depends(get_kg_builder)
):
    """Validate compliance for entities or standards"""
    try:
        if request.entity_id:
            # Validate specific entity
            checks = await builder.validate_compliance(request.entity_id)
        else:
            # Validate all entities (or filtered by standard/check_type)
            checks = []
            for entity_id in builder.entities.keys():
                entity_checks = await builder.validate_compliance(entity_id)
                if request.standard_id:
                    entity_checks = [c for c in entity_checks if c.standard_id == request.standard_id]
                checks.extend(entity_checks)
        
        # Convert to response format
        response_checks = []
        for check in checks:
            response_checks.append(ComplianceResponse(
                entity_id=check.entity_id,
                standard_id=check.standard_id,
                status=check.status,
                message=check.message,
                severity=check.severity.value,
                timestamp=check.timestamp.isoformat(),
                evidence=check.evidence
            ))
        
        return response_checks
    except Exception as e:
        logger.error(f"Error validating compliance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/report")
async def generate_report(
    format: str = Query("json", description="Report format"),
    builder: KnowledgeGraphBuilder = Depends(get_kg_builder)
):
    """Generate comprehensive compliance report"""
    try:
        report = builder.generate_compliance_report(format=format)
        
        if format == "json":
            return JSONResponse(content=report)
        else:
            # For other formats, you might want to generate files and return them
            return {"error": f"Format {format} not yet implemented"}
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/visualize")
async def get_visualization_data(
    request: VisualizationRequest,
    builder: KnowledgeGraphBuilder = Depends(get_kg_builder)
):
    """Get data formatted for visualization"""
    try:
        viz_data = builder.get_visualization_data(layout=request.layout)
        
        # Apply filters
        if request.filter_by:
            filtered_nodes = []
            for node in viz_data["nodes"]:
                include = True
                for key, value in request.filter_by.items():
                    if node.get(key) != value:
                        include = False
                        break
                if include:
                    filtered_nodes.append(node)
            
            # Filter edges to only include those between filtered nodes
            filtered_node_ids = {node["id"] for node in filtered_nodes}
            filtered_edges = [
                edge for edge in viz_data["edges"] 
                if edge["source"] in filtered_node_ids and edge["target"] in filtered_node_ids
            ]
            
            viz_data["nodes"] = filtered_nodes[:request.max_nodes]
            viz_data["edges"] = filtered_edges
        else:
            viz_data["nodes"] = viz_data["nodes"][:request.max_nodes]
        
        return viz_data
    except Exception as e:
        logger.error(f"Error getting visualization data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export")
async def export_graph(
    format: str = Query("json", description="Export format"),
    filename: Optional[str] = Query(None, description="Output filename"),
    builder: KnowledgeGraphBuilder = Depends(get_kg_builder)
):
    """Export knowledge graph in various formats"""
    try:
        if not filename:
            filename = f"sutazai_knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        
        # Create exports directory if it doesn't exist
        export_dir = Path("/tmp/kg_exports")
        export_dir.mkdir(exist_ok=True)
        
        output_path = export_dir / filename
        
        # Export the graph
        builder.export_graph(format, str(output_path))
        
        # Return the file
        return FileResponse(
            path=str(output_path),
            filename=filename,
            media_type="application/octet-stream"
        )
    except Exception as e:
        logger.error(f"Error exporting graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search")
async def search_entities(
    q: str = Query(..., description="Search query"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    limit: int = Query(10, description="Maximum results"),
    builder: KnowledgeGraphBuilder = Depends(get_kg_builder)
):
    """Search entities by name or description"""
    try:
        query_lower = q.lower()
        results = []
        
        for entity in builder.entities.values():
            # Simple text search - in practice, you'd use more sophisticated search
            score = 0
            if query_lower in entity.name.lower():
                score += 10
            if query_lower in entity.description.lower():
                score += 5
            
            # Check properties
            for prop_value in entity.properties.values():
                if isinstance(prop_value, str) and query_lower in prop_value.lower():
                    score += 2
            
            if score > 0:
                if entity_type is None or entity.type.value == entity_type:
                    results.append({
                        "score": score,
                        "entity": {
                            "id": entity.id,
                            "type": entity.type.value,
                            "name": entity.name,
                            "description": entity.description[:200] + "..." if len(entity.description) > 200 else entity.description,
                            "category": entity.properties.get('category')
                        }
                    })
        
        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:limit]
        
        return {
            "query": q,
            "results": [r["entity"] for r in results],
            "total_found": len(results)
        }
    except Exception as e:
        logger.error(f"Error searching entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Resource not found", "detail": str(exc)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

# Main entry point
def run_api(host: str = "0.0.0.0", port: int = 8048, debug: bool = False):
    """Run the FastAPI application"""
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug"
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SutazAI Knowledge Graph API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8048, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    run_api(host=args.host, port=args.port, debug=args.debug)