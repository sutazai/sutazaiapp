"""
Knowledge Graph Visualization
============================

Provides web-based visualization interfaces for the SutazAI knowledge graph.
Creates interactive, explorative visualizations for system understanding,
agent discovery, and architectural insights.

Features:
- Interactive graph visualization with D3.js
- Multiple layout algorithms (force-directed, hierarchical, circular)
- Node filtering and clustering
- Real-time graph updates
- Export capabilities (PNG, SVG, JSON)
- Responsive design for different screen sizes
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
from dataclasses import asdict

from .query_engine import QueryEngine, QueryResult
from .schema import NodeType, RelationshipType


class GraphVisualizationData:
    """Prepares data for graph visualization"""
    
    def __init__(self, query_engine: QueryEngine):
        self.query_engine = query_engine
        self.logger = logging.getLogger("graph_viz_data")
    
    async def prepare_agent_capability_graph(self, include_connections: bool = True) -> Dict[str, Any]:
        """Prepare data for agent-capability visualization"""
        
        # Get all agents and their capabilities
        cypher = """
        MATCH (a:Agent)-[:HAS_CAPABILITY]->(c:Capability)
        RETURN a, c, collect(c.name) as capabilities
        """
        
        results = await self.query_engine.neo4j_manager.execute_cypher(cypher)
        
        nodes = []
        links = []
        node_ids = set()
        
        # Process agents and capabilities
        for record in results:
            agent = record.get("a", {})
            capability = record.get("c", {})
            
            # Add agent node
            if agent.get("id") not in node_ids:
                agent_node = {
                    "id": agent.get("id"),
                    "name": agent.get("name", "Unknown Agent"),
                    "type": "agent",
                    "group": agent.get("agent_type", "generic"),
                    "health": agent.get("health_status", "unknown"),
                    "size": 20,
                    "metadata": {
                        "description": agent.get("description", ""),
                        "capabilities": record.get("capabilities", []),
                        "status": agent.get("status", "unknown")
                    }
                }
                nodes.append(agent_node)
                node_ids.add(agent.get("id"))
            
            # Add capability node
            if capability.get("id") not in node_ids:
                cap_node = {
                    "id": capability.get("id"),
                    "name": capability.get("name", "Unknown Capability"),
                    "type": "capability",
                    "group": capability.get("capability_type", "generic"),
                    "size": 15,
                    "metadata": {
                        "description": capability.get("description", ""),
                        "complexity": capability.get("complexity_level", 1)
                    }
                }
                nodes.append(cap_node)
                node_ids.add(capability.get("id"))
            
            # Add relationship
            if include_connections:
                link = {
                    "source": agent.get("id"),
                    "target": capability.get("id"),
                    "type": "has_capability",
                    "strength": 1.0
                }
                links.append(link)
        
        return {
            "nodes": nodes,
            "links": links,
            "metadata": {
                "total_agents": len([n for n in nodes if n["type"] == "agent"]),
                "total_capabilities": len([n for n in nodes if n["type"] == "capability"]),
                "total_connections": len(links),
                "generated_at": datetime.utcnow().isoformat()
            }
        }
    
    async def prepare_service_dependency_graph(self, max_depth: int = 3) -> Dict[str, Any]:
        """Prepare data for service dependency visualization"""
        
        cypher = f"""
        MATCH (s:Service)
        OPTIONAL MATCH path = (s)-[:DEPENDS_ON*1..{max_depth}]->(dep:Service)
        RETURN s, dep, path,
               CASE WHEN dep IS NOT NULL THEN length(path) ELSE 0 END as depth
        """
        
        results = await self.query_engine.neo4j_manager.execute_cypher(cypher)
        
        nodes = []
        links = []
        node_ids = set()
        
        for record in results:
            service = record.get("s", {})
            dependency = record.get("dep")
            depth = record.get("depth", 0)
            
            # Add source service node
            if service.get("id") not in node_ids:
                service_node = {
                    "id": service.get("id"),
                    "name": service.get("name", "Unknown Service"),
                    "type": "service",
                    "group": service.get("service_type", "generic"),
                    "size": 25,
                    "metadata": {
                        "description": service.get("description", ""),
                        "endpoints": service.get("endpoints", []),
                        "status": service.get("status", "unknown"),
                        "port": service.get("port")
                    }
                }
                nodes.append(service_node)
                node_ids.add(service.get("id"))
            
            # Add dependency node if exists
            if dependency and dependency.get("id") not in node_ids:
                dep_node = {
                    "id": dependency.get("id"),
                    "name": dependency.get("name", "Unknown Service"),
                    "type": "service",
                    "group": dependency.get("service_type", "generic"),
                    "size": 25,
                    "metadata": {
                        "description": dependency.get("description", ""),
                        "endpoints": dependency.get("endpoints", []),
                        "status": dependency.get("status", "unknown"),
                        "port": dependency.get("port")
                    }
                }
                nodes.append(dep_node)
                node_ids.add(dependency.get("id"))
                
                # Add dependency link
                link = {
                    "source": service.get("id"),
                    "target": dependency.get("id"),
                    "type": "depends_on",
                    "strength": max(0.1, 1.0 - (depth * 0.2)),  # Weaker for deeper dependencies
                    "depth": depth
                }
                links.append(link)
        
        return {
            "nodes": nodes,
            "links": links,
            "metadata": {
                "total_services": len(nodes),
                "total_dependencies": len(links),
                "max_dependency_depth": max_depth,
                "generated_at": datetime.utcnow().isoformat()
            }
        }
    
    async def prepare_system_architecture_graph(self) -> Dict[str, Any]:
        """Prepare comprehensive system architecture visualization"""
        
        cypher = """
        MATCH (n)-[r]-(m)
        WHERE labels(n)[0] IN ['Agent', 'Service', 'Database', 'Workflow']
        AND labels(m)[0] IN ['Agent', 'Service', 'Database', 'Workflow']
        RETURN n, r, m, labels(n)[0] as n_type, labels(m)[0] as m_type, type(r) as rel_type
        """
        
        results = await self.query_engine.neo4j_manager.execute_cypher(cypher)
        
        nodes = []
        links = []
        node_ids = set()
        
        # Color schemes for different node types
        type_colors = {
            "agent": "#4CAF50",
            "service": "#2196F3", 
            "database": "#FF9800",
            "workflow": "#9C27B0",
            "capability": "#607D8B"
        }
        
        # Size mapping for different node types
        type_sizes = {
            "agent": 20,
            "service": 25,
            "database": 30,
            "workflow": 22,
            "capability": 15
        }
        
        for record in results:
            source_node = record.get("n", {})
            target_node = record.get("m", {})
            relationship = record.get("r", {})
            source_type = record.get("n_type", "").lower()
            target_type = record.get("m_type", "").lower()
            rel_type = record.get("rel_type", "")
            
            # Add source node
            if source_node.get("id") not in node_ids:
                node = {
                    "id": source_node.get("id"),
                    "name": source_node.get("name", "Unknown"),
                    "type": source_type,
                    "group": source_node.get(f"{source_type}_type", source_type),
                    "size": type_sizes.get(source_type, 20),
                    "color": type_colors.get(source_type, "#666666"),
                    "metadata": self._extract_node_metadata(source_node, source_type)
                }
                nodes.append(node)
                node_ids.add(source_node.get("id"))
            
            # Add target node
            if target_node.get("id") not in node_ids:
                node = {
                    "id": target_node.get("id"),
                    "name": target_node.get("name", "Unknown"),
                    "type": target_type,
                    "group": target_node.get(f"{target_type}_type", target_type),
                    "size": type_sizes.get(target_type, 20),
                    "color": type_colors.get(target_type, "#666666"),
                    "metadata": self._extract_node_metadata(target_node, target_type)
                }
                nodes.append(node)
                node_ids.add(target_node.get("id"))
            
            # Add relationship
            link = {
                "source": source_node.get("id"),
                "target": target_node.get("id"),
                "type": rel_type.lower(),
                "strength": self._calculate_link_strength(rel_type),
                "metadata": {
                    "relationship_id": relationship.get("id"),
                    "created_at": relationship.get("created_at"),
                    "weight": relationship.get("weight", 1.0)
                }
            }
            links.append(link)
        
        return {
            "nodes": nodes,
            "links": links,
            "metadata": {
                "node_counts": self._count_nodes_by_type(nodes),
                "relationship_counts": self._count_relationships_by_type(links),
                "color_scheme": type_colors,
                "generated_at": datetime.utcnow().isoformat()
            }
        }
    
    async def prepare_data_flow_graph(self, source_id: Optional[str] = None) -> Dict[str, Any]:
        """Prepare data flow visualization"""
        
        if source_id:
            cypher = """
            MATCH path = (source {id: $source_id})-[:WRITES_TO|READS_FROM*1..5]-(target)
            RETURN path, nodes(path) as path_nodes, relationships(path) as path_rels
            """
            parameters = {"source_id": source_id}
        else:
            cypher = """
            MATCH (s)-[r:WRITES_TO|READS_FROM]->(t)
            RETURN s, r, t
            """
            parameters = {}
        
        results = await self.query_engine.neo4j_manager.execute_cypher(cypher, parameters)
        
        nodes = []
        links = []
        node_ids = set()
        
        if source_id:
            # Process path results
            for record in results:
                path_nodes = record.get("path_nodes", [])
                path_rels = record.get("path_rels", [])
                
                # Add nodes from path
                for node in path_nodes:
                    if node and node.get("id") not in node_ids:
                        viz_node = self._create_data_flow_node(node)
                        nodes.append(viz_node)
                        node_ids.add(node.get("id"))
                
                # Add relationships from path
                for i, rel in enumerate(path_rels):
                    if i < len(path_nodes) - 1:
                        link = {
                            "source": path_nodes[i].get("id"),
                            "target": path_nodes[i + 1].get("id"),
                            "type": rel.get("type", "").lower(),
                            "strength": 0.8,
                            "flow_direction": "forward" if "writes_to" in rel.get("type", "").lower() else "bidirectional"
                        }
                        links.append(link)
        else:
            # Process direct relationships
            for record in results:
                source = record.get("s", {})
                target = record.get("t", {})
                relationship = record.get("r", {})
                
                # Add source node
                if source.get("id") not in node_ids:
                    nodes.append(self._create_data_flow_node(source))
                    node_ids.add(source.get("id"))
                
                # Add target node
                if target.get("id") not in node_ids:
                    nodes.append(self._create_data_flow_node(target))
                    node_ids.add(target.get("id"))
                
                # Add relationship
                rel_type = relationship.get("type", "").lower()
                link = {
                    "source": source.get("id"),
                    "target": target.get("id"),
                    "type": rel_type,
                    "strength": 0.8,
                    "flow_direction": "forward" if "writes_to" in rel_type else "bidirectional"
                }
                links.append(link)
        
        return {
            "nodes": nodes,
            "links": links,
            "metadata": {
                "source_node": source_id,
                "data_flow_type": "trace" if source_id else "overview",
                "generated_at": datetime.utcnow().isoformat()
            }
        }
    
    def _extract_node_metadata(self, node: Dict[str, Any], node_type: str) -> Dict[str, Any]:
        """Extract relevant metadata for visualization"""
        base_metadata = {
            "description": node.get("description", ""),
            "status": node.get("status", "unknown"),
            "created_at": node.get("created_at"),
            "updated_at": node.get("updated_at")
        }
        
        if node_type == "agent":
            base_metadata.update({
                "agent_type": node.get("agent_type", ""),
                "health_status": node.get("health_status", "unknown"),
                "capabilities": node.get("capabilities", [])
            })
        elif node_type == "service":
            base_metadata.update({
                "service_type": node.get("service_type", ""),
                "port": node.get("port"),
                "endpoints": node.get("endpoints", [])
            })
        elif node_type == "database":
            base_metadata.update({
                "database_type": node.get("database_type", ""),
                "tables_collections": node.get("tables_collections", [])
            })
        
        return base_metadata
    
    def _calculate_link_strength(self, relationship_type: str) -> float:
        """Calculate link strength based on relationship type"""
        strength_mapping = {
            "depends_on": 0.9,
            "has_capability": 0.7,
            "communicates_with": 0.6,
            "uses": 0.6,
            "orchestrates": 0.8,
            "writes_to": 0.8,
            "reads_from": 0.7,
            "triggers": 0.9,
            "monitors": 0.5
        }
        
        return strength_mapping.get(relationship_type.lower(), 0.5)
    
    def _count_nodes_by_type(self, nodes: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count nodes by type"""
        counts = {}
        for node in nodes:
            node_type = node.get("type", "unknown")
            counts[node_type] = counts.get(node_type, 0) + 1
        return counts
    
    def _count_relationships_by_type(self, links: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count relationships by type"""
        counts = {}
        for link in links:
            rel_type = link.get("type", "unknown")
            counts[rel_type] = counts.get(rel_type, 0) + 1
        return counts
    
    def _create_data_flow_node(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Create a node for data flow visualization"""
        node_labels = node.get("labels", ["Unknown"])
        node_type = node_labels[0].lower() if node_labels else "unknown"
        
        # Determine node characteristics based on type
        if node_type == "database":
            color = "#FF9800"
            size = 30
            shape = "square"
        elif node_type == "service":
            color = "#2196F3"
            size = 25
            shape = "circle"
        elif node_type == "agent":
            color = "#4CAF50"
            size = 20
            shape = "triangle"
        else:
            color = "#666666"
            size = 20
            shape = "circle"
        
        return {
            "id": node.get("id"),
            "name": node.get("name", "Unknown"),
            "type": node_type,
            "color": color,
            "size": size,
            "shape": shape,
            "metadata": self._extract_node_metadata(node, node_type)
        }


class HTMLGenerator:
    """Generates HTML visualization interfaces"""
    
    def __init__(self):
        self.logger = logging.getLogger("html_generator")
    
    def generate_graph_viewer_html(self, graph_data: Dict[str, Any], 
                                 title: str = "SutazAI Knowledge Graph",
                                 layout: str = "force") -> str:
        """Generate HTML for interactive graph visualization"""
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a1a;
            color: #ffffff;
            overflow: hidden;
        }}
        
        .header {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 60px;
            background: #2d2d2d;
            border-bottom: 2px solid #4CAF50;
            display: flex;
            align-items: center;
            padding: 0 20px;
            z-index: 1000;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 24px;
            color: #4CAF50;
        }}
        
        .controls {{
            position: fixed;
            top: 80px;
            left: 20px;
            background: rgba(45, 45, 45, 0.9);
            padding: 15px;
            border-radius: 8px;
            z-index: 1000;
            backdrop-filter: blur(10px);
            border: 1px solid #444;
        }}
        
        .control-group {{
            margin-bottom: 15px;
        }}
        
        .control-group label {{
            display: block;
            margin-bottom: 5px;
            font-size: 12px;
            color: #ccc;
        }}
        
        .control-group select, .control-group input, .control-group button {{
            width: 100%;
            padding: 8px;
            border: 1px solid #555;
            background: #333;
            color: #fff;
            border-radius: 4px;
            font-size: 12px;
        }}
        
        .control-group button {{
            background: #4CAF50;
            cursor: pointer;
            transition: background 0.3s;
        }}
        
        .control-group button:hover {{
            background: #45a049;
        }}
        
        .legend {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(45, 45, 45, 0.9);
            padding: 15px;
            border-radius: 8px;
            backdrop-filter: blur(10px);
            border: 1px solid #444;
            z-index: 1000;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            font-size: 12px;
        }}
        
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        
        .info-panel {{
            position: fixed;
            top: 80px;
            right: 20px;
            width: 300px;
            background: rgba(45, 45, 45, 0.9);
            padding: 15px;
            border-radius: 8px;
            backdrop-filter: blur(10px);
            border: 1px solid #444;
            z-index: 1000;
            max-height: calc(100vh - 120px);
            overflow-y: auto;
            display: none;
        }}
        
        .graph-container {{
            position: fixed;
            top: 60px;
            left: 0;
            right: 0;
            bottom: 0;
            background: #1a1a1a;
        }}
        
        .node {{
            cursor: pointer;
            stroke: #fff;
            stroke-width: 2px;
        }}
        
        .node:hover {{
            stroke-width: 3px;
        }}
        
        .link {{
            stroke: #666;
            stroke-opacity: 0.6;
            fill: none;
        }}
        
        .link.highlighted {{
            stroke: #4CAF50;
            stroke-width: 3px;
            stroke-opacity: 1;
        }}
        
        .node-label {{
            font-size: 10px;
            fill: #fff;
            text-anchor: middle;
            pointer-events: none;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
        }}
        
        .zoom-controls {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            display: flex;
            flex-direction: column;
            z-index: 1000;
        }}
        
        .zoom-btn {{
            width: 40px;
            height: 40px;
            background: #4CAF50;
            border: none;
            color: white;
            font-size: 18px;
            cursor: pointer;
            margin-bottom: 5px;
            border-radius: 4px;
            transition: background 0.3s;
        }}
        
        .zoom-btn:hover {{
            background: #45a049;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <div style="margin-left: auto; color: #ccc; font-size: 14px;">
            Nodes: <span id="node-count">{len(graph_data.get('nodes', []))}</span> | 
            Links: <span id="link-count">{len(graph_data.get('links', []))}</span>
        </div>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <label>Layout</label>
            <select id="layout-select">
                <option value="force" {"selected" if layout == "force" else ""}>Force Directed</option>
                <option value="hierarchical" {"selected" if layout == "hierarchical" else ""}>Hierarchical</option>
                <option value="circular" {"selected" if layout == "circular" else ""}>Circular</option>
            </select>
        </div>
        
        <div class="control-group">
            <label>Filter by Type</label>
            <select id="type-filter">
                <option value="all">All Types</option>
                <option value="agent">Agents</option>
                <option value="service">Services</option>
                <option value="database">Databases</option>
                <option value="capability">Capabilities</option>
            </select>
        </div>
        
        <div class="control-group">
            <label>Search</label>
            <input type="text" id="search-input" placeholder="Search nodes...">
        </div>
        
        <div class="control-group">
            <button id="reset-view">Reset View</button>
        </div>
        
        <div class="control-group">
            <button id="export-svg">Export SVG</button>
        </div>
    </div>
    
    <div class="legend">
        <div class="legend-item">
            <div class="legend-color" style="background: #4CAF50;"></div>
            <span>Agents</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #2196F3;"></div>
            <span>Services</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #FF9800;"></div>
            <span>Databases</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #9C27B0;"></div>
            <span>Workflows</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #607D8B;"></div>
            <span>Capabilities</span>
        </div>
    </div>
    
    <div class="info-panel" id="info-panel">
        <h3 id="info-title">Node Information</h3>
        <div id="info-content"></div>
    </div>
    
    <div class="zoom-controls">
        <button class="zoom-btn" id="zoom-in">+</button>
        <button class="zoom-btn" id="zoom-out">−</button>
        <button class="zoom-btn" id="zoom-fit">⌂</button>
    </div>
    
    <div class="graph-container">
        <svg id="graph-svg"></svg>
    </div>
    
    <script>
        // Graph data
        const graphData = {json.dumps(graph_data, indent=2)};
        
        // Initialize visualization
        const GraphVisualization = {{
            init() {{
                this.setupSVG();
                this.setupForceSimulation();
                this.setupControls();
                this.render();
            }},
            
            setupSVG() {{
                const container = d3.select('.graph-container');
                const rect = container.node().getBoundingClientRect();
                
                this.width = rect.width;
                this.height = rect.height;
                
                this.svg = d3.select('#graph-svg')
                    .attr('width', this.width)
                    .attr('height', this.height);
                
                this.g = this.svg.append('g');
                
                // Setup zoom
                this.zoom = d3.zoom()
                    .scaleExtent([0.1, 4])
                    .on('zoom', (event) => {{
                        this.g.attr('transform', event.transform);
                    }});
                
                this.svg.call(this.zoom);
            }},
            
            setupForceSimulation() {{
                this.simulation = d3.forceSimulation()
                    .force('link', d3.forceLink().id(d => d.id).distance(d => {{
                        const baseDistance = 100;
                        return baseDistance * (2 - d.strength);
                    }}))
                    .force('charge', d3.forceManyBody().strength(-300))
                    .force('center', d3.forceCenter(this.width / 2, this.height / 2))
                    .force('collision', d3.forceCollide().radius(d => d.size + 5));
            }},
            
            setupControls() {{
                // Layout selector
                d3.select('#layout-select').on('change', () => {{
                    this.changeLayout(d3.event.target.value);
                }});
                
                // Type filter
                d3.select('#type-filter').on('change', () => {{
                    this.filterByType(d3.event.target.value);
                }});
                
                // Search
                d3.select('#search-input').on('input', () => {{
                    this.searchNodes(d3.event.target.value);
                }});
                
                // Reset view
                d3.select('#reset-view').on('click', () => {{
                    this.resetView();
                }});
                
                // Export SVG
                d3.select('#export-svg').on('click', () => {{
                    this.exportSVG();
                }});
                
                // Zoom controls
                d3.select('#zoom-in').on('click', () => {{
                    this.svg.transition().call(
                        this.zoom.scaleBy, 1.5
                    );
                }});
                
                d3.select('#zoom-out').on('click', () => {{
                    this.svg.transition().call(
                        this.zoom.scaleBy, 1 / 1.5
                    );
                }});
                
                d3.select('#zoom-fit').on('click', () => {{
                    this.fitToView();
                }});
            }},
            
            render() {{
                this.currentData = {{
                    nodes: [...graphData.nodes],
                    links: [...graphData.links]
                }};
                
                this.renderLinks();
                this.renderNodes();
                this.startSimulation();
            }},
            
            renderLinks() {{
                this.link = this.g.selectAll('.link')
                    .data(this.currentData.links, d => `${{d.source.id || d.source}}-${{d.target.id || d.target}}`);
                
                this.linkEnter = this.link.enter()
                    .append('line')
                    .attr('class', 'link')
                    .attr('stroke-width', d => Math.sqrt(d.strength * 5));
                
                this.link = this.linkEnter.merge(this.link);
                this.link.exit().remove();
            }},
            
            renderNodes() {{
                this.node = this.g.selectAll('.node')
                    .data(this.currentData.nodes, d => d.id);
                
                this.nodeEnter = this.node.enter()
                    .append('g')
                    .attr('class', 'node-group');
                
                this.nodeEnter.append('circle')
                    .attr('class', 'node')
                    .attr('r', d => d.size)
                    .attr('fill', d => d.color || this.getTypeColor(d.type))
                    .on('click', (event, d) => {{
                        this.showNodeInfo(d);
                        this.highlightConnections(d);
                    }})
                    .on('mouseover', (event, d) => {{
                        this.showTooltip(event, d);
                    }})
                    .on('mouseout', () => {{
                        this.hideTooltip();
                    }})
                    .call(d3.drag()
                        .on('start', (event, d) => {{
                            if (!event.active) this.simulation.alphaTarget(0.3).restart();
                            d.fx = d.x;
                            d.fy = d.y;
                        }})
                        .on('drag', (event, d) => {{
                            d.fx = event.x;
                            d.fy = event.y;
                        }})
                        .on('end', (event, d) => {{
                            if (!event.active) this.simulation.alphaTarget(0);
                            d.fx = null;
                            d.fy = null;
                        }})
                    );
                
                this.nodeEnter.append('text')
                    .attr('class', 'node-label')
                    .attr('dy', d => d.size + 15)
                    .text(d => d.name.length > 15 ? d.name.substring(0, 15) + '...' : d.name);
                
                this.node = this.nodeEnter.merge(this.node);
                this.node.exit().remove();
            }},
            
            startSimulation() {{
                this.simulation
                    .nodes(this.currentData.nodes)
                    .on('tick', () => {{
                        this.link
                            .attr('x1', d => d.source.x)
                            .attr('y1', d => d.source.y)
                            .attr('x2', d => d.target.x)
                            .attr('y2', d => d.target.y);
                        
                        this.node
                            .attr('transform', d => `translate(${{d.x}},${{d.y}})`);
                    }});
                
                this.simulation.force('link')
                    .links(this.currentData.links);
            }},
            
            getTypeColor(type) {{
                const colors = {{
                    'agent': '#4CAF50',
                    'service': '#2196F3',
                    'database': '#FF9800',
                    'workflow': '#9C27B0',
                    'capability': '#607D8B'
                }};
                return colors[type] || '#666666';
            }},
            
            showNodeInfo(node) {{
                const panel = d3.select('#info-panel');
                const title = d3.select('#info-title');
                const content = d3.select('#info-content');
                
                title.text(node.name);
                
                let html = `
                    <p><strong>Type:</strong> ${{node.type}}</p>
                    <p><strong>Group:</strong> ${{node.group || 'N/A'}}</p>
                `;
                
                if (node.metadata) {{
                    html += '<h4>Details:</h4>';
                    for (const [key, value] of Object.entries(node.metadata)) {{
                        if (value && value !== '') {{
                            html += `<p><strong>${{key}}:</strong> ${{Array.isArray(value) ? value.join(', ') : value}}</p>`;
                        }}
                    }}
                }}
                
                content.html(html);
                panel.style('display', 'block');
            }},
            
            highlightConnections(node) {{
                // Reset all links
                this.link.classed('highlighted', false);
                
                // Highlight connected links
                this.link.classed('highlighted', d => 
                    d.source.id === node.id || d.target.id === node.id
                );
            }},
            
            showTooltip(event, node) {{
                // Implementation for tooltip
            }},
            
            hideTooltip() {{
                // Implementation for hiding tooltip
            }},
            
            filterByType(type) {{
                if (type === 'all') {{
                    this.currentData = {{
                        nodes: [...graphData.nodes],
                        links: [...graphData.links]
                    }};
                }} else {{
                    this.currentData = {{
                        nodes: graphData.nodes.filter(n => n.type === type),
                        links: graphData.links.filter(l => {{
                            const sourceType = graphData.nodes.find(n => n.id === l.source.id || n.id === l.source)?.type;
                            const targetType = graphData.nodes.find(n => n.id === l.target.id || n.id === l.target)?.type;
                            return sourceType === type && targetType === type;
                        }})
                    }};
                }}
                
                this.render();
            }},
            
            searchNodes(query) {{
                if (!query) {{
                    this.node.selectAll('circle').attr('opacity', 1);
                    this.node.selectAll('text').attr('opacity', 1);
                    return;
                }}
                
                this.node.selectAll('circle').attr('opacity', d => 
                    d.name.toLowerCase().includes(query.toLowerCase()) ? 1 : 0.2
                );
                
                this.node.selectAll('text').attr('opacity', d => 
                    d.name.toLowerCase().includes(query.toLowerCase()) ? 1 : 0.2
                );
            }},
            
            resetView() {{
                this.svg.transition().duration(750).call(
                    this.zoom.transform,
                    d3.zoomIdentity
                );
                
                d3.select('#info-panel').style('display', 'none');
                this.link.classed('highlighted', false);
            }},
            
            fitToView() {{
                const bounds = this.g.node().getBBox();
                const fullWidth = this.width;
                const fullHeight = this.height;
                const width = bounds.width;
                const height = bounds.height;
                const midX = bounds.x + width / 2;
                const midY = bounds.y + height / 2;
                
                if (width == 0 || height == 0) return;
                
                const scale = 0.8 / Math.max(width / fullWidth, height / fullHeight);
                const translate = [fullWidth / 2 - scale * midX, fullHeight / 2 - scale * midY];
                
                this.svg.transition().duration(750).call(
                    this.zoom.transform,
                    d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
                );
            }},
            
            exportSVG() {{
                const svgData = new XMLSerializer().serializeToString(this.svg.node());
                const blob = new Blob([svgData], {{type: 'image/svg+xml;charset=utf-8'}});
                const url = URL.createObjectURL(blob);
                
                const link = document.createElement('a');
                link.href = url;
                link.download = 'knowledge-graph.svg';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                URL.revokeObjectURL(url);
            }}
        }};
        
        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', () => {{
            GraphVisualization.init();
        }});
        
        // Handle window resize
        window.addEventListener('resize', () => {{
            const container = d3.select('.graph-container');
            const rect = container.node().getBoundingClientRect();
            
            GraphVisualization.width = rect.width;
            GraphVisualization.height = rect.height;
            
            GraphVisualization.svg
                .attr('width', GraphVisualization.width)
                .attr('height', GraphVisualization.height);
            
            GraphVisualization.simulation
                .force('center', d3.forceCenter(GraphVisualization.width / 2, GraphVisualization.height / 2))
                .restart();
        }});
    </script>
</body>
</html>"""
        
        return html_template
    
    def save_visualization(self, html_content: str, filename: str = "knowledge_graph.html") -> str:
        """Save visualization HTML to file"""
        try:
            output_path = Path(filename)
            output_path.write_text(html_content, encoding='utf-8')
            self.logger.info(f"Visualization saved to {output_path.absolute()}")
            return str(output_path.absolute())
        
        except Exception as e:
            self.logger.error(f"Failed to save visualization: {e}")
            return ""


class VisualizationManager:
    """Main manager for knowledge graph visualizations"""
    
    def __init__(self, query_engine: QueryEngine):
        self.query_engine = query_engine
        self.data_preparer = GraphVisualizationData(query_engine)
        self.html_generator = HTMLGenerator()
        self.logger = logging.getLogger("viz_manager")
    
    async def create_agent_capability_view(self, output_file: str = "agent_capabilities.html") -> str:
        """Create agent capability visualization"""
        self.logger.info("Creating agent capability visualization")
        
        graph_data = await self.data_preparer.prepare_agent_capability_graph()
        html_content = self.html_generator.generate_graph_viewer_html(
            graph_data,
            title="SutazAI Agent Capabilities",
            layout="force"
        )
        
        return self.html_generator.save_visualization(html_content, output_file)
    
    async def create_service_dependency_view(self, output_file: str = "service_dependencies.html") -> str:
        """Create service dependency visualization"""
        self.logger.info("Creating service dependency visualization")
        
        graph_data = await self.data_preparer.prepare_service_dependency_graph()
        html_content = self.html_generator.generate_graph_viewer_html(
            graph_data,
            title="SutazAI Service Dependencies",
            layout="hierarchical"
        )
        
        return self.html_generator.save_visualization(html_content, output_file)
    
    async def create_system_architecture_view(self, output_file: str = "system_architecture.html") -> str:
        """Create comprehensive system architecture visualization"""
        self.logger.info("Creating system architecture visualization")
        
        graph_data = await self.data_preparer.prepare_system_architecture_graph()
        html_content = self.html_generator.generate_graph_viewer_html(
            graph_data,
            title="SutazAI System Architecture",
            layout="force"
        )
        
        return self.html_generator.save_visualization(html_content, output_file)
    
    async def create_data_flow_view(self, source_id: Optional[str] = None, 
                                  output_file: str = "data_flow.html") -> str:
        """Create data flow visualization"""
        self.logger.info(f"Creating data flow visualization{' for ' + source_id if source_id else ''}")
        
        graph_data = await self.data_preparer.prepare_data_flow_graph(source_id)
        title = f"Data Flow from {source_id}" if source_id else "System Data Flow"
        
        html_content = self.html_generator.generate_graph_viewer_html(
            graph_data,
            title=title,
            layout="hierarchical"
        )
        
        return self.html_generator.save_visualization(html_content, output_file)
    
    async def create_custom_view(self, cypher_query: str, parameters: Dict[str, Any] = None,
                               title: str = "Custom Graph View",
                               output_file: str = "custom_graph.html") -> str:
        """Create visualization from custom Cypher query"""
        self.logger.info("Creating custom graph visualization")
        
        # Execute custom query
        results = await self.query_engine.neo4j_manager.execute_cypher(cypher_query, parameters or {})
        
        # Process results into graph data format
        nodes = []
        links = []
        node_ids = set()
        
        for record in results:
            for key, value in record.items():
                if isinstance(value, dict) and "id" in value:
                    # This looks like a node
                    if value["id"] not in node_ids:
                        node = {
                            "id": value["id"],
                            "name": value.get("name", "Unknown"),
                            "type": value.get("type", "unknown"),
                            "size": 20,
                            "metadata": value
                        }
                        nodes.append(node)
                        node_ids.add(value["id"])
        
        graph_data = {
            "nodes": nodes,
            "links": links,
            "metadata": {
                "custom_query": True,
                "cypher": cypher_query,
                "parameters": parameters,
                "generated_at": datetime.utcnow().isoformat()
            }
        }
        
        html_content = self.html_generator.generate_graph_viewer_html(
            graph_data,
            title=title
        )
        
        return self.html_generator.save_visualization(html_content, output_file)