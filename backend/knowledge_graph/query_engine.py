"""
Knowledge Graph Query Engine
===========================

Provides intelligent query interfaces for the SutazAI knowledge graph.
Supports agent discovery, service dependency analysis, capability matching,
and system navigation with natural language and structured queries.

Features:
- Agent discovery by capabilities
- Service dependency analysis
- Data flow tracing
- System architecture exploration
- Natural language query processing
- Complex graph pattern matching
- Performance optimization
"""

import re
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime
from dataclasses import dataclass

from .neo4j_manager import Neo4jManager
from .schema import NodeType, RelationshipType, AgentCapability


@dataclass
class QueryResult:
    """Structure for query results"""
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    query_time_ms: float
    total_results: int


class QueryTemplate:
    """Pre-defined query templates for common operations"""
    
    # Agent discovery queries
    FIND_AGENTS_BY_CAPABILITY = """
    MATCH (a:Agent)-[:HAS_CAPABILITY]->(c:Capability)
    WHERE c.name IN $capabilities
    WITH a, collect(c.name) as agent_capabilities
    WHERE size(agent_capabilities) >= $min_capabilities
    RETURN a, agent_capabilities
    ORDER BY a.name
    """
    
    FIND_AGENTS_BY_TYPE = """
    MATCH (a:Agent)
    WHERE a.agent_type = $agent_type
    RETURN a
    ORDER BY a.name
    """
    
    FIND_AVAILABLE_AGENTS = """
    MATCH (a:Agent)
    WHERE a.health_status = 'healthy' 
    AND a.status IN ['idle', 'active']
    RETURN a
    ORDER BY a.performance_metrics.success_rate DESC
    """
    
    # Service dependency queries
    FIND_SERVICE_DEPENDENCIES = """
    MATCH (s:Service)-[:DEPENDS_ON*1..3]->(dep:Service)
    WHERE s.name = $service_name
    RETURN s, dep, length(path) as depth
    ORDER BY depth
    """
    
    FIND_SERVICE_DEPENDENTS = """
    MATCH (dep:Service)<-[:DEPENDS_ON*1..3]-(s:Service)
    WHERE dep.name = $service_name
    RETURN dep, s, length(path) as depth
    ORDER BY depth
    """
    
    SERVICE_IMPACT_ANALYSIS = """
    MATCH path = (s:Service)-[:DEPENDS_ON*1..5]->(affected:Service)
    WHERE s.name = $service_name
    WITH affected, length(path) as impact_distance
    ORDER BY impact_distance
    RETURN affected.name as service_name, 
           affected.service_type as type,
           impact_distance,
           affected.status as current_status
    """
    
    # Data flow queries
    TRACE_DATA_FLOW = """
    MATCH path = (source)-[:WRITES_TO|READS_FROM*1..5]->(target)
    WHERE source.name = $source_name
    RETURN path, 
           [node in nodes(path) | node.name] as flow_path,
           [rel in relationships(path) | type(rel)] as flow_types
    """
    
    DATABASE_CONNECTIONS = """
    MATCH (s:Service)-[r:READS_FROM|WRITES_TO]->(db:Database)
    RETURN s.name as service, 
           type(r) as operation,
           db.name as database,
           db.database_type as db_type
    ORDER BY s.name
    """
    
    # Architecture exploration
    SYSTEM_OVERVIEW = """
    MATCH (n)
    WITH labels(n)[0] as node_type, count(n) as count
    RETURN node_type, count
    ORDER BY count DESC
    """
    
    CAPABILITY_COVERAGE = """
    MATCH (c:Capability)<-[:HAS_CAPABILITY]-(a:Agent)
    WITH c, count(a) as agent_count
    RETURN c.name as capability,
           c.capability_type as category,
           agent_count,
           c.complexity_level as complexity
    ORDER BY agent_count DESC, complexity DESC
    """
    
    # Workflow queries
    FIND_WORKFLOWS_USING_AGENT = """
    MATCH (w:Workflow)-[:USES]->(a:Agent)
    WHERE a.name = $agent_name
    RETURN w, a
    """
    
    WORKFLOW_DEPENDENCIES = """
    MATCH (w:Workflow)-[:USES|TRIGGERS*1..3]->(dep)
    WHERE w.name = $workflow_name
    RETURN w, dep, labels(dep)[0] as dep_type
    """


class NaturalLanguageProcessor:
    """Process natural language queries and convert to Cypher"""
    
    def __init__(self):
        self.intent_patterns = {
            "find_agents": [
                r"find.*agents?.*with.*capabilities?",
                r"what.*agents?.*can.*do",
                r"show.*agents?.*for.*task",
                r"list.*agents?.*that.*have"
            ],
            "find_services": [
                r"find.*services?",
                r"what.*services?.*depend.*on",
                r"show.*service.*dependencies",
                r"list.*services?"
            ],
            "trace_data": [
                r"trace.*data.*flow",
                r"where.*data.*goes",
                r"data.*path.*from",
                r"follow.*data.*through"
            ],
            "system_overview": [
                r"system.*overview",
                r"architecture.*summary",
                r"what.*components.*exist",
                r"show.*system.*structure"
            ],
            "agent_capabilities": [
                r"what.*can.*agent.*do",
                r"agent.*capabilities",
                r"show.*agent.*skills"
            ]
        }
        
        self.entity_patterns = {
            "agent_name": r"agent[:\s]+(\w+)",
            "service_name": r"service[:\s]+(\w+)",
            "capability": r"capabilit[yi][:\s]+(\w+)",
            "database": r"database[:\s]+(\w+)"
        }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process natural language query and return structured query info"""
        query_lower = query.lower()
        
        # Determine intent
        intent = self._detect_intent(query_lower)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Generate Cypher query
        cypher_query, parameters = self._generate_cypher(intent, entities, query_lower)
        
        return {
            "intent": intent,
            "entities": entities,
            "cypher": cypher_query,
            "parameters": parameters,
            "original_query": query
        }
    
    def _detect_intent(self, query: str) -> str:
        """Detect the intent of the query"""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return intent
        
        return "unknown"
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from the query"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches
        
        # Extract capability names from common capability keywords
        capability_keywords = [
            "code", "security", "testing", "monitoring", "deployment",
            "analysis", "generation", "orchestration", "communication"
        ]
        
        found_capabilities = []
        query_lower = query.lower()
        for keyword in capability_keywords:
            if keyword in query_lower:
                found_capabilities.append(keyword)
        
        if found_capabilities:
            entities["capabilities"] = found_capabilities
        
        return entities
    
    def _generate_cypher(self, intent: str, entities: Dict[str, Any], query: str) -> Tuple[str, Dict[str, Any]]:
        """Generate Cypher query based on intent and entities"""
        parameters = {}
        
        if intent == "find_agents":
            if "capabilities" in entities:
                cypher = QueryTemplate.FIND_AGENTS_BY_CAPABILITY
                parameters = {
                    "capabilities": entities["capabilities"],
                    "min_capabilities": 1
                }
            else:
                cypher = QueryTemplate.FIND_AVAILABLE_AGENTS
        
        elif intent == "find_services":
            if "service_name" in entities:
                cypher = QueryTemplate.FIND_SERVICE_DEPENDENCIES
                parameters = {"service_name": entities["service_name"][0]}
            else:
                cypher = """
                MATCH (s:Service)
                RETURN s
                ORDER BY s.name
                """
        
        elif intent == "trace_data":
            if "service_name" in entities:
                cypher = QueryTemplate.TRACE_DATA_FLOW
                parameters = {"source_name": entities["service_name"][0]}
            else:
                cypher = QueryTemplate.DATABASE_CONNECTIONS
        
        elif intent == "system_overview":
            cypher = QueryTemplate.SYSTEM_OVERVIEW
        
        elif intent == "agent_capabilities":
            cypher = QueryTemplate.CAPABILITY_COVERAGE
        
        else:
            # Default query
            cypher = """
            MATCH (n)
            RETURN n
            LIMIT 10
            """
        
        return cypher, parameters


class QueryEngine:
    """
    Main query engine for the SutazAI knowledge graph
    """
    
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j_manager = neo4j_manager
        self.nlp_processor = NaturalLanguageProcessor()
        self.logger = logging.getLogger("query_engine")
        
        # Query cache for performance
        self.query_cache = {}
        self.cache_size_limit = 100
    
    async def find_agents_by_capability(self, capabilities: List[str], 
                                      min_match: int = 1,
                                      include_health: bool = True) -> QueryResult:
        """Find agents with specific capabilities"""
        start_time = datetime.now()
        
        cypher = QueryTemplate.FIND_AGENTS_BY_CAPABILITY
        if include_health:
            cypher += " AND a.health_status IN ['healthy', 'warning']"
        
        parameters = {
            "capabilities": capabilities,
            "min_capabilities": min_match
        }
        
        results = await self.neo4j_manager.execute_cypher(cypher, parameters)
        
        query_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return QueryResult(
            nodes=[record["a"] for record in results if "a" in record],
            relationships=[],
            metadata={
                "query_type": "find_agents_by_capability",
                "capabilities_requested": capabilities,
                "min_match": min_match
            },
            query_time_ms=query_time,
            total_results=len(results)
        )
    
    async def analyze_service_dependencies(self, service_name: str, 
                                         max_depth: int = 3) -> QueryResult:
        """Analyze service dependencies and impact"""
        start_time = datetime.now()
        
        # Get dependencies
        deps_cypher = QueryTemplate.FIND_SERVICE_DEPENDENCIES
        deps_results = await self.neo4j_manager.execute_cypher(
            deps_cypher, {"service_name": service_name}
        )
        
        # Get dependents (services that depend on this one)
        dependents_cypher = QueryTemplate.FIND_SERVICE_DEPENDENTS
        dependents_results = await self.neo4j_manager.execute_cypher(
            dependents_cypher, {"service_name": service_name}
        )
        
        # Get impact analysis
        impact_cypher = QueryTemplate.SERVICE_IMPACT_ANALYSIS
        impact_results = await self.neo4j_manager.execute_cypher(
            impact_cypher, {"service_name": service_name}
        )
        
        query_time = (datetime.now() - start_time).total_seconds() * 1000
        
        all_nodes = []
        all_relationships = []
        
        # Process dependencies
        for record in deps_results:
            if "s" in record:
                all_nodes.append(record["s"])
            if "dep" in record:
                all_nodes.append(record["dep"])
        
        return QueryResult(
            nodes=all_nodes,
            relationships=all_relationships,
            metadata={
                "query_type": "service_dependencies",
                "service_name": service_name,
                "dependencies_count": len(deps_results),
                "dependents_count": len(dependents_results),
                "impact_analysis": impact_results
            },
            query_time_ms=query_time,
            total_results=len(deps_results) + len(dependents_results)
        )
    
    async def trace_data_flow(self, source_name: str, 
                            max_hops: int = 5) -> QueryResult:
        """Trace data flow from a source component"""
        start_time = datetime.now()
        
        cypher = QueryTemplate.TRACE_DATA_FLOW
        parameters = {"source_name": source_name}
        
        results = await self.neo4j_manager.execute_cypher(cypher, parameters)
        
        query_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Process path results
        flow_paths = []
        for record in results:
            if "flow_path" in record and "flow_types" in record:
                flow_paths.append({
                    "path": record["flow_path"],
                    "operations": record["flow_types"]
                })
        
        return QueryResult(
            nodes=[],
            relationships=[],
            metadata={
                "query_type": "data_flow_trace",
                "source_name": source_name,
                "flow_paths": flow_paths
            },
            query_time_ms=query_time,
            total_results=len(flow_paths)
        )
    
    async def get_system_overview(self) -> QueryResult:
        """Get comprehensive system overview"""
        start_time = datetime.now()
        
        # Get node counts by type
        overview_cypher = QueryTemplate.SYSTEM_OVERVIEW
        overview_results = await self.neo4j_manager.execute_cypher(overview_cypher)
        
        # Get capability coverage
        capability_cypher = QueryTemplate.CAPABILITY_COVERAGE
        capability_results = await self.neo4j_manager.execute_cypher(capability_cypher)
        
        # Get database connections
        db_cypher = QueryTemplate.DATABASE_CONNECTIONS
        db_results = await self.neo4j_manager.execute_cypher(db_cypher)
        
        query_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return QueryResult(
            nodes=[],
            relationships=[],
            metadata={
                "query_type": "system_overview",
                "node_counts": {record["node_type"]: record["count"] 
                              for record in overview_results},
                "capability_coverage": capability_results,
                "database_connections": db_results
            },
            query_time_ms=query_time,
            total_results=len(overview_results)
        )
    
    async def find_optimal_agent_for_task(self, required_capabilities: List[str],
                                        preferred_type: Optional[str] = None,
                                        performance_weight: float = 0.3) -> QueryResult:
        """Find the optimal agent for a specific task"""
        start_time = datetime.now()
        
        # Build scoring query
        cypher = """
        MATCH (a:Agent)-[:HAS_CAPABILITY]->(c:Capability)
        WHERE c.name IN $capabilities
        WITH a, collect(c.name) as agent_capabilities,
             size([cap IN $capabilities WHERE cap IN collect(c.name)]) as matched_caps
        WHERE matched_caps > 0
        """
        
        # Add type filter if specified
        if preferred_type:
            cypher += " AND a.agent_type = $preferred_type"
        
        # Add scoring and ordering
        cypher += """
        WITH a, agent_capabilities, matched_caps,
             toFloat(matched_caps) / size($capabilities) as capability_score,
             CASE WHEN a.performance_metrics IS NOT NULL 
                  THEN coalesce(a.performance_metrics.success_rate, 0.0) / 100.0 
                  ELSE 0.5 END as performance_score,
             CASE WHEN a.health_status = 'healthy' THEN 1.0
                  WHEN a.health_status = 'warning' THEN 0.7
                  ELSE 0.3 END as health_score
        WITH a, agent_capabilities, matched_caps,
             capability_score * 0.5 + performance_score * $performance_weight + health_score * 0.2 as total_score
        ORDER BY total_score DESC, matched_caps DESC
        LIMIT 10
        RETURN a, agent_capabilities, matched_caps, total_score
        """
        
        parameters = {
            "capabilities": required_capabilities,
            "performance_weight": performance_weight
        }
        
        if preferred_type:
            parameters["preferred_type"] = preferred_type
        
        results = await self.neo4j_manager.execute_cypher(cypher, parameters)
        
        query_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Process results to include scoring information
        scored_agents = []
        for record in results:
            agent_data = record["a"]
            agent_data["_score_info"] = {
                "total_score": record.get("total_score", 0),
                "matched_capabilities": record.get("matched_caps", 0),
                "available_capabilities": record.get("agent_capabilities", [])
            }
            scored_agents.append(agent_data)
        
        return QueryResult(
            nodes=scored_agents,
            relationships=[],
            metadata={
                "query_type": "optimal_agent_selection",
                "required_capabilities": required_capabilities,
                "preferred_type": preferred_type,
                "scoring_weights": {
                    "capability_match": 0.5,
                    "performance": performance_weight,
                    "health": 0.2
                }
            },
            query_time_ms=query_time,
            total_results=len(results)
        )
    
    async def process_natural_language_query(self, query: str) -> QueryResult:
        """Process a natural language query"""
        start_time = datetime.now()
        
        # Process the query
        query_info = self.nlp_processor.process_query(query)
        
        # Execute the generated Cypher
        results = await self.neo4j_manager.execute_cypher(
            query_info["cypher"], 
            query_info["parameters"]
        )
        
        query_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Extract nodes and relationships from results
        nodes = []
        relationships = []
        
        for record in results:
            for key, value in record.items():
                if isinstance(value, dict) and "id" in value:
                    # This looks like a node
                    nodes.append(value)
        
        return QueryResult(
            nodes=nodes,
            relationships=relationships,
            metadata={
                "query_type": "natural_language",
                "original_query": query,
                "detected_intent": query_info["intent"],
                "extracted_entities": query_info["entities"],
                "generated_cypher": query_info["cypher"]
            },
            query_time_ms=query_time,
            total_results=len(results)
        )
    
    async def get_agent_network(self, agent_id: str, 
                              relationship_types: Optional[List[str]] = None,
                              max_depth: int = 2) -> QueryResult:
        """Get the network of relationships around an agent"""
        start_time = datetime.now()
        
        # Build relationship filter
        rel_filter = ""
        if relationship_types:
            rel_types = "|".join(relationship_types)
            rel_filter = f":{rel_types}"
        
        cypher = f"""
        MATCH path = (a:Agent {{id: $agent_id}})-[r{rel_filter}*1..{max_depth}]-(connected)
        RETURN path, 
               [node in nodes(path) | node] as path_nodes,
               [rel in relationships(path) | rel] as path_relationships
        """
        
        parameters = {"agent_id": agent_id}
        
        results = await self.neo4j_manager.execute_cypher(cypher, parameters)
        
        query_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Process paths to extract unique nodes and relationships
        all_nodes = {}
        all_relationships = {}
        
        for record in results:
            if "path_nodes" in record:
                for node in record["path_nodes"]:
                    if node and "id" in node:
                        all_nodes[node["id"]] = node
            
            if "path_relationships" in record:
                for rel in record["path_relationships"]:
                    if rel and "id" in rel:
                        all_relationships[rel["id"]] = rel
        
        return QueryResult(
            nodes=list(all_nodes.values()),
            relationships=list(all_relationships.values()),
            metadata={
                "query_type": "agent_network",
                "agent_id": agent_id,
                "relationship_types": relationship_types,
                "max_depth": max_depth,
                "network_size": len(all_nodes)
            },
            query_time_ms=query_time,
            total_results=len(all_nodes) + len(all_relationships)
        )
    
    async def execute_custom_query(self, cypher: str, 
                                 parameters: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute a custom Cypher query"""
        start_time = datetime.now()
        
        results = await self.neo4j_manager.execute_cypher(cypher, parameters or {})
        
        query_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return QueryResult(
            nodes=[],
            relationships=[],
            metadata={
                "query_type": "custom",
                "cypher": cypher,
                "parameters": parameters
            },
            query_time_ms=query_time,
            total_results=len(results)
        )
    
    def get_query_suggestions(self, partial_query: str) -> List[str]:
        """Get query suggestions based on partial input"""
        suggestions = []
        
        query_lower = partial_query.lower()
        
        # Common query patterns
        common_queries = [
            "Find agents with code generation capability",
            "Show service dependencies for api_service",
            "Trace data flow from database to services",
            "What agents can do security analysis?",
            "List all available agents",
            "Show system architecture overview",
            "Find workflows using orchestration agents",
            "Which services depend on redis?",
            "Show agent network for agent_123",
            "Find optimal agent for testing tasks"
        ]
        
        # Filter suggestions based on partial query
        for suggestion in common_queries:
            if any(word in suggestion.lower() for word in query_lower.split()):
                suggestions.append(suggestion)
        
        return suggestions[:5]  # Return top 5 suggestions