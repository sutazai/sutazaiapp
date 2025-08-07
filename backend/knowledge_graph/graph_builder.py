"""
Knowledge Graph Builder
======================

Extracts knowledge from the SutazAI codebase and documentation to build
a comprehensive knowledge graph. Analyzes agent capabilities, service
dependencies, data flow patterns, and system architecture.

Features:
- Code analysis and parsing
- Documentation extraction
- Agent capability detection
- Service dependency mapping
- API endpoint discovery
- Configuration analysis
- Automated graph construction
"""

import ast
import os
import re
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
import yaml

from .schema import (
    NodeType, RelationshipType,
    AgentNode, ServiceNode, DatabaseNode, WorkflowNode, 
    CapabilityNode, ModelNode, DocumentNode,
    RelationshipProperties, KnowledgeGraphSchema,
    CAPABILITY_CATEGORIES, SERVICE_TYPES
)
from .neo4j_manager import Neo4jManager


class CodeAnalyzer:
    """Analyzes Python code to extract structural information"""
    
    def __init__(self):
        self.logger = logging.getLogger("code_analyzer")
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Python file and extract information"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            analysis = {
                "file_path": file_path,
                "classes": [],
                "functions": [],
                "imports": [],
                "constants": [],
                "docstring": ast.get_docstring(tree),
                "decorators": [],
                "dependencies": set(),
                "capabilities": set(),
                "agent_types": set()
            }
            
            for node in ast.walk(tree):
                self._analyze_node(node, analysis)
            
            # Convert sets to lists for JSON serialization
            analysis["dependencies"] = list(analysis["dependencies"])
            analysis["capabilities"] = list(analysis["capabilities"])
            analysis["agent_types"] = list(analysis["agent_types"])
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
            return {"file_path": file_path, "error": str(e)}
    
    def _analyze_node(self, node: ast.AST, analysis: Dict[str, Any]):
        """Analyze individual AST node"""
        if isinstance(node, ast.ClassDef):
            class_info = {
                "name": node.name,
                "bases": [self._get_name(base) for base in node.bases],
                "methods": [],
                "docstring": ast.get_docstring(node),
                "decorators": [self._get_decorator_name(dec) for dec in node.decorator_list],
                "line_number": node.lineno
            }
            
            # Analyze methods
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_info = {
                        "name": item.name,
                        "args": [arg.arg for arg in item.args.args],
                        "decorators": [self._get_decorator_name(dec) for dec in item.decorator_list],
                        "docstring": ast.get_docstring(item),
                        "is_async": isinstance(item, ast.AsyncFunctionDef)
                    }
                    class_info["methods"].append(method_info)
            
            analysis["classes"].append(class_info)
            
            # Detect agent classes
            if any("agent" in base.lower() for base in class_info["bases"]):
                analysis["agent_types"].add(node.name)
                
                # Extract capabilities from class name or docstring
                capabilities = self._extract_capabilities(node.name, class_info["docstring"])
                analysis["capabilities"].update(capabilities)
        
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            func_info = {
                "name": node.name,
                "args": [arg.arg for arg in node.args.args],
                "decorators": [self._get_decorator_name(dec) for dec in node.decorator_list],
                "docstring": ast.get_docstring(node),
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "line_number": node.lineno
            }
            analysis["functions"].append(func_info)
        
        elif isinstance(node, ast.Import):
            for alias in node.names:
                analysis["imports"].append(alias.name)
                analysis["dependencies"].add(alias.name.split('.')[0])
        
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                analysis["imports"].append(node.module)
                analysis["dependencies"].add(node.module.split('.')[0])
        
        elif isinstance(node, ast.Assign):
            # Look for constants (uppercase variables)
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    try:
                        value = ast.literal_eval(node.value)
                        analysis["constants"].append({
                            "name": target.id,
                            "value": value,
                            "line_number": node.lineno
                        })
                    except:
                        pass
    
    def _get_name(self, node: ast.AST) -> str:
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        else:
            return str(node)
    
    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """Get decorator name from AST node"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
            return decorator.func.id
        else:
            return str(decorator)
    
    def _extract_capabilities(self, class_name: str, docstring: Optional[str]) -> Set[str]:
        """Extract capabilities from class name and docstring"""
        capabilities = set()
        
        # Capability keywords to look for
        capability_keywords = {
            "code": ["code", "generation", "analysis", "review"],
            "security": ["security", "vulnerability", "audit", "compliance"],
            "test": ["test", "testing", "validation", "verification"],
            "deploy": ["deploy", "deployment", "provision", "infrastructure"],
            "monitor": ["monitor", "monitoring", "health", "metrics"],
            "orchestrat": ["orchestrat", "coordination", "workflow", "task"],
            "communication": ["communication", "message", "protocol", "api"],
            "reasoning": ["reasoning", "decision", "planning", "intelligence"],
            "data": ["data", "processing", "analysis", "transformation"]
        }
        
        text = f"{class_name} {docstring or ''}".lower()
        
        for capability, keywords in capability_keywords.items():
            if any(keyword in text for keyword in keywords):
                capabilities.add(capability)
        
        return capabilities


class DocumentationParser:
    """Parses documentation files to extract system information"""
    
    def __init__(self):
        self.logger = logging.getLogger("doc_parser")
    
    def parse_markdown_file(self, file_path: str) -> Dict[str, Any]:
        """Parse markdown documentation file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            doc_info = {
                "file_path": file_path,
                "title": self._extract_title(content),
                "sections": self._extract_sections(content),
                "endpoints": self._extract_endpoints(content),
                "services": self._extract_services(content),
                "dependencies": self._extract_dependencies(content),
                "configurations": self._extract_configurations(content),
                "word_count": len(content.split())
            }
            
            return doc_info
            
        except Exception as e:
            self.logger.error(f"Error parsing markdown file {file_path}: {e}")
            return {"file_path": file_path, "error": str(e)}
    
    def _extract_title(self, content: str) -> str:
        """Extract document title"""
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        return match.group(1).strip() if match else ""
    
    def _extract_sections(self, content: str) -> List[Dict[str, str]]:
        """Extract document sections"""
        sections = []
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_section:
                    sections.append({
                        "level": len(current_section["marker"]),
                        "title": current_section["title"],
                        "content": '\n'.join(current_content).strip()
                    })
                
                # Start new section
                current_section = {
                    "marker": header_match.group(1),
                    "title": header_match.group(2).strip()
                }
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_section:
            sections.append({
                "level": len(current_section["marker"]),
                "title": current_section["title"],
                "content": '\n'.join(current_content).strip()
            })
        
        return sections
    
    def _extract_endpoints(self, content: str) -> List[Dict[str, str]]:
        """Extract API endpoints from documentation"""
        endpoints = []
        
        # Pattern for API endpoints
        endpoint_patterns = [
            r'`(GET|POST|PUT|DELETE|PATCH)\s+([^`]+)`',
            r'(GET|POST|PUT|DELETE|PATCH)\s+`([^`]+)`',
            r'- `(GET|POST|PUT|DELETE|PATCH)\s+([^`]+)`'
        ]
        
        for pattern in endpoint_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                endpoints.append({
                    "method": match[0].upper(),
                    "path": match[1].strip()
                })
        
        return endpoints
    
    def _extract_services(self, content: str) -> List[str]:
        """Extract service names from documentation"""
        services = set()
        
        # Look for service patterns
        service_patterns = [
            r'(\w+)\s+[Ss]ervice',
            r'[Ss]ervice:\s*(\w+)',
            r'`(\w+_service|service_\w+)`',
            r'## (\w+) Service',
            r'### (\w+) Component'
        ]
        
        for pattern in service_patterns:
            matches = re.findall(pattern, content)
            services.update(match.lower() for match in matches if len(match) > 2)
        
        return list(services)
    
    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract dependencies mentioned in documentation"""
        dependencies = set()
        
        # Common dependency patterns
        dep_patterns = [
            r'pip install\s+([a-zA-Z0-9\-_]+)',
            r'npm install\s+([a-zA-Z0-9\-_@/]+)',
            r'import\s+([a-zA-Z0-9\-_]+)',
            r'from\s+([a-zA-Z0-9\-_]+)\s+import',
            r'`([a-zA-Z0-9\-_]+)==[\d\.]+`'
        ]
        
        for pattern in dep_patterns:
            matches = re.findall(pattern, content)
            dependencies.update(matches)
        
        return list(dependencies)
    
    def _extract_configurations(self, content: str) -> List[Dict[str, Any]]:
        """Extract configuration examples from documentation"""
        configs = []
        
        # Look for JSON/YAML configuration blocks
        json_blocks = re.findall(r'```json\n(.*?)\n```', content, re.DOTALL)
        for block in json_blocks:
            try:
                config = json.loads(block)
                configs.append({"type": "json", "config": config})
            except:
                pass
        
        yaml_blocks = re.findall(r'```yaml\n(.*?)\n```', content, re.DOTALL)
        for block in yaml_blocks:
            try:
                config = yaml.safe_load(block)
                configs.append({"type": "yaml", "config": config})
            except:
                pass
        
        return configs


class SystemDiscovery:
    """Discovers system components by analyzing the codebase"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.code_analyzer = CodeAnalyzer()
        self.doc_parser = DocumentationParser()
        self.logger = logging.getLogger("system_discovery")
    
    def discover_agents(self) -> List[Dict[str, Any]]:
        """Discover AI agents in the codebase"""
        agents = []
        
        # Search for agent files
        agent_patterns = [
            "**/ai_agents/**/*.py",
            "**/agents/**/*.py", 
            "**/*agent*.py"
        ]
        
        for pattern in agent_patterns:
            for file_path in self.base_path.glob(pattern):
                if file_path.is_file():
                    analysis = self.code_analyzer.analyze_file(str(file_path))
                    
                    # Extract agent information
                    for class_info in analysis.get("classes", []):
                        if self._is_agent_class(class_info):
                            agent_info = {
                                "file_path": str(file_path),
                                "class_name": class_info["name"],
                                "capabilities": analysis.get("capabilities", []),
                                "methods": class_info["methods"],
                                "docstring": class_info["docstring"],
                                "base_classes": class_info["bases"]
                            }
                            agents.append(agent_info)
        
        return agents
    
    def discover_services(self) -> List[Dict[str, Any]]:
        """Discover services in the codebase"""
        services = []
        
        # Search for service files
        service_patterns = [
            "**/services/**/*.py",
            "**/app/**/*.py",
            "**/*service*.py",
            "**/api/**/*.py"
        ]
        
        for pattern in service_patterns:
            for file_path in self.base_path.glob(pattern):
                if file_path.is_file():
                    analysis = self.code_analyzer.analyze_file(str(file_path))
                    
                    # Extract service information
                    service_info = self._extract_service_info(file_path, analysis)
                    if service_info:
                        services.append(service_info)
        
        return services
    
    def discover_databases(self) -> List[Dict[str, Any]]:
        """Discover database configurations"""
        databases = []
        
        # Search for database configuration files
        config_patterns = [
            "**/config/**/*.py",
            "**/config/**/*.json",
            "**/config/**/*.yaml",
            "**/database.py",
            "**/db.py"
        ]
        
        for pattern in config_patterns:
            for file_path in self.base_path.glob(pattern):
                if file_path.is_file():
                    db_info = self._extract_database_info(file_path)
                    if db_info:
                        databases.extend(db_info)
        
        return databases
    
    def discover_workflows(self) -> List[Dict[str, Any]]:
        """Discover workflow definitions"""
        workflows = []
        
        # Search for workflow files
        workflow_patterns = [
            "**/workflows/**/*.py",
            "**/orchestration/**/*.py",
            "**/*workflow*.py",
            "**/*orchestrator*.py"
        ]
        
        for pattern in workflow_patterns:
            for file_path in self.base_path.glob(pattern):
                if file_path.is_file():
                    analysis = self.code_analyzer.analyze_file(str(file_path))
                    workflow_info = self._extract_workflow_info(file_path, analysis)
                    if workflow_info:
                        workflows.extend(workflow_info)
        
        return workflows
    
    def discover_documentation(self) -> List[Dict[str, Any]]:
        """Discover and parse documentation files"""
        docs = []
        
        # Search for documentation files
        doc_patterns = [
            "**/*.md",
            "**/*.rst", 
            "**/docs/**/*.txt"
        ]
        
        for pattern in doc_patterns:
            for file_path in self.base_path.glob(pattern):
                if file_path.is_file():
                    if file_path.suffix.lower() == '.md':
                        doc_info = self.doc_parser.parse_markdown_file(str(file_path))
                        docs.append(doc_info)
        
        return docs
    
    def _is_agent_class(self, class_info: Dict[str, Any]) -> bool:
        """Determine if a class is an agent"""
        name = class_info["name"].lower()
        bases = [base.lower() for base in class_info["bases"]]
        
        # Check if it's an agent class
        agent_indicators = [
            "agent" in name,
            any("agent" in base for base in bases),
            any("baseagent" in base for base in bases)
        ]
        
        return any(agent_indicators)
    
    def _extract_service_info(self, file_path: Path, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract service information from file analysis"""
        # Look for service indicators
        file_name = file_path.name.lower()
        
        if any(keyword in file_name for keyword in ["service", "api", "server", "app"]):
            service_info = {
                "file_path": str(file_path),
                "name": file_path.stem,
                "type": self._determine_service_type(file_path, analysis),
                "classes": analysis.get("classes", []),
                "functions": analysis.get("functions", []),
                "dependencies": analysis.get("dependencies", []),
                "endpoints": self._extract_endpoints_from_code(analysis)
            }
            return service_info
        
        return None
    
    def _determine_service_type(self, file_path: Path, analysis: Dict[str, Any]) -> str:
        """Determine the type of service"""
        file_name = file_path.name.lower()
        dependencies = analysis.get("dependencies", [])
        
        if "fastapi" in dependencies or "api" in file_name:
            return "api_service"
        elif "database" in file_name or "db" in file_name:
            return "database_service"
        elif "queue" in file_name or "worker" in file_name:
            return "worker_service"
        elif "monitor" in file_name:
            return "monitoring_service"
        else:
            return "generic_service"
    
    def _extract_endpoints_from_code(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract API endpoints from code analysis"""
        endpoints = []
        
        for class_info in analysis.get("classes", []):
            for method in class_info.get("methods", []):
                # Look for HTTP method decorators
                for decorator in method.get("decorators", []):
                    if decorator.lower() in ["get", "post", "put", "delete", "patch"]:
                        endpoints.append({
                            "method": decorator.upper(),
                            "function": method["name"],
                            "class": class_info["name"]
                        })
        
        return endpoints
    
    def _extract_database_info(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract database information from configuration files"""
        databases = []
        
        try:
            if file_path.suffix == '.py':
                # Analyze Python configuration
                analysis = self.code_analyzer.analyze_file(str(file_path))
                
                # Look for database configurations in constants
                for constant in analysis.get("constants", []):
                    if any(keyword in constant["name"].lower() 
                          for keyword in ["database", "db", "connection", "url"]):
                        
                        db_info = {
                            "name": constant["name"],
                            "type": self._infer_db_type(constant["value"]),
                            "config_file": str(file_path),
                            "config_value": constant["value"]
                        }
                        databases.append(db_info)
            
            elif file_path.suffix in ['.json', '.yaml', '.yml']:
                # Parse configuration files
                with open(file_path, 'r') as f:
                    if file_path.suffix == '.json':
                        config = json.load(f)
                    else:
                        config = yaml.safe_load(f)
                
                # Look for database configurations
                db_configs = self._find_db_configs(config)
                for db_config in db_configs:
                    db_config["config_file"] = str(file_path)
                    databases.append(db_config)
        
        except Exception as e:
            self.logger.debug(f"Could not parse config file {file_path}: {e}")
        
        return databases
    
    def _infer_db_type(self, connection_string: str) -> str:
        """Infer database type from connection string"""
        if isinstance(connection_string, str):
            conn_lower = connection_string.lower()
            if "postgresql" in conn_lower or "psql" in conn_lower:
                return "postgresql"
            elif "redis" in conn_lower:
                return "redis"
            elif "neo4j" in conn_lower:
                return "neo4j"
            elif "mongodb" in conn_lower or "mongo" in conn_lower:
                return "mongodb"
            elif "sqlite" in conn_lower:
                return "sqlite"
        
        return "unknown"
    
    def _find_db_configs(self, config: Dict[str, Any], path: str = "") -> List[Dict[str, Any]]:
        """Recursively find database configurations in config dict"""
        db_configs = []
        
        if isinstance(config, dict):
            for key, value in config.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check if this looks like a database config
                if any(keyword in key.lower() for keyword in ["database", "db", "redis", "neo4j", "postgres"]):
                    if isinstance(value, (str, dict)):
                        db_config = {
                            "name": key,
                            "type": self._infer_db_type(str(value)),
                            "config_path": current_path,
                            "config": value
                        }
                        db_configs.append(db_config)
                
                # Recurse into nested dictionaries
                elif isinstance(value, dict):
                    db_configs.extend(self._find_db_configs(value, current_path))
        
        return db_configs
    
    def _extract_workflow_info(self, file_path: Path, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract workflow information from file analysis"""
        workflows = []
        
        for class_info in analysis.get("classes", []):
            if self._is_workflow_class(class_info):
                workflow_info = {
                    "file_path": str(file_path),
                    "name": class_info["name"],
                    "type": "workflow",
                    "methods": class_info["methods"],
                    "docstring": class_info["docstring"],
                    "dependencies": analysis.get("dependencies", [])
                }
                workflows.append(workflow_info)
        
        return workflows
    
    def _is_workflow_class(self, class_info: Dict[str, Any]) -> bool:
        """Determine if a class represents a workflow"""
        name = class_info["name"].lower()
        
        workflow_indicators = [
            "workflow" in name,
            "orchestrator" in name,
            "pipeline" in name,
            "processor" in name
        ]
        
        return any(workflow_indicators)


class KnowledgeGraphBuilder:
    """
    Main class for building the knowledge graph from discovered system components
    """
    
    def __init__(self, base_path: str, neo4j_manager: Neo4jManager):
        self.base_path = base_path
        self.neo4j_manager = neo4j_manager
        self.schema = KnowledgeGraphSchema()
        self.discovery = SystemDiscovery(base_path)
        self.logger = logging.getLogger("kg_builder")
    
    async def build_knowledge_graph(self) -> Dict[str, Any]:
        """Build the complete knowledge graph"""
        self.logger.info("Starting knowledge graph construction")
        
        build_stats = {
            "nodes_created": 0,
            "relationships_created": 0,
            "agents_discovered": 0,
            "services_discovered": 0,
            "databases_discovered": 0,
            "workflows_discovered": 0,
            "documents_processed": 0,
            "errors": 0
        }
        
        try:
            # Discover system components
            self.logger.info("Discovering system components...")
            
            agents = self.discovery.discover_agents()
            services = self.discovery.discover_services()
            databases = self.discovery.discover_databases()
            workflows = self.discovery.discover_workflows()
            documentation = self.discovery.discover_documentation()
            
            build_stats.update({
                "agents_discovered": len(agents),
                "services_discovered": len(services),
                "databases_discovered": len(databases),
                "workflows_discovered": len(workflows),
                "documents_processed": len(documentation)
            })
            
            self.logger.info(f"Discovery complete: {sum(build_stats.values())} components found")
            
            # Create nodes
            all_nodes = []
            all_relationships = []
            
            # Create agent nodes
            agent_nodes, agent_relationships = await self._create_agent_nodes(agents)
            all_nodes.extend(agent_nodes)
            all_relationships.extend(agent_relationships)
            
            # Create service nodes
            service_nodes, service_relationships = await self._create_service_nodes(services)
            all_nodes.extend(service_nodes)
            all_relationships.extend(service_relationships)
            
            # Create database nodes
            database_nodes = await self._create_database_nodes(databases)
            all_nodes.extend(database_nodes)
            
            # Create workflow nodes
            workflow_nodes, workflow_relationships = await self._create_workflow_nodes(workflows)
            all_nodes.extend(workflow_nodes)
            all_relationships.extend(workflow_relationships)
            
            # Create document nodes
            document_nodes = await self._create_document_nodes(documentation)
            all_nodes.extend(document_nodes)
            
            # Create capability nodes
            capability_nodes = await self._create_capability_nodes()
            all_nodes.extend(capability_nodes)
            
            # Store in Neo4j
            self.logger.info(f"Storing {len(all_nodes)} nodes and {len(all_relationships)} relationships")
            
            nodes_created = await self.neo4j_manager.create_nodes_batch(all_nodes)
            relationships_created = await self.neo4j_manager.create_relationships_batch(all_relationships)
            
            build_stats["nodes_created"] = nodes_created
            build_stats["relationships_created"] = relationships_created
            
            self.logger.info("Knowledge graph construction completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error building knowledge graph: {e}")
            build_stats["errors"] += 1
        
        return build_stats
    
    async def _create_agent_nodes(self, agents: List[Dict[str, Any]]) -> Tuple[List[AgentNode], List[RelationshipProperties]]:
        """Create agent nodes and their relationships"""
        nodes = []
        relationships = []
        
        for agent_info in agents:
            # Create agent node
            agent_node = AgentNode(
                name=agent_info["class_name"],
                description=agent_info.get("docstring", ""),
                agent_type=self._classify_agent_type(agent_info),
                capabilities=set(agent_info.get("capabilities", [])),
                metadata={
                    "file_path": agent_info["file_path"],
                    "methods": agent_info["methods"],
                    "base_classes": agent_info["base_classes"]
                }
            )
            nodes.append(agent_node)
            
            # Create capability relationships
            for capability in agent_info.get("capabilities", []):
                relationship = RelationshipProperties(
                    source_id=agent_node.id,
                    target_id=f"capability_{capability}",
                    type=RelationshipType.HAS_CAPABILITY,
                    metadata={"source": "code_analysis"}
                )
                relationships.append(relationship)
        
        return nodes, relationships
    
    async def _create_service_nodes(self, services: List[Dict[str, Any]]) -> Tuple[List[ServiceNode], List[RelationshipProperties]]:
        """Create service nodes and their relationships"""
        nodes = []
        relationships = []
        
        for service_info in services:
            # Create service node
            service_node = ServiceNode(
                name=service_info["name"],
                service_type=service_info["type"],
                endpoints=[ep.get("function", "") for ep in service_info.get("endpoints", [])],
                metadata={
                    "file_path": service_info["file_path"],
                    "classes": service_info["classes"],
                    "functions": service_info["functions"]
                }
            )
            nodes.append(service_node)
            
            # Create dependency relationships
            for dependency in service_info.get("dependencies", []):
                # Look for other services this depends on
                if dependency in SERVICE_TYPES:
                    relationship = RelationshipProperties(
                        source_id=service_node.id,
                        target_id=f"service_{dependency}",
                        type=RelationshipType.DEPENDS_ON,
                        metadata={"dependency_type": "code_import"}
                    )
                    relationships.append(relationship)
        
        return nodes, relationships
    
    async def _create_database_nodes(self, databases: List[Dict[str, Any]]) -> List[DatabaseNode]:
        """Create database nodes"""
        nodes = []
        
        for db_info in databases:
            # Create database node
            db_node = DatabaseNode(
                name=db_info["name"],
                database_type=db_info["type"],
                connection_string=str(db_info.get("config_value", "")),
                metadata={
                    "config_file": db_info["config_file"],
                    "config_path": db_info.get("config_path", ""),
                    "config": db_info.get("config", {})
                }
            )
            nodes.append(db_node)
        
        return nodes
    
    async def _create_workflow_nodes(self, workflows: List[Dict[str, Any]]) -> Tuple[List[WorkflowNode], List[RelationshipProperties]]:
        """Create workflow nodes and their relationships"""
        nodes = []
        relationships = []
        
        for workflow_info in workflows:
            # Create workflow node
            workflow_node = WorkflowNode(
                name=workflow_info["name"],
                workflow_type="orchestration",
                steps=[{"method": method["name"], "async": method["is_async"]} 
                      for method in workflow_info.get("methods", [])],
                metadata={
                    "file_path": workflow_info["file_path"],
                    "docstring": workflow_info.get("docstring", "")
                }
            )
            nodes.append(workflow_node)
        
        return nodes, relationships
    
    async def _create_document_nodes(self, documentation: List[Dict[str, Any]]) -> List[DocumentNode]:
        """Create document nodes"""
        nodes = []
        
        for doc_info in documentation:
            if "error" not in doc_info:
                # Create document node
                doc_node = DocumentNode(
                    name=doc_info.get("title", Path(doc_info["file_path"]).name),
                    document_type="markdown",
                    file_path=doc_info["file_path"],
                    content_summary=doc_info.get("title", ""),
                    word_count=doc_info.get("word_count", 0),
                    metadata={
                        "sections": doc_info.get("sections", []),
                        "endpoints": doc_info.get("endpoints", []),
                        "services": doc_info.get("services", []),
                        "dependencies": doc_info.get("dependencies", [])
                    }
                )
                nodes.append(doc_node)
        
        return nodes
    
    async def _create_capability_nodes(self) -> List[CapabilityNode]:
        """Create capability nodes"""
        nodes = []
        
        for category, capabilities in CAPABILITY_CATEGORIES.items():
            for capability in capabilities:
                cap_node = CapabilityNode(
                    name=capability,
                    capability_type=category,
                    metadata={"category": category}
                )
                # Use predictable ID for capability relationships
                cap_node.id = f"capability_{capability}"
                nodes.append(cap_node)
        
        return nodes
    
    def _classify_agent_type(self, agent_info: Dict[str, Any]) -> str:
        """Classify the type of agent based on its information"""
        name = agent_info["class_name"].lower()
        capabilities = agent_info.get("capabilities", [])
        
        # Classification based on name patterns
        if "code" in name or "generator" in name:
            return "code_generation_agent"
        elif "security" in name or "audit" in name:
            return "security_agent"
        elif "test" in name:
            return "testing_agent"
        elif "orchestrat" in name or "coordinator" in name:
            return "orchestration_agent"
        elif "monitor" in name:
            return "monitoring_agent"
        
        # Classification based on capabilities
        if "code" in capabilities:
            return "code_generation_agent"
        elif "security" in capabilities:
            return "security_agent"
        elif "test" in capabilities:
            return "testing_agent"
        elif "orchestrat" in capabilities:
            return "orchestration_agent"
        
        return "generic_agent"