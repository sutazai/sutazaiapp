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
SutazAI Knowledge Graph Builder Demonstration

This script demonstrates the capabilities of the Knowledge Graph Builder
by creating a sample knowledge graph and showcasing various features.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from knowledge_graph_builder import KnowledgeGraphBuilder, Entity, Relationship, EntityType, RelationType, EnforcementLevel
from visualization import KnowledgeGraphVisualizer
from cli import KnowledgeGraphCLI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeGraphDemo:
    """Demonstration of SutazAI Knowledge Graph capabilities"""
    
    def __init__(self, output_dir: str = "./demo_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.kg_builder = None
        self.visualizer = None
        
    async def initialize(self):
        """Initialize the knowledge graph builder"""
        logger.info("🚀 Initializing SutazAI Knowledge Graph Builder Demo")
        
        # Create config for demo
        demo_config = {
            "neo4j": {
                "uri": "bolt://localhost:7687",
                "user": "neo4j", 
                "password": "sutazai123"
            },
            "extraction": {"confidence_threshold": 0.7},
            "reasoning": {"max_hops": 3},
            "visualization": {"max_nodes": 100}
        }
        
        # Save demo config
        config_path = self.output_dir / "demo_config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(demo_config, f, indent=2)
        
        # Initialize builder
        self.kg_builder = KnowledgeGraphBuilder(config_path=str(config_path))
        self.visualizer = KnowledgeGraphVisualizer(self.kg_builder)
        
        logger.info("✅ Knowledge Graph Builder initialized")
        
    async def create_sample_knowledge_graph(self):
        """Create a sample knowledge graph with SutazAI hygiene standards"""
        logger.info("🏗️  Creating sample knowledge graph")
        
        # Create sample entities representing SutazAI standards
        sample_entities = [
            # Core Rules
            Entity(
                id="rule_no_fantasy",
                type=EntityType.RULE,
                name="Rule 1: No Fantasy Elements",
                description="Only real, production-ready implementations are allowed",
                properties={
                    "enforcement_level": EnforcementLevel.BLOCKING,
                    "category": "professional_standards",
                    "rule_number": 1
                }
            ),
            Entity(
                id="rule_preserve_functionality", 
                type=EntityType.RULE,
                name="Rule 2: Preserve Existing Functionality",
                description="Every change must respect what already works",
                properties={
                    "enforcement_level": EnforcementLevel.BLOCKING,
                    "category": "stability",
                    "rule_number": 2
                }
            ),
            Entity(
                id="rule_professional_project",
                type=EntityType.RULE,
                name="Rule 3: Professional Project Treatment",
                description="Approach every task with professional mindset",
                properties={
                    "enforcement_level": EnforcementLevel.WARNING,
                    "category": "professional_standards",
                    "rule_number": 3
                }
            ),
            
            # Standards
            Entity(
                id="standard_docker_excellence",
                type=EntityType.STANDARD,
                name="Docker Excellence",
                description="All Docker configurations must be clean, efficient, and structured",
                properties={
                    "enforcement_level": EnforcementLevel.WARNING,
                    "category": "technical_standards"
                }
            ),
            Entity(
                id="standard_health_checks",
                type=EntityType.STANDARD,
                name="Health Check Requirements",
                description="Every service must implement comprehensive health checks",
                properties={
                    "enforcement_level": EnforcementLevel.BLOCKING,
                    "category": "process_standards"
                }
            ),
            Entity(
                id="standard_code_reuse",
                type=EntityType.STANDARD,
                name="Code Reuse and DRY Principles",
                description="Always reuse existing code before creating new implementations",
                properties={
                    "enforcement_level": EnforcementLevel.WARNING,
                    "category": "technical_standards"
                }
            ),
            
            # Sample Agents
            Entity(
                id="agent_knowledge_graph_builder",
                type=EntityType.AGENT,
                name="knowledge-graph-builder",
                description="Builds knowledge graphs for codebase hygiene standards",
                properties={
                    "agent_type": "knowledge-graph-builder",
                    "container_name": "sutazai-knowledge-graph-builder",
                    "health_check": True,
                    "resource_limits": {"cpu": "2", "memory": "4G"},
                    "networks": ["sutazai-network"]
                }
            ),
            Entity(
                id="agent_senior_ai_engineer",
                type=EntityType.AGENT,
                name="senior-ai-engineer", 
                description="Senior AI engineering agent for complex development tasks",
                properties={
                    "agent_type": "senior-ai-engineer",
                    "container_name": "sutazai-senior-ai-engineer",
                    "health_check": True,
                    "resource_limits": {"cpu": "0.5", "memory": "512M"},
                    "networks": ["sutazai-network"]
                }
            ),
            Entity(
                id="agent_testing_qa_validator",
                type=EntityType.AGENT,
                name="testing-qa-validator",
                description="QA validation and testing automation agent",
                properties={
                    "agent_type": "testing-qa-validator",
                    "container_name": "sutazai-testing-qa-validator",
                    "health_check": True,
                    "resource_limits": {"cpu": "0.5", "memory": "512M"},
                    "networks": ["sutazai-network"]
                }
            ),
            
            # Requirements
            Entity(
                id="req_health_endpoint",
                type=EntityType.REQUIREMENT,
                name="/health endpoint",
                description="All services must expose a /health endpoint",
                properties={
                    "enforcement_level": EnforcementLevel.BLOCKING,
                    "category": "health_checks"
                }
            ),
            Entity(
                id="req_resource_limits",
                type=EntityType.REQUIREMENT,
                name="Resource Limits",
                description="All containers must have CPU and memory limits defined",
                properties={
                    "enforcement_level": EnforcementLevel.WARNING,
                    "category": "resource_management"
                }
            ),
            Entity(
                id="req_naming_convention",
                type=EntityType.REQUIREMENT,
                name="Naming Convention",
                description="Container names must follow sutazai- prefix convention",
                properties={
                    "enforcement_level": EnforcementLevel.WARNING,
                    "category": "conventions"
                }
            )
        ]
        
        # Create sample relationships
        sample_relationships = [
            # Rules enforce standards
            Relationship(
                source_id="rule_no_fantasy",
                target_id="standard_docker_excellence",
                type=RelationType.ENFORCES,
                confidence=1.0,
                properties={"relationship_type": "rule_to_standard"}
            ),
            Relationship(
                source_id="rule_preserve_functionality",
                target_id="standard_health_checks",
                type=RelationType.ENFORCES,
                confidence=1.0,
                properties={"relationship_type": "rule_to_standard"}
            ),
            
            # Standards require specific requirements
            Relationship(
                source_id="standard_health_checks",
                target_id="req_health_endpoint",
                type=RelationType.REQUIRES,
                confidence=1.0,
                properties={"relationship_type": "standard_to_requirement"}
            ),
            Relationship(
                source_id="standard_docker_excellence",
                target_id="req_resource_limits",
                type=RelationType.REQUIRES,
                confidence=0.9,
                properties={"relationship_type": "standard_to_requirement"}
            ),
            Relationship(
                source_id="standard_docker_excellence",
                target_id="req_naming_convention",
                type=RelationType.REQUIRES,
                confidence=0.8,
                properties={"relationship_type": "standard_to_requirement"}
            ),
            
            # Agents implement/comply with standards
            Relationship(
                source_id="agent_knowledge_graph_builder",
                target_id="standard_health_checks",
                type=RelationType.COMPLIES_WITH,
                confidence=1.0,
                properties={"compliance_status": "compliant"}
            ),
            Relationship(
                source_id="agent_knowledge_graph_builder",
                target_id="standard_docker_excellence",
                type=RelationType.COMPLIES_WITH,
                confidence=0.9,
                properties={"compliance_status": "mostly_compliant"}
            ),
            Relationship(
                source_id="agent_senior_ai_engineer",
                target_id="standard_health_checks",
                type=RelationType.COMPLIES_WITH,
                confidence=1.0,
                properties={"compliance_status": "compliant"}
            ),
            Relationship(
                source_id="agent_testing_qa_validator",
                target_id="standard_health_checks",
                type=RelationType.COMPLIES_WITH,
                confidence=1.0,
                properties={"compliance_status": "compliant"}
            ),
            
            # Agent dependencies
            Relationship(
                source_id="agent_knowledge_graph_builder",
                target_id="agent_senior_ai_engineer",
                type=RelationType.DEPENDS_ON,
                confidence=0.7,
                properties={"dependency_type": "collaboration"}
            ),
            Relationship(
                source_id="agent_testing_qa_validator",
                target_id="agent_knowledge_graph_builder",
                type=RelationType.VALIDATES,
                confidence=0.8,
                properties={"validation_type": "compliance_testing"}
            )
        ]
        
        # Build the graph manually for demo
        for entity in sample_entities:
            self.kg_builder.entities[entity.id] = entity
            self.kg_builder.graph.add_node(
                entity.id,
                type=entity.type.value,
                name=entity.name,
                description=entity.description,
                **entity.properties
            )
        
        for relationship in sample_relationships:
            self.kg_builder.relationships.append(relationship)
            self.kg_builder.graph.add_edge(
                relationship.source_id,
                relationship.target_id,
                type=relationship.type.value,
                confidence=relationship.confidence,
                **relationship.properties
            )
        
        logger.info(f"✅ Sample knowledge graph created:")
        logger.info(f"   📊 Nodes: {self.kg_builder.graph.number_of_nodes()}")
        logger.info(f"   🔗 Edges: {self.kg_builder.graph.number_of_edges()}")
        logger.info(f"   📦 Entities: {len(self.kg_builder.entities)}")
        logger.info(f"   🔄 Relationships: {len(self.kg_builder.relationships)}")
        
    async def demonstrate_compliance_validation(self):
        """Demonstrate compliance validation capabilities"""
        logger.info("🔍 Demonstrating compliance validation")
        
        print("\n" + "="*60)
        print("🔍 COMPLIANCE VALIDATION DEMONSTRATION")
        print("="*60)
        
        # Test compliance for each agent
        agents = [entity for entity in self.kg_builder.entities.values() if entity.type == EntityType.AGENT]
        
        for agent in agents:
            print(f"\n📋 Validating: {agent.name}")
            print("-" * 40)
            
            # Perform compliance checks
            compliance_checks = await self.kg_builder.validate_compliance(agent.id)
            
            if not compliance_checks:
                print("✅ COMPLIANT - No issues found")
            else:
                for check in compliance_checks:
                    severity_icon = {"BLOCKING": "🚨", "WARNING": "⚠️", "GUIDANCE": "📋"}
                    icon = severity_icon.get(check.severity.value, "ℹ️")
                    print(f"{icon} {check.severity.value}: {check.message}")
                    if check.evidence:
                        print(f"   Evidence: {', '.join(check.evidence)}")
        
        # Generate overall compliance report
        print(f"\n📊 SYSTEM COMPLIANCE SUMMARY")
        print("-" * 40)
        
        report = self.kg_builder.generate_compliance_report()
        print(f"Total Entities: {report['summary']['total_entities']}")
        print(f"Total Agents: {report['summary']['total_agents']}")
        print(f"Total Standards: {report['summary']['total_standards']}")
        
        for category, stats in report['compliance_by_category'].items():
            rate = stats['compliance_rate'] * 100
            print(f"{category}: {stats['compliant']}/{stats['total']} ({rate:.1f}%)")
        
        # Save compliance report
        report_path = self.output_dir / "compliance_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"💾 Compliance report saved to {report_path}")
        
    async def demonstrate_graph_queries(self):
        """Demonstrate graph querying capabilities"""
        logger.info("🔍 Demonstrating graph queries")
        
        print("\n" + "="*60)
        print("🔍 GRAPH QUERY DEMONSTRATIONS")
        print("="*60)
        
        # Sample queries
        queries = [
            {
                "description": "Find all blocking standards",
                "query": "enforcement_level:BLOCKING",
                "type": "simple"
            },
            {
                "description": "Find agents that depend on other agents",
                "query": "depends_on",
                "type": "relationship"
            },
            {
                "description": "Find compliance violations",
                "query": "complies_with",
                "type": "relationship"
            }
        ]
        
        for i, query_info in enumerate(queries, 1):
            print(f"\n{i}. {query_info['description']}")
            print(f"   Query: {query_info['query']}")
            print("   Results:")
            
            if query_info['type'] == 'simple':
                # Simple entity search
                results = []
                for entity in self.kg_builder.entities.values():
                    if 'enforcement_level' in entity.properties:
                        if str(entity.properties['enforcement_level']) == 'EnforcementLevel.BLOCKING':
                            results.append(f"   • {entity.name} ({entity.type.value})")
                
                if results:
                    for result in results:
                        print(result)
                else:
                    print("   • No results found")
            
            elif query_info['type'] == 'relationship':
                # Relationship search
                results = []
                for rel in self.kg_builder.relationships:
                    if query_info['query'] in rel.type.value:
                        source_entity = self.kg_builder.entities.get(rel.source_id)
                        target_entity = self.kg_builder.entities.get(rel.target_id)
                        if source_entity and target_entity:
                            results.append(f"   • {source_entity.name} → {target_entity.name} ({rel.type.value})")
                
                if results:
                    for result in results[:5]:  # Limit to first 5 results
                        print(result)
                    if len(results) > 5:
                        print(f"   • ... and {len(results) - 5} more")
                else:
                    print("   • No results found")
        
        # Save query results
        query_results = {
            "timestamp": datetime.now().isoformat(),
            "queries": queries,
            "graph_stats": {
                "nodes": self.kg_builder.graph.number_of_nodes(),
                "edges": self.kg_builder.graph.number_of_edges()
            }
        }
        
        results_path = self.output_dir / "query_results.json"
        with open(results_path, 'w') as f:
            json.dump(query_results, f, indent=2)
        logger.info(f"💾 Query results saved to {results_path}")
        
    async def demonstrate_visualizations(self):
        """Demonstrate visualization capabilities"""
        logger.info("🎨 Demonstrating visualizations")
        
        print("\n" + "="*60)
        print("🎨 VISUALIZATION DEMONSTRATIONS")
        print("="*60)
        
        viz_outputs = {}
        
        try:
            # 1. Interactive visualization
            print("\n1. Creating interactive visualization...")
            interactive_data = self.visualizer.create_interactive_visualization(
                layout="force",
                color_by="type",
                output_path=str(self.output_dir / "interactive_demo.html")
            )
            viz_outputs["interactive"] = str(self.output_dir / "interactive_demo.html")
            print(f"   ✅ Interactive HTML saved to: {viz_outputs['interactive']}")
            
        except Exception as e:
            print(f"   ❌ Interactive visualization failed: {e}")
        
        try:
            # 2. Static diagram
            print("\n2. Creating static diagram...")
            static_path = self.visualizer.create_static_diagram(
                layout="hierarchical",
                color_by="enforcement",
                output_path=str(self.output_dir / "static_demo.png"),
                figsize=(12, 8)
            )
            viz_outputs["static"] = static_path
            print(f"   ✅ Static diagram saved to: {static_path}")
            
        except Exception as e:
            print(f"   ❌ Static diagram failed: {e}")
        
        try:
            # 3. Compliance heatmap
            print("\n3. Creating compliance heatmap...")
            heatmap_path = self.visualizer.create_compliance_heatmap(
                output_path=str(self.output_dir / "compliance_heatmap_demo.png")
            )
            if heatmap_path:
                viz_outputs["heatmap"] = heatmap_path
                print(f"   ✅ Compliance heatmap saved to: {heatmap_path}")
            else:
                print("   ⚠️ Compliance heatmap skipped (insufficient data)")
                
        except Exception as e:
            print(f"   ❌ Compliance heatmap failed: {e}")
        
        try:
            # 4. Hierarchy diagram
            print("\n4. Creating hierarchy diagram...")
            hierarchy_path = self.visualizer.create_hierarchy_diagram(
                output_path=str(self.output_dir / "hierarchy_demo.png")
            )
            if hierarchy_path:
                viz_outputs["hierarchy"] = hierarchy_path
                print(f"   ✅ Hierarchy diagram saved to: {hierarchy_path}")
            else:
                print("   ⚠️ Hierarchy diagram skipped (insufficient hierarchical data)")
                
        except Exception as e:
            print(f"   ❌ Hierarchy diagram failed: {e}")
        
        # Save visualization summary
        viz_summary = {
            "timestamp": datetime.now().isoformat(),
            "visualizations_created": viz_outputs,
            "graph_stats": {
                "nodes": self.kg_builder.graph.number_of_nodes(),
                "edges": self.kg_builder.graph.number_of_edges(),
                "entity_types": list(set(e.type.value for e in self.kg_builder.entities.values())),
                "relationship_types": list(set(r.type.value for r in self.kg_builder.relationships))
            }
        }
        
        summary_path = self.output_dir / "visualization_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(viz_summary, f, indent=2)
        logger.info(f"💾 Visualization summary saved to {summary_path}")
        
        return viz_outputs
        
    async def demonstrate_export_capabilities(self):
        """Demonstrate export capabilities"""
        logger.info("📦 Demonstrating export capabilities")
        
        print("\n" + "="*60)
        print("📦 EXPORT DEMONSTRATIONS")
        print("="*60)
        
        export_formats = ["json", "graphml", "rdf"]
        exported_files = {}
        
        for format_type in export_formats:
            try:
                print(f"\n📄 Exporting to {format_type.upper()} format...")
                output_path = str(self.output_dir / f"knowledge_graph_demo.{format_type}")
                
                self.kg_builder.export_graph(format_type, output_path)
                exported_files[format_type] = output_path
                
                # Get file size
                file_size = os.path.getsize(output_path)
                print(f"   ✅ Export successful: {output_path} ({file_size:,} bytes)")
                
            except Exception as e:
                print(f"   ❌ Export to {format_type} failed: {e}")
        
        # Create export summary
        export_summary = {
            "timestamp": datetime.now().isoformat(),
            "exported_files": exported_files,
            "formats_supported": ["json", "graphml", "gexf", "rdf", "turtle"],
            "graph_content": {
                "nodes": self.kg_builder.graph.number_of_nodes(),
                "edges": self.kg_builder.graph.number_of_edges(),
                "entities": len(self.kg_builder.entities),
                "relationships": len(self.kg_builder.relationships)
            }
        }
        
        summary_path = self.output_dir / "export_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(export_summary, f, indent=2)
        logger.info(f"💾 Export summary saved to {summary_path}")
        
        return exported_files
        
    async def demonstrate_cli_capabilities(self):
        """Demonstrate CLI capabilities"""
        logger.info("💻 Demonstrating CLI capabilities")
        
        print("\n" + "="*60)
        print("💻 CLI DEMONSTRATION")
        print("="*60)
        
        # Create a simple CLI demo script
        cli_demo_script = f"""#!/bin/bash
# SutazAI Knowledge Graph Builder CLI Demo Script

echo "🚀 SutazAI Knowledge Graph Builder CLI Demo"
echo "==========================================="
echo

# Set the output directory
OUTPUT_DIR="{self.output_dir}"

echo "📊 1. Show knowledge graph status"
python cli.py status
echo

echo "🔍 2. Validate agent compliance"
python cli.py validate --entity agent_knowledge_graph_builder
echo

echo "📋 3. Generate compliance report"
python cli.py report --format json --output "$OUTPUT_DIR/cli_demo_report.json"
echo

echo "🎨 4. Create visualization"
python cli.py visualize --type static --output "$OUTPUT_DIR/cli_demo_viz"
echo

echo "📦 5. Export knowledge graph"
python cli.py export --format json --output "$OUTPUT_DIR/cli_demo_export.json"
echo

echo "✅ CLI demonstration completed!"
echo "📁 Output files saved to: $OUTPUT_DIR"
"""
        
        script_path = self.output_dir / "cli_demo.sh"
        with open(script_path, 'w') as f:
            f.write(cli_demo_script)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        print(f"📝 CLI demo script created: {script_path}")
        print(f"   Run with: bash {script_path}")
        
        # Show available CLI commands
        print(f"\n💻 Available CLI Commands:")
        commands = [
            "build --comprehensive",
            "query 'MATCH (n:Agent) RETURN n.name'",
            "validate --entity agent_name",
            "report --format json",
            "visualize --type interactive",
            "export --format graphml",
            "serve --port 8048",
            "status"
        ]
        
        for cmd in commands:
            print(f"   • python cli.py {cmd}")
        
        return str(script_path)
        
    async def run_full_demonstration(self):
        """Run the complete demonstration"""
        try:
            start_time = datetime.now()
            
            print("\n" + "🌟"*20)
            print("🌟 SUTAZAI KNOWLEDGE GRAPH BUILDER DEMO 🌟")
            print("🌟"*20)
            print(f"\n📅 Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📁 Output Directory: {self.output_dir}")
            
            # Initialize
            await self.initialize()
            
            # Create sample knowledge graph
            await self.create_sample_knowledge_graph()
            
            # Demonstrate capabilities
            await self.demonstrate_compliance_validation()
            await self.demonstrate_graph_queries()
            viz_files = await self.demonstrate_visualizations()
            export_files = await self.demonstrate_export_capabilities()
            cli_script = await self.demonstrate_cli_capabilities()
            
            # Create final summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            demo_summary = {
                "demo_info": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_seconds": duration.total_seconds(),
                    "output_directory": str(self.output_dir)
                },
                "knowledge_graph_stats": {
                    "nodes": self.kg_builder.graph.number_of_nodes(),
                    "edges": self.kg_builder.graph.number_of_edges(),
                    "entities": len(self.kg_builder.entities),
                    "relationships": len(self.kg_builder.relationships),
                    "entity_types": list(set(e.type.value for e in self.kg_builder.entities.values())),
                    "relationship_types": list(set(r.type.value for r in self.kg_builder.relationships))
                },
                "files_created": {
                    "visualizations": viz_files,
                    "exports": export_files,
                    "cli_script": cli_script,
                    "compliance_report": str(self.output_dir / "compliance_report.json"),
                    "query_results": str(self.output_dir / "query_results.json")
                },
                "capabilities_demonstrated": [
                    "Knowledge graph construction",
                    "Compliance validation",
                    "Graph querying",
                    "Interactive visualizations",
                    "Static diagram generation",
                    "Compliance heatmaps",
                    "Multiple export formats",
                    "CLI interface",
                    "Integration capabilities"
                ]
            }
            
            summary_path = self.output_dir / "demo_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(demo_summary, f, indent=2)
            
            # Final output
            print("\n" + "🎉"*20)
            print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY! 🎉")
            print("🎉"*20)
            print(f"\n⏱️  Duration: {duration.total_seconds():.1f} seconds")
            print(f"📊 Knowledge Graph: {self.kg_builder.graph.number_of_nodes()} nodes, {self.kg_builder.graph.number_of_edges()} edges")
            print(f"📁 Output Directory: {self.output_dir}")
            print(f"📋 Demo Summary: {summary_path}")
            
            print(f"\n📂 Files Created:")
            for category, files in demo_summary["files_created"].items():
                print(f"   {category}:")
                if isinstance(files, dict):
                    for file_type, file_path in files.items():
                        print(f"     • {file_type}: {file_path}")
                else:
                    print(f"     • {files}")
            
            print(f"\n🚀 Next Steps:")
            print(f"   1. View interactive visualization: open {self.output_dir}/interactive_demo.html")
            print(f"   2. Run CLI demo: bash {cli_script}")
            print(f"   3. Start API server: python api.py")
            print(f"   4. Explore exported data in: {self.output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            import traceback
            traceback.print_exc()
            return False

async def main():
    """Main entry point for the demonstration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SutazAI Knowledge Graph Builder Demo")
    parser.add_argument("--output-dir", default="./demo_output", help="Output directory for demo files")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Run the demonstration
    demo = KnowledgeGraphDemo(output_dir=args.output_dir)
    success = await demo.run_full_demonstration()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)