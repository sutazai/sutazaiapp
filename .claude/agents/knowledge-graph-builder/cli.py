#!/usr/bin/env python3
"""
SutazAI Knowledge Graph Builder CLI

Command-line interface for building, querying, and managing the SutazAI
codebase hygiene standards knowledge graph.
"""

import os
import sys
import json
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

from knowledge_graph_builder import KnowledgeGraphBuilder
from visualization import KnowledgeGraphVisualizer
from integration import create_sutazai_integration
from api import run_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KnowledgeGraphCLI:
    """Command-line interface for Knowledge Graph Builder"""
    
    def __init__(self):
        self.kg_builder = None
        self.visualizer = None
        self.integration = None
        
    async def initialize(self, config_path: Optional[str] = None):
        """Initialize the knowledge graph builder"""
        if not config_path:
            config_path = Path(__file__).parent / "knowledge_graph_config.yaml"
        
        self.kg_builder = KnowledgeGraphBuilder(config_path=str(config_path))
        self.visualizer = KnowledgeGraphVisualizer(self.kg_builder)
        logger.info("Knowledge Graph Builder initialized")
    
    async def cmd_build(self, args):
        """Build knowledge graph from sources"""
        logger.info("Building knowledge graph from sources")
        
        sources = {}
        
        if args.comprehensive:
            # Build from all available sources
            default_sources = {
                'claude_md': '/opt/sutazaiapp/CLAUDE.md',
                'docker_configs': [
                    '/opt/sutazaiapp/docker-compose.complete-agents.yml',
                    '/opt/sutazaiapp/docker-compose.agents-simple.yml'
                ]
            }
            
            for source_type, source_paths in default_sources.items():
                if source_type == 'claude_md':
                    if os.path.exists(source_paths):
                        sources[source_type] = source_paths
                        logger.info(f"Added source: {source_paths}")
                    else:
                        logger.warning(f"Source file not found: {source_paths}")
                else:
                    existing_paths = [p for p in source_paths if os.path.exists(p)]
                    if existing_paths:
                        sources[source_type] = existing_paths
                        logger.info(f"Added sources: {existing_paths}")
                    else:
                        logger.warning(f"No Docker config files found in: {source_paths}")
        
        elif args.source:
            # Build from specified source
            if args.source.endswith('.md'):
                sources['claude_md'] = args.source
            elif 'docker-compose' in args.source or args.source.endswith('.yml'):
                sources['docker_configs'] = [args.source]
            else:
                logger.error(f"Unknown source type: {args.source}")
                return False
        
        else:
            logger.error("No sources specified. Use --comprehensive or --source")
            return False
        
        if not sources:
            logger.error("No valid sources found")
            return False
        
        # Build the graph
        try:
            graph = await self.kg_builder.build_comprehensive_graph(sources)
            
            success_msg = f"""
✅ Knowledge Graph Built Successfully!
   📊 Nodes: {graph.number_of_nodes()}
   🔗 Edges: {graph.number_of_edges()}
   📦 Entities: {len(self.kg_builder.entities)}
   🔄 Relationships: {len(self.kg_builder.relationships)}
   
   Sources processed:
"""
            for source_type, source_paths in sources.items():
                if isinstance(source_paths, list):
                    for path in source_paths:
                        success_msg += f"   • {path}\n"
                else:
                    success_msg += f"   • {source_paths}\n"
            
            print(success_msg)
            
            # Save build info
            if args.output:
                build_info = {
                    "timestamp": graph.graph.get("build_timestamp", "unknown"),
                    "sources": sources,
                    "stats": {
                        "nodes": graph.number_of_nodes(),
                        "edges": graph.number_of_edges(),
                        "entities": len(self.kg_builder.entities),
                        "relationships": len(self.kg_builder.relationships)
                    }
                }
                
                with open(args.output, 'w') as f:
                    json.dump(build_info, f, indent=2)
                logger.info(f"Build info saved to {args.output}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build knowledge graph: {e}")
            return False
    
    async def cmd_query(self, args):
        """Execute queries against the knowledge graph"""
        if not self.kg_builder.graph.number_of_nodes():
            logger.error("Knowledge graph is empty. Build it first with 'build' command.")
            return False
        
        try:
            results = self.kg_builder.query_graph(args.query, args.query_type)
            
            print(f"\n🔍 Query Results ({len(results)} found):")
            print(f"Query: {args.query}")
            print(f"Type: {args.query_type}")
            print("-" * 50)
            
            if args.format == "json":
                print(json.dumps(results, indent=2))
            elif args.format == "table":
                self._print_results_table(results)
            else:
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result}")
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to {args.output}")
            
            return True
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return False
    
    async def cmd_validate(self, args):
        """Validate compliance for entities or standards"""
        if not self.kg_builder.graph.number_of_nodes():
            logger.error("Knowledge graph is empty. Build it first with 'build' command.")
            return False
        
        try:
            if args.entity:
                # Validate specific entity
                checks = await self.kg_builder.validate_compliance(args.entity)
                entity_name = self.kg_builder.entities.get(args.entity, {}).name if args.entity in self.kg_builder.entities else args.entity
                
                print(f"\n🔍 Compliance Validation for: {entity_name}")
                print(f"Entity ID: {args.entity}")
                print("-" * 60)
                
                if not checks:
                    print("✅ No compliance issues found!")
                    return True
                
                # Group by severity
                blocking = [c for c in checks if c.severity.value == "BLOCKING"]
                warnings = [c for c in checks if c.severity.value == "WARNING"]
                guidance = [c for c in checks if c.severity.value == "GUIDANCE"]
                
                if blocking:
                    print(f"\n🚨 BLOCKING VIOLATIONS ({len(blocking)}):")
                    for i, check in enumerate(blocking, 1):
                        print(f"  {i}. {check.message}")
                        print(f"     Standard: {check.standard_id}")
                        if check.evidence:
                            print(f"     Evidence: {', '.join(check.evidence)}")
                        print()
                
                if warnings:
                    print(f"\n⚠️  WARNINGS ({len(warnings)}):")
                    for i, check in enumerate(warnings, 1):
                        print(f"  {i}. {check.message}")
                        print(f"     Standard: {check.standard_id}")
                        if check.evidence:
                            print(f"     Evidence: {', '.join(check.evidence)}")
                        print()
                
                if guidance:
                    print(f"\n📋 GUIDANCE ({len(guidance)}):")
                    for i, check in enumerate(guidance, 1):
                        print(f"  {i}. {check.message}")
                        print(f"     Standard: {check.standard_id}")
                        print()
                
                # Summary
                print(f"\n📊 Summary:")
                print(f"   🚨 Blocking: {len(blocking)}")
                print(f"   ⚠️  Warnings: {len(warnings)}")
                print(f"   📋 Guidance: {len(guidance)}")
                print(f"   📝 Total Issues: {len(checks)}")
                
                if blocking:
                    print(f"\n❌ DEPLOYMENT BLOCKED - Fix blocking violations first!")
                    return False
                elif warnings:
                    print(f"\n⚠️  REVIEW REQUIRED - Address warnings before deployment")
                else:
                    print(f"\n✅ COMPLIANT - No blocking issues found")
                
            else:
                # Validate all entities
                print("\n🔍 System-wide Compliance Validation")
                print("-" * 50)
                
                total_entities = 0
                total_issues = 0
                blocking_entities = 0
                warning_entities = 0
                
                for entity_id in self.kg_builder.entities.keys():
                    if self.kg_builder.entities[entity_id].type.value == "Agent":
                        total_entities += 1
                        checks = await self.kg_builder.validate_compliance(entity_id)
                        
                        if checks:
                            total_issues += len(checks)
                            has_blocking = any(c.severity.value == "BLOCKING" for c in checks)
                            has_warnings = any(c.severity.value == "WARNING" for c in checks)
                            
                            if has_blocking:
                                blocking_entities += 1
                                print(f"🚨 {self.kg_builder.entities[entity_id].name}: {len([c for c in checks if c.severity.value == 'BLOCKING'])} blocking violations")
                            elif has_warnings:
                                warning_entities += 1
                                print(f"⚠️  {self.kg_builder.entities[entity_id].name}: {len([c for c in checks if c.severity.value == 'WARNING'])} warnings")
                            else:
                                print(f"✅ {self.kg_builder.entities[entity_id].name}: compliant")
                        else:
                            print(f"✅ {self.kg_builder.entities[entity_id].name}: compliant")
                
                # System summary
                compliant_entities = total_entities - blocking_entities - warning_entities
                compliance_rate = (compliant_entities / total_entities * 100) if total_entities > 0 else 0
                
                print(f"\n📊 System Compliance Summary:")
                print(f"   📦 Total Agents: {total_entities}")
                print(f"   ✅ Compliant: {compliant_entities}")
                print(f"   ⚠️  Warnings: {warning_entities}")
                print(f"   🚨 Violations: {blocking_entities}")
                print(f"   📈 Compliance Rate: {compliance_rate:.1f}%")
                print(f"   📝 Total Issues: {total_issues}")
                
                if blocking_entities > 0:
                    print(f"\n❌ SYSTEM DEPLOYMENT BLOCKED")
                    print(f"   {blocking_entities} agents have blocking violations")
                    return False
                elif warning_entities > 0:
                    print(f"\n⚠️  SYSTEM REVIEW REQUIRED")
                    print(f"   {warning_entities} agents have warnings")
                else:
                    print(f"\n✅ SYSTEM COMPLIANT")
                    print(f"   All agents meet hygiene standards")
            
            # Save results if requested
            if args.output:
                if args.entity:
                    results = [
                        {
                            "entity_id": check.entity_id,
                            "standard_id": check.standard_id,
                            "status": check.status,
                            "message": check.message,
                            "severity": check.severity.value,
                            "timestamp": check.timestamp.isoformat(),
                            "evidence": check.evidence
                        }
                        for check in checks
                    ]
                else:
                    # System-wide validation results
                    results = {"system_compliance": "Implementation needed"}
                
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Validation results saved to {args.output}")
            
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    async def cmd_report(self, args):
        """Generate comprehensive compliance report"""
        if not self.kg_builder.graph.number_of_nodes():
            logger.error("Knowledge graph is empty. Build it first with 'build' command.")
            return False
        
        try:
            report = self.kg_builder.generate_compliance_report(format=args.format)
            
            if args.format == "json":
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(report, f, indent=2)
                    print(f"📄 Compliance report saved to {args.output}")
                else:
                    print(json.dumps(report, indent=2))
            else:
                # Print formatted report
                print("\n📊 SutazAI Compliance Report")
                print("=" * 50)
                print(f"Generated: {report['timestamp']}")
                print(f"Total Entities: {report['summary']['total_entities']}")
                print(f"Total Standards: {report['summary']['total_standards']}")
                print(f"Total Agents: {report['summary']['total_agents']}")
                
                print(f"\n📈 Compliance by Category:")
                for category, stats in report['compliance_by_category'].items():
                    rate = stats['compliance_rate'] * 100
                    print(f"   {category}: {stats['compliant']}/{stats['total']} ({rate:.1f}%)")
                
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(report, f, indent=2)
                    print(f"\n📄 Detailed report saved to {args.output}")
            
            return True
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return False
    
    async def cmd_visualize(self, args):
        """Generate visualizations"""
        if not self.kg_builder.graph.number_of_nodes():
            logger.error("Knowledge graph is empty. Build it first with 'build' command.")
            return False
        
        try:
            output_dir = Path(args.output) if args.output else Path("./kg_visualizations")
            output_dir.mkdir(exist_ok=True)
            
            if args.type == "interactive":
                result = self.visualizer.create_interactive_visualization(
                    layout=args.layout,
                    color_by=args.color_by,
                    output_path=str(output_dir / "interactive.html")
                )
                print(f"🌐 Interactive visualization created: {output_dir / 'interactive.html'}")
                
            elif args.type == "static":
                result = self.visualizer.create_static_diagram(
                    layout=args.layout,
                    color_by=args.color_by,
                    output_path=str(output_dir / "static.png")
                )
                print(f"🖼️  Static diagram created: {result}")
                
            elif args.type == "compliance":
                result = self.visualizer.create_compliance_heatmap(
                    output_path=str(output_dir / "compliance_heatmap.png")
                )
                print(f"🔥 Compliance heatmap created: {result}")
                
            elif args.type == "hierarchy":
                result = self.visualizer.create_hierarchy_diagram(
                    output_path=str(output_dir / "hierarchy.png")
                )
                print(f"🌳 Hierarchy diagram created: {result}")
                
            elif args.type == "dashboard":
                result = self.visualizer.create_dashboard_summary(
                    output_path=str(output_dir / "dashboard.png")
                )
                print(f"📊 Dashboard created: {result}")
                
            elif args.type == "web":
                files = self.visualizer.export_for_web(str(output_dir))
                print(f"🌐 Web export completed:")
                for file_type, file_path in files.items():
                    print(f"   {file_type}: {file_path}")
                
            else:
                # Create all visualizations
                print("🎨 Creating all visualizations...")
                
                # Interactive
                self.visualizer.create_interactive_visualization(
                    output_path=str(output_dir / "interactive.html")
                )
                
                # Static overview
                self.visualizer.create_static_diagram(
                    output_path=str(output_dir / "overview.png")
                )
                
                # Compliance heatmap
                self.visualizer.create_compliance_heatmap(
                    output_path=str(output_dir / "compliance.png")
                )
                
                # Hierarchy
                self.visualizer.create_hierarchy_diagram(
                    output_path=str(output_dir / "hierarchy.png")
                )
                
                # Dashboard
                self.visualizer.create_dashboard_summary(
                    output_path=str(output_dir / "dashboard.png")
                )
                
                print(f"✅ All visualizations created in: {output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return False
    
    async def cmd_export(self, args):
        """Export knowledge graph in various formats"""
        if not self.kg_builder.graph.number_of_nodes():
            logger.error("Knowledge graph is empty. Build it first with 'build' command.")
            return False
        
        try:
            output_path = args.output or f"knowledge_graph.{args.format}"
            
            self.kg_builder.export_graph(args.format, output_path)
            
            print(f"📦 Knowledge graph exported to: {output_path}")
            print(f"   Format: {args.format}")
            print(f"   Nodes: {self.kg_builder.graph.number_of_nodes()}")
            print(f"   Edges: {self.kg_builder.graph.number_of_edges()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    async def cmd_serve(self, args):
        """Start the API server"""
        logger.info(f"Starting Knowledge Graph API server on {args.host}:{args.port}")
        
        # Initialize if not done already
        if not self.kg_builder:
            await self.initialize()
        
        # Create integration if requested
        if args.integrate:
            logger.info("Setting up SutazAI integration")
            self.integration = await create_sutazai_integration(self.kg_builder)
        
        # Run the API server
        run_api(host=args.host, port=args.port, debug=args.debug)
        
        return True
    
    async def cmd_status(self, args):
        """Show knowledge graph status"""
        if not self.kg_builder:
            print("❌ Knowledge Graph Builder not initialized")
            return False
        
        print("\n📊 Knowledge Graph Status")
        print("=" * 40)
        
        # Graph stats
        print(f"📦 Nodes: {self.kg_builder.graph.number_of_nodes()}")
        print(f"🔗 Edges: {self.kg_builder.graph.number_of_edges()}")
        print(f"📋 Entities: {len(self.kg_builder.entities)}")
        print(f"🔄 Relationships: {len(self.kg_builder.relationships)}")
        
        # Entity breakdown
        if self.kg_builder.entities:
            print(f"\n📊 Entity Breakdown:")
            entity_counts = {}
            for entity in self.kg_builder.entities.values():
                entity_type = entity.type.value
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            
            for entity_type, count in sorted(entity_counts.items()):
                print(f"   {entity_type}: {count}")
        
        # Database status
        if self.kg_builder.neo4j_driver:
            try:
                with self.kg_builder.neo4j_driver.session() as session:
                    result = session.run("CALL db.labels()")
                    labels = [record["label"] for record in result]
                    print(f"\n🗄️  Neo4j Status: ✅ Connected ({len(labels)} node types)")
            except Exception as e:
                print(f"\n🗄️  Neo4j Status: ❌ Error - {e}")
        else:
            print(f"\n🗄️  Neo4j Status: ❌ Not connected")
        
        # Graph metrics
        if self.kg_builder.graph.number_of_nodes() > 0:
            import networkx as nx
            try:
                density = nx.density(self.kg_builder.graph)
                print(f"\n📈 Graph Metrics:")
                print(f"   Density: {density:.4f}")
                
                if self.kg_builder.graph.number_of_nodes() < 1000:  # Only for smaller graphs
                    avg_clustering = nx.average_clustering(self.kg_builder.graph)
                    print(f"   Avg Clustering: {avg_clustering:.4f}")
                
            except Exception as e:
                logger.debug(f"Could not calculate graph metrics: {e}")
        
        return True
    
    def _print_results_table(self, results):
        """Print results in table format"""
        if not results:
            print("No results found.")
            return
        
        # Simple table printing - in a real implementation you'd use a library like tabulate
        for i, result in enumerate(results, 1):
            print(f"{i}. {result}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="SutazAI Knowledge Graph Builder CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build knowledge graph from all sources
  %(prog)s build --comprehensive
  
  # Build from specific source
  %(prog)s build --source /path/to/CLAUDE.md
  
  # Query the graph
  %(prog)s query "MATCH (n:Agent) RETURN n.name LIMIT 10" --query-type cypher
  
  # Validate agent compliance
  %(prog)s validate --entity agent_knowledge_graph_builder
  
  # Generate compliance report
  %(prog)s report --format json --output compliance_report.json
  
  # Create visualizations
  %(prog)s visualize --type web --output ./web_export
  
  # Start API server
  %(prog)s serve --host 0.0.0.0 --port 8048 --integrate
  
  # Export graph
  %(prog)s export --format graphml --output knowledge_graph.graphml
        """
    )
    
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build knowledge graph from sources")
    build_parser.add_argument("--source", type=str, help="Source file or directory")
    build_parser.add_argument("--comprehensive", action="store_true", help="Build from all available sources")
    build_parser.add_argument("--output", type=str, help="Save build info to file")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Execute queries against the knowledge graph")
    query_parser.add_argument("query", type=str, help="Query text")
    query_parser.add_argument("--query-type", type=str, default="cypher", 
                             choices=["cypher", "sparql", "networkx"], help="Query type")
    query_parser.add_argument("--format", type=str, default="json", 
                             choices=["json", "table", "raw"], help="Output format")
    query_parser.add_argument("--output", type=str, help="Save results to file")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate compliance")
    validate_parser.add_argument("--entity", type=str, help="Entity ID to validate")
    validate_parser.add_argument("--standard", type=str, help="Standard ID to check")
    validate_parser.add_argument("--output", type=str, help="Save results to file")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate compliance report")
    report_parser.add_argument("--format", type=str, default="json", 
                              choices=["json", "yaml", "text"], help="Report format")
    report_parser.add_argument("--output", type=str, help="Save report to file")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    viz_parser.add_argument("--type", type=str, default="all",
                           choices=["interactive", "static", "compliance", "hierarchy", "dashboard", "web", "all"],
                           help="Visualization type")
    viz_parser.add_argument("--layout", type=str, default="force",
                           choices=["force", "hierarchical", "circular", "tree"],
                           help="Graph layout algorithm")
    viz_parser.add_argument("--color-by", type=str, default="type",
                           choices=["type", "enforcement", "category"],
                           help="Node coloring scheme")
    viz_parser.add_argument("--output", type=str, help="Output directory")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export knowledge graph")
    export_parser.add_argument("--format", type=str, default="json",
                              choices=["json", "graphml", "gexf", "rdf", "turtle"],
                              help="Export format")
    export_parser.add_argument("--output", type=str, help="Output file path")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8048, help="Port to bind to")
    serve_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    serve_parser.add_argument("--integrate", action="store_true", help="Enable SutazAI integration")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show knowledge graph status")
    
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create CLI instance
    cli = KnowledgeGraphCLI()
    
    async def run_command():
        try:
            # Initialize for most commands (except serve which handles its own initialization)
            if args.command != "serve":
                await cli.initialize(args.config)
            
            # Execute command
            if args.command == "build":
                return await cli.cmd_build(args)
            elif args.command == "query":
                return await cli.cmd_query(args)
            elif args.command == "validate":
                return await cli.cmd_validate(args)
            elif args.command == "report":
                return await cli.cmd_report(args)
            elif args.command == "visualize":
                return await cli.cmd_visualize(args)
            elif args.command == "export":
                return await cli.cmd_export(args)
            elif args.command == "serve":
                return await cli.cmd_serve(args)
            elif args.command == "status":
                return await cli.cmd_status(args)
            else:
                parser.print_help()
                return False
                
        except KeyboardInterrupt:
            logger.info("Operation cancelled by user")
            return False
        except Exception as e:
            logger.error(f"Command failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    # Run the command
    success = asyncio.run(run_command())
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()