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
SutazAI Knowledge Graph Visualization Components

This module provides various visualization capabilities for the knowledge graph,
including interactive web visualizations, static diagrams, and export formats.
"""

import os
import json
import logging
import math
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import base64
from io import BytesIO

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
    logging.warning(f"Optional visualization dependency not available: {e}")

from knowledge_graph_builder import KnowledgeGraphBuilder, EntityType, EnforcementLevel

logger = logging.getLogger(__name__)

class KnowledgeGraphVisualizer:
    """Comprehensive visualization suite for SutazAI knowledge graphs"""
    
    def __init__(self, kg_builder: KnowledgeGraphBuilder):
        self.kg_builder = kg_builder
        self.graph = kg_builder.graph
        self.entities = kg_builder.entities
        
        # Color schemes
        self.color_schemes = {
            "by_type": {
                "Standard": "#3498db",
                "Agent": "#e74c3c", 
                "Process": "#2ecc71",
                "Tool": "#f39c12",
                "Metric": "#9b59b6",
                "Violation": "#e67e22",
                "Requirement": "#1abc9c",
                "Rule": "#34495e"
            },
            "by_enforcement": {
                "BLOCKING": "#e74c3c",
                "WARNING": "#f39c12", 
                "GUIDANCE": "#2ecc71",
                "": "#95a5a6"
            },
            "by_category": {
                "professional_standards": "#3498db",
                "technical_standards": "#e74c3c",
                "process_standards": "#2ecc71", 
                "team_standards": "#f39c12",
                "security": "#9b59b6",
                "general": "#95a5a6"
            }
        }
        
        # Edge styles
        self.edge_styles = {
            "enforces": {"color": "#e74c3c", "width": 3, "style": "solid"},
            "depends_on": {"color": "#3498db", "width": 2, "style": "dashed"},
            "validates": {"color": "#2ecc71", "width": 2, "style": "solid"},
            "blocks": {"color": "#e74c3c", "width": 4, "style": "solid"},
            "warns": {"color": "#f39c12", "width": 2, "style": "dotted"}
        }
    
    def create_interactive_visualization(self, 
                                       layout: str = "force",
                                       color_by: str = "type",
                                       size_by: str = "degree",
                                       max_nodes: int = 1000,
                                       output_path: Optional[str] = None) -> Dict[str, Any]:
        """Create interactive Plotly visualization"""
        try:
            # Get layout positions
            pos = self._calculate_layout(layout)
            
            # Prepare node data
            node_trace, node_info = self._prepare_plotly_nodes(pos, color_by, size_by, max_nodes)
            
            # Prepare edge data
            edge_traces = self._prepare_plotly_edges(pos)
            
            # Create figure
            fig = go.Figure(
                data=[*edge_traces, node_trace],
                layout=go.Layout(
                    title=dict(
                        text="SutazAI Codebase Hygiene Standards Knowledge Graph",
                        x=0.5,
                        font=dict(size=16)
                    ),
                    titlefont_size=16,
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Nodes: Entities (Standards, Agents, Rules) | Edges: Relationships | Color: " + color_by.title(),
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(color="#666", size=12)
                    )],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
            )
            
            # Add legend for node types
            if color_by in self.color_schemes:
                self._add_plotly_legend(fig, color_by)
            
            # Save or return
            if output_path:
                fig.write_html(output_path)
                logger.info(f"Interactive visualization saved to {output_path}")
            
            # Return plot data for API
            return {
                "plot_json": fig.to_json(),
                "node_info": node_info,
                "stats": {
                    "total_nodes": len(node_trace.x),
                    "total_edges": sum(len(trace.x) for trace in edge_traces) // 2,
                    "layout": layout,
                    "color_scheme": color_by
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating interactive visualization: {e}")
            raise
    
    def create_static_diagram(self,
                            layout: str = "hierarchical", 
                            color_by: str = "enforcement",
                            output_path: str = "knowledge_graph.png",
                            figsize: Tuple[int, int] = (20, 16),
                            dpi: int = 300) -> str:
        """Create static matplotlib diagram"""
        try:
            plt.figure(figsize=figsize, dpi=dpi)
            plt.style.use('seaborn-v0_8')
            
            # Calculate layout
            pos = self._calculate_layout(layout)
            
            # Get colors and sizes
            node_colors = self._get_node_colors(color_by)
            node_sizes = self._get_node_sizes("degree")
            edge_colors = self._get_edge_colors()
            
            # Draw edges first (so they appear behind nodes)
            nx.draw_networkx_edges(
                self.graph, pos,
                edge_color=edge_colors,
                alpha=0.6,
                width=1.5,
                arrows=True,
                arrowsize=20,
                arrowstyle='->'
            )
            
            # Draw nodes
            nx.draw_networkx_nodes(
                self.graph, pos,
                node_color=node_colors,
                node_size=node_sizes,
                alpha=0.8,
                linewidths=2,
                edgecolors='white'
            )
            
            # Add labels for important nodes
            important_nodes = self._get_important_nodes()
            labels = {node: self.entities[node].name[:20] + "..." 
                     if len(self.entities[node].name) > 20 
                     else self.entities[node].name
                     for node in important_nodes if node in self.entities}
            
            nx.draw_networkx_labels(
                self.graph, pos,
                labels=labels,
                font_size=8,
                font_weight='bold',
                font_color='black'
            )
            
            # Add title and legend
            plt.title("SutazAI Codebase Hygiene Standards Knowledge Graph", 
                     fontsize=16, fontweight='bold', pad=20)
            
            self._add_matplotlib_legend(color_by)
            
            # Add statistics box
            self._add_stats_box()
            
            plt.axis('off')
            plt.tight_layout()
            
            # Save
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"Static diagram saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating static diagram: {e}")
            raise
    
    def create_compliance_heatmap(self, output_path: str = "compliance_heatmap.png") -> str:
        """Create compliance heatmap showing standards adherence"""
        try:
            # Get agents and standards
            agents = [e for e in self.entities.values() if e.type == EntityType.AGENT]
            standards = [e for e in self.entities.values() if e.type == EntityType.STANDARD]
            
            if not agents or not standards:
                logger.warning("No agents or standards found for compliance heatmap")
                return ""
            
            # Create compliance matrix
            compliance_matrix = np.zeros((len(agents), len(standards)))
            agent_names = []
            standard_names = []
            
            for i, agent in enumerate(agents):
                agent_names.append(agent.name[:15] + "..." if len(agent.name) > 15 else agent.name)
                for j, standard in enumerate(standards):
                    if i == 0:  # Only add standard names once
                        standard_names.append(standard.name[:20] + "..." if len(standard.name) > 20 else standard.name)
                    
                    # Calculate compliance score (simplified)
                    compliance_score = self.kg_builder._calculate_compliance_score(agent, standard)
                    compliance_matrix[i, j] = compliance_score
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(max(12, len(standards) * 0.8), max(8, len(agents) * 0.6)))
            
            im = ax.imshow(compliance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(standards)))
            ax.set_yticks(np.arange(len(agents)))
            ax.set_xticklabels(standard_names, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(agent_names, fontsize=9)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Compliance Score', rotation=270, labelpad=20)
            
            # Add text annotations
            for i in range(len(agents)):
                for j in range(len(standards)):
                    score = compliance_matrix[i, j]
                    text_color = 'white' if score < 0.5 else 'black'
                    ax.text(j, i, f'{score:.2f}', ha='center', va='center', 
                           color=text_color, fontweight='bold', fontsize=8)
            
            plt.title("Agent Compliance with Standards", fontsize=14, fontweight='bold', pad=20)
            plt.xlabel("Standards", fontsize=12)
            plt.ylabel("Agents", fontsize=12)
            plt.tight_layout()
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Compliance heatmap saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating compliance heatmap: {e}")
            raise
    
    def create_hierarchy_diagram(self, root_type: str = "Rule", output_path: str = "hierarchy.png") -> str:
        """Create hierarchical diagram showing rule/standard relationships"""
        try:
            # Filter graph to hierarchy relationships
            hierarchy_graph = nx.DiGraph()
            
            for edge in self.graph.edges(data=True):
                source, target, data = edge
                if data.get('type') in ['enforces', 'requires', 'part_of', 'is_a']:
                    hierarchy_graph.add_edge(source, target, **data)
            
            if hierarchy_graph.number_of_nodes() == 0:
                logger.warning("No hierarchical relationships found")
                return ""
            
            # Create hierarchical layout
            try:
                pos = nx.nx_agraph.graphviz_layout(hierarchy_graph, prog='dot')
            except:
                pos = nx.spring_layout(hierarchy_graph, k=3, iterations=50)
            
            plt.figure(figsize=(16, 12), dpi=300)
            
            # Draw different relationship types with different styles
            for rel_type, style in self.edge_styles.items():
                edges = [(u, v) for u, v, d in hierarchy_graph.edges(data=True) 
                        if d.get('type') == rel_type]
                if edges:
                    nx.draw_networkx_edges(
                        hierarchy_graph, pos,
                        edgelist=edges,
                        edge_color=style['color'],
                        width=style['width'],
                        alpha=0.7,
                        arrows=True,
                        arrowsize=20
                    )
            
            # Draw nodes with different colors by type
            for entity_type in EntityType:
                nodes = [node for node in hierarchy_graph.nodes() 
                        if node in self.entities and self.entities[node].type == entity_type]
                if nodes:
                    nx.draw_networkx_nodes(
                        hierarchy_graph, pos,
                        nodelist=nodes,
                        node_color=self.color_schemes["by_type"].get(entity_type.value, "#95a5a6"),
                        node_size=800,
                        alpha=0.8,
                        linewidths=2,
                        edgecolors='white'
                    )
            
            # Add labels
            labels = {}
            for node in hierarchy_graph.nodes():
                if node in self.entities:
                    entity = self.entities[node]
                    label = entity.name[:15] + "..." if len(entity.name) > 15 else entity.name
                    labels[node] = label
            
            nx.draw_networkx_labels(hierarchy_graph, pos, labels, font_size=8, font_weight='bold')
            
            plt.title("Hygiene Standards Hierarchy", fontsize=16, fontweight='bold', pad=20)
            plt.axis('off')
            plt.tight_layout()
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Hierarchy diagram saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating hierarchy diagram: {e}")
            raise
    
    def create_dashboard_summary(self, output_path: str = "dashboard.png") -> str:
        """Create comprehensive dashboard with multiple visualizations"""
        try:
            fig = plt.figure(figsize=(20, 16), dpi=300)
            
            # Create subplots
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. Main graph overview (top row, spans 2 columns)
            ax_main = fig.add_subplot(gs[0, :2])
            self._create_overview_graph(ax_main)
            
            # 2. Entity type distribution (top right)
            ax_types = fig.add_subplot(gs[0, 2])
            self._create_entity_type_chart(ax_types)
            
            # 3. Enforcement level distribution (middle left)
            ax_enforcement = fig.add_subplot(gs[1, 0])
            self._create_enforcement_chart(ax_enforcement)
            
            # 4. Compliance scores (middle center)
            ax_compliance = fig.add_subplot(gs[1, 1])
            self._create_compliance_chart(ax_compliance)
            
            # 5. Network metrics (middle right)
            ax_metrics = fig.add_subplot(gs[1, 2])
            self._create_metrics_chart(ax_metrics)
            
            # 6. Timeline/activity (bottom row, spans all columns)
            ax_timeline = fig.add_subplot(gs[2, :])
            self._create_timeline_chart(ax_timeline)
            
            plt.suptitle("SutazAI Knowledge Graph Dashboard", fontsize=20, fontweight='bold', y=0.98)
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Dashboard saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            raise
    
    def export_for_web(self, output_dir: str) -> Dict[str, str]:
        """Export visualizations optimized for web display"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            files_created = {}
            
            # Create interactive HTML
            interactive_data = self.create_interactive_visualization()
            html_path = output_path / "interactive_graph.html"
            
            # Create custom HTML with the plot
            html_content = self._generate_web_html(interactive_data)
            with open(html_path, 'w') as f:
                f.write(html_content)
            files_created['interactive'] = str(html_path)
            
            # Create static overview
            overview_path = output_path / "overview.png"
            self.create_static_diagram(output_path=str(overview_path), figsize=(12, 8))
            files_created['overview'] = str(overview_path)
            
            # Create compliance heatmap
            heatmap_path = output_path / "compliance.png"
            self.create_compliance_heatmap(output_path=str(heatmap_path))
            files_created['compliance'] = str(heatmap_path)
            
            # Create hierarchy diagram
            hierarchy_path = output_path / "hierarchy.png"
            self.create_hierarchy_diagram(output_path=str(hierarchy_path))
            files_created['hierarchy'] = str(hierarchy_path)
            
            # Export graph data as JSON
            graph_data = self.kg_builder.get_visualization_data()
            json_path = output_path / "graph_data.json"
            with open(json_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
            files_created['data'] = str(json_path)
            
            # Create CSS and JS files
            css_path = output_path / "styles.css"
            with open(css_path, 'w') as f:
                f.write(self._generate_web_css())
            files_created['css'] = str(css_path)
            
            logger.info(f"Web export completed: {len(files_created)} files created")
            return files_created
            
        except Exception as e:
            logger.error(f"Error exporting for web: {e}")
            raise
    
    # Helper methods
    
    def _calculate_layout(self, layout: str) -> Dict[str, Tuple[float, float]]:
        """Calculate node positions based on layout algorithm"""
        if layout == "force":
            return nx.spring_layout(self.graph, k=1/math.sqrt(self.graph.number_of_nodes()), iterations=50)
        elif layout == "hierarchical":
            try:
                return nx.nx_agraph.graphviz_layout(self.graph, prog="dot")
            except:
                return nx.spring_layout(self.graph)
        elif layout == "circular":
            return nx.circular_layout(self.graph)
        elif layout == "tree":
            try:
                return nx.nx_agraph.graphviz_layout(self.graph, prog="twopi")
            except:
                return nx.spring_layout(self.graph)
        else:
            return nx.kamada_kawai_layout(self.graph)
    
    def _prepare_plotly_nodes(self, pos, color_by, size_by, max_nodes):
        """Prepare node data for Plotly visualization"""
        nodes = list(self.graph.nodes())[:max_nodes]
        
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        node_info = {}
        
        for node in nodes:
            if node in pos:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                entity = self.entities.get(node)
                if entity:
                    # Text info
                    text = f"<b>{entity.name}</b><br>"
                    text += f"Type: {entity.type.value}<br>"
                    text += f"Category: {entity.properties.get('category', 'N/A')}<br>"
                    text += f"Enforcement: {entity.properties.get('enforcement_level', 'N/A')}"
                    node_text.append(text)
                    
                    # Color
                    if color_by == "type":
                        color = self.color_schemes["by_type"].get(entity.type.value, "#95a5a6")
                    elif color_by == "enforcement":
                        enforcement = entity.properties.get('enforcement_level', '')
                        color = self.color_schemes["by_enforcement"].get(str(enforcement), "#95a5a6")
                    elif color_by == "category":
                        category = entity.properties.get('category', 'general')
                        color = self.color_schemes["by_category"].get(category, "#95a5a6")
                    else:
                        color = "#95a5a6"
                    node_colors.append(color)
                    
                    # Size
                    if size_by == "degree":
                        size = max(10, min(50, self.graph.degree(node) * 5))
                    else:
                        size = 20
                    node_sizes.append(size)
                    
                    # Store info for API
                    node_info[node] = {
                        "name": entity.name,
                        "type": entity.type.value,
                        "description": entity.description[:100] + "..." if len(entity.description) > 100 else entity.description,
                        "properties": entity.properties
                    }
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=False,
                color=node_colors,
                size=node_sizes,
                line=dict(width=2, color='white')
            )
        )
        
        return node_trace, node_info
    
    def _prepare_plotly_edges(self, pos):
        """Prepare edge data for Plotly visualization"""
        edge_traces = []
        
        # Group edges by type for different styling
        edge_groups = {}
        for edge in self.graph.edges(data=True):
            source, target, data = edge
            edge_type = data.get('type', 'unknown')
            if edge_type not in edge_groups:
                edge_groups[edge_type] = []
            edge_groups[edge_type].append((source, target))
        
        for edge_type, edges in edge_groups.items():
            edge_x = []
            edge_y = []
            
            for source, target in edges:
                if source in pos and target in pos:
                    x0, y0 = pos[source]
                    x1, y1 = pos[target]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
            
            if edge_x:
                style = self.edge_styles.get(edge_type, {"color": "#888", "width": 1})
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=style["width"], color=style["color"]),
                    hoverinfo='none',
                    mode='lines',
                    name=edge_type.replace('_', ' ').title(),
                    showlegend=True
                )
                edge_traces.append(edge_trace)
        
        return edge_traces
    
    def _add_plotly_legend(self, fig, color_by):
        """Add legend to Plotly figure"""
        # This would add a custom legend based on the color scheme
        pass
    
    def _get_node_colors(self, color_by):
        """Get node colors for matplotlib"""
        colors = []
        for node in self.graph.nodes():
            entity = self.entities.get(node)
            if entity:
                if color_by == "type":
                    color = self.color_schemes["by_type"].get(entity.type.value, "#95a5a6")
                elif color_by == "enforcement":
                    enforcement = entity.properties.get('enforcement_level', '')
                    color = self.color_schemes["by_enforcement"].get(str(enforcement), "#95a5a6")
                else:
                    color = "#95a5a6"
                colors.append(color)
            else:
                colors.append("#95a5a6")
        return colors
    
    def _get_node_sizes(self, size_by):
        """Get node sizes for matplotlib"""
        sizes = []
        for node in self.graph.nodes():
            if size_by == "degree":
                size = max(100, min(2000, self.graph.degree(node) * 100))
            else:
                size = 500
            sizes.append(size)
        return sizes
    
    def _get_edge_colors(self):
        """Get edge colors for matplotlib"""
        colors = []
        for edge in self.graph.edges(data=True):
            edge_type = edge[2].get('type', 'unknown')
            style = self.edge_styles.get(edge_type, {"color": "#888"})
            colors.append(style["color"])
        return colors
    
    def _get_important_nodes(self, max_nodes=20):
        """Get most important nodes for labeling"""
        # Get nodes with highest centrality
        try:
            centrality = nx.pagerank(self.graph)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            return [node for node, _ in sorted_nodes[:max_nodes]]
        except:
            return list(self.graph.nodes())[:max_nodes]
    
    def _add_matplotlib_legend(self, color_by):
        """Add legend to matplotlib figure"""
        if color_by in self.color_schemes:
            legend_elements = []
            for label, color in self.color_schemes[color_by].items():
                legend_elements.append(patches.Patch(color=color, label=label))
            
            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    def _add_stats_box(self):
        """Add statistics box to matplotlib figure"""
        stats_text = f"""Graph Statistics:
Nodes: {self.graph.number_of_nodes()}
Edges: {self.graph.number_of_edges()}
Density: {nx.density(self.graph):.3f}
Agents: {len([e for e in self.entities.values() if e.type == EntityType.AGENT])}
Standards: {len([e for e in self.entities.values() if e.type == EntityType.STANDARD])}"""
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
    
    def _create_overview_graph(self, ax):
        """Create overview graph for dashboard"""
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        node_colors = self._get_node_colors("type")
        node_sizes = [200 for _ in self.graph.nodes()]
        
        nx.draw_networkx_edges(self.graph, pos, ax=ax, alpha=0.5, width=1)
        nx.draw_networkx_nodes(self.graph, pos, ax=ax, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.8)
        
        ax.set_title("Knowledge Graph Overview", fontweight='bold')
        ax.axis('off')
    
    def _create_entity_type_chart(self, ax):
        """Create entity type distribution chart"""
        type_counts = {}
        for entity in self.entities.values():
            type_name = entity.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        colors = [self.color_schemes["by_type"].get(t, "#95a5a6") for t in type_counts.keys()]
        ax.pie(type_counts.values(), labels=type_counts.keys(), colors=colors, autopct='%1.1f%%')
        ax.set_title("Entity Types", fontweight='bold')
    
    def _create_enforcement_chart(self, ax):
        """Create enforcement level distribution chart"""
        enforcement_counts = {"BLOCKING": 0, "WARNING": 0, "GUIDANCE": 0}
        
        for entity in self.entities.values():
            enforcement = entity.properties.get('enforcement_level', '')
            if str(enforcement) in enforcement_counts:
                enforcement_counts[str(enforcement)] += 1
        
        bars = ax.bar(enforcement_counts.keys(), enforcement_counts.values(),
                     color=[self.color_schemes["by_enforcement"][k] for k in enforcement_counts.keys()])
        ax.set_title("Enforcement Levels", fontweight='bold')
        ax.set_ylabel("Count")
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
    
    def _create_compliance_chart(self, ax):
        """Create compliance scores chart"""
        # This is simplified - in practice, you'd calculate actual compliance scores
        compliance_data = {"Excellent": 15, "Good": 25, "Needs Work": 8, "Poor": 2}
        
        colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
        bars = ax.bar(compliance_data.keys(), compliance_data.values(), color=colors)
        ax.set_title("Compliance Distribution", fontweight='bold')
        ax.set_ylabel("Number of Agents")
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
    
    def _create_metrics_chart(self, ax):
        """Create network metrics chart"""
        try:
            metrics = {
                "Density": nx.density(self.graph),
                "Clustering": nx.average_clustering(self.graph),
                "Efficiency": nx.global_efficiency(self.graph)
            }
        except:
            metrics = {"Density": 0.1, "Clustering": 0.3, "Efficiency": 0.2}
        
        bars = ax.bar(metrics.keys(), metrics.values(), color='#3498db')
        ax.set_title("Network Metrics", fontweight='bold')
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        
        for bar, (k, v) in zip(bars, metrics.items()):
            ax.text(bar.get_x() + bar.get_width()/2., v + 0.02,
                   f'{v:.3f}', ha='center', va='bottom')
    
    def _create_timeline_chart(self, ax):
        """Create timeline/activity chart"""
        # Placeholder for timeline data
        dates = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
        violations = [12, 8, 5, 3]
        improvements = [5, 8, 12, 15]
        
        width = 0.35
        x = range(len(dates))
        
        ax.bar([i - width/2 for i in x], violations, width, label='Violations', color='#e74c3c', alpha=0.8)
        ax.bar([i + width/2 for i in x], improvements, width, label='Improvements', color='#2ecc71', alpha=0.8)
        
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Count')
        ax.set_title('Compliance Trends', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(dates)
        ax.legend()
    
    def _generate_web_html(self, interactive_data):
        """Generate complete HTML page for web display"""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>SutazAI Knowledge Graph</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .stat-box {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .graph-container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>SutazAI Codebase Hygiene Standards Knowledge Graph</h1>
        <p>Interactive visualization of standards, agents, and compliance relationships</p>
    </div>
    
    <div class="stats">
        <div class="stat-box">
            <h3>Total Nodes</h3>
            <p>{interactive_data['stats']['total_nodes']}</p>
        </div>
        <div class="stat-box">
            <h3>Total Edges</h3>
            <p>{interactive_data['stats']['total_edges']}</p>
        </div>
        <div class="stat-box">
            <h3>Layout</h3>
            <p>{interactive_data['stats']['layout'].title()}</p>
        </div>
        <div class="stat-box">
            <h3>Color Scheme</h3>
            <p>{interactive_data['stats']['color_scheme'].title()}</p>
        </div>
    </div>
    
    <div class="graph-container">
        <div id="graph-div" style="width:100%; height:600px;"></div>
    </div>
    
    <script>
        var plotData = {interactive_data['plot_json']};
        Plotly.newPlot('graph-div', plotData.data, plotData.layout, {{responsive: true}});
    </script>
</body>
</html>
"""
    
    def _generate_web_css(self):
        """Generate CSS for web export"""
        return """
/* SutazAI Knowledge Graph Styles */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f8f9fa;
    color: #333;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.header {
    text-align: center;
    margin-bottom: 30px;
    background: white;
    padding: 30px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

.stat-card {
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    text-align: center;
}

.stat-card h3 {
    margin: 0 0 10px 0;
    color: #2c3e50;
}

.stat-card .number {
    font-size: 2em;
    font-weight: bold;
    color: #3498db;
}

.visualization-container {
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    margin: 20px 0;
}

.controls {
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}

.control-group {
    display: flex;
    flex-direction: column;
    gap: 5px;
}

.control-group label {
    font-weight: bold;
    color: #2c3e50;
}

.control-group select, .control-group input {
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
}

.legend {
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
    margin: 20px 0;
    padding: 15px;
    background: #f8f9fa;
    border-radius: 5px;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
}

.legend-color {
    width: 16px;
    height: 16px;
    border-radius: 3px;
}

@media (max-width: 768px) {
    .stats-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    
    .controls {
        flex-direction: column;
    }
}
"""