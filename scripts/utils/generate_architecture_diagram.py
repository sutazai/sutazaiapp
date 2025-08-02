#!/usr/bin/env python3
"""
Generate Architecture Diagram for SutazAI System
Creates a visual representation of the system architecture
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_architecture_diagram():
    """Create a visual architecture diagram"""
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    colors = {
        'frontend': '#4CAF50',
        'backend': '#2196F3', 
        'database': '#FF9800',
        'ai': '#9C27B0',
        'monitoring': '#F44336'
    }
    
    # Title
    ax.text(5, 9.5, 'SutazAI automation/advanced automation System Architecture', 
            ha='center', va='center', fontsize=20, fontweight='bold')
    
    # Frontend Layer
    frontend_box = FancyBboxPatch((0.5, 7.5), 9, 1.2, 
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['frontend'],
                                  alpha=0.3,
                                  edgecolor=colors['frontend'],
                                  linewidth=2)
    ax.add_patch(frontend_box)
    ax.text(5, 8.1, 'Frontend Layer (Streamlit UI - Port 8501)', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(2.5, 7.7, '• Dashboard\n• Chat Interface\n• Model Management', 
            ha='center', va='center', fontsize=9)
    ax.text(7.5, 7.7, '• System Monitor\n• Agent Control\n• Settings', 
            ha='center', va='center', fontsize=9)
    
    # Backend API Layer
    backend_box = FancyBboxPatch((0.5, 5.8), 9, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['backend'],
                                 alpha=0.3,
                                 edgecolor=colors['backend'],
                                 linewidth=2)
    ax.add_patch(backend_box)
    ax.text(5, 6.4, 'Backend API Layer (FastAPI - Port 8000)', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(2.5, 6, '• REST APIs\n• WebSockets\n• Authentication', 
            ha='center', va='center', fontsize=9)
    ax.text(5, 6, '• Model Manager\n• Vector DB Manager\n• Agent Orchestrator', 
            ha='center', va='center', fontsize=9)
    ax.text(7.5, 6, '• Code Generation\n• Self-Improvement\n• Monitoring', 
            ha='center', va='center', fontsize=9)
    
    # AI Model Layer
    ai_box = FancyBboxPatch((0.5, 4.1), 9, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['ai'],
                            alpha=0.3,
                            edgecolor=colors['ai'],
                            linewidth=2)
    ax.add_patch(ai_box)
    ax.text(5, 4.7, 'AI Model & Agent Layer', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(2, 4.3, 'Ollama\n(Port 11434)', 
            ha='center', va='center', fontsize=9)
    ax.text(3.5, 4.3, 'AutoGPT\nCrewAI', 
            ha='center', va='center', fontsize=9)
    ax.text(5, 4.3, 'GPT-Engineer\nAider', 
            ha='center', va='center', fontsize=9)
    ax.text(6.5, 4.3, 'DeepSeek-R1\nQwen 2.5', 
            ha='center', va='center', fontsize=9)
    ax.text(8, 4.3, 'Llama 3.2\nNomic Embed', 
            ha='center', va='center', fontsize=9)
    
    # Data Layer
    data_box = FancyBboxPatch((0.5, 2.4), 9, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['database'],
                              alpha=0.3,
                              edgecolor=colors['database'],
                              linewidth=2)
    ax.add_patch(data_box)
    ax.text(5, 3, 'Data & Storage Layer', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(2, 2.6, 'PostgreSQL\n(Port 5432)', 
            ha='center', va='center', fontsize=9)
    ax.text(3.5, 2.6, 'Redis\n(Port 6379)', 
            ha='center', va='center', fontsize=9)
    ax.text(5, 2.6, 'ChromaDB\n(Port 8001)', 
            ha='center', va='center', fontsize=9)
    ax.text(6.5, 2.6, 'Qdrant\n(Port 6333)', 
            ha='center', va='center', fontsize=9)
    ax.text(8, 2.6, 'FAISS\n(In-Memory)', 
            ha='center', va='center', fontsize=9)
    
    # Monitoring Layer
    monitoring_box = FancyBboxPatch((0.5, 0.7), 9, 1.2,
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['monitoring'],
                                    alpha=0.3,
                                    edgecolor=colors['monitoring'],
                                    linewidth=2)
    ax.add_patch(monitoring_box)
    ax.text(5, 1.3, 'Monitoring & Observability', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(3, 0.9, 'Prometheus\n(Port 9090)', 
            ha='center', va='center', fontsize=9)
    ax.text(7, 0.9, 'Grafana\n(Port 3000)', 
            ha='center', va='center', fontsize=9)
    
    # Add connections
    # Frontend to Backend
    arrow1 = ConnectionPatch((5, 7.5), (5, 7.0), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5,
                            mutation_scale=20, fc="black")
    ax.add_artist(arrow1)
    
    # Backend to AI Layer
    arrow2 = ConnectionPatch((5, 5.8), (5, 5.3), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5,
                            mutation_scale=20, fc="black")
    ax.add_artist(arrow2)
    
    # AI Layer to Data Layer
    arrow3 = ConnectionPatch((5, 4.1), (5, 3.6), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5,
                            mutation_scale=20, fc="black")
    ax.add_artist(arrow3)
    
    # Data Layer to Monitoring
    arrow4 = ConnectionPatch((5, 2.4), (5, 1.9), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5,
                            mutation_scale=20, fc="black")
    ax.add_artist(arrow4)
    
    # Add network info
    network_box = FancyBboxPatch((0.2, 0.1), 2, 0.4,
                                boxstyle="round,pad=0.05",
                                facecolor='lightgray',
                                alpha=0.5)
    ax.add_patch(network_box)
    ax.text(1.2, 0.3, 'Docker Network:\nsutazai-network', 
            ha='center', va='center', fontsize=8)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['frontend'], label='User Interface'),
        patches.Patch(color=colors['backend'], label='API & Logic'),
        patches.Patch(color=colors['ai'], label='AI & Models'),
        patches.Patch(color=colors['database'], label='Data Storage'),
        patches.Patch(color=colors['monitoring'], label='Monitoring')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    return fig

def save_diagram(output_path='/opt/sutazaiapp/architecture_diagram.png'):
    """Save the architecture diagram"""
    fig = create_architecture_diagram()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Architecture diagram saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    save_diagram()