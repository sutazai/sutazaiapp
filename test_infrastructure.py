#!/usr/bin/env python3
"""
Test infrastructure services for SutazAI
"""

import socket
import time
import httpx
from rich.console import Console
from rich.table import Table

console = Console()

def test_port(host: str, port: int) -> bool:
    """Test if a port is accessible"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def main():
    console.print("\n[bold cyan]SutazAI Infrastructure Test[/bold cyan]\n")
    
    # Define services to test
    services = [
        ("PostgreSQL", "localhost", 5432),
        ("Redis", "localhost", 6379),
        ("Neo4j", "localhost", 7687),
        ("ChromaDB", "localhost", 8001),
        ("Qdrant", "localhost", 6333),
        ("Ollama", "localhost", 11434),
        ("Backend AGI", "localhost", 8000),
        ("Frontend AGI", "localhost", 8501),
        ("Prometheus", "localhost", 9090),
        ("Grafana", "localhost", 3003),
    ]
    
    # Create results table
    table = Table(title="Infrastructure Status")
    table.add_column("Service", style="cyan", no_wrap=True)
    table.add_column("Port", style="magenta")
    table.add_column("Status", style="bold")
    table.add_column("Additional Info", style="yellow")
    
    passed = 0
    total = len(services)
    
    for service_name, host, port in services:
        if test_port(host, port):
            status = "✅ Running"
            passed += 1
            
            # Try to get additional info for some services
            info = ""
            try:
                if service_name == "Ollama":
                    response = httpx.get(f"http://{host}:{port}/api/tags", timeout=2)
                    if response.status_code == 200:
                        models = response.json().get('models', [])
                        info = f"{len(models)} models loaded"
                elif service_name == "Backend AGI":
                    response = httpx.get(f"http://{host}:{port}/health", timeout=2)
                    if response.status_code == 200:
                        info = "API responding"
                elif service_name == "Frontend AGI":
                    response = httpx.get(f"http://{host}:{port}", timeout=2)
                    if response.status_code == 200:
                        info = "UI accessible"
                elif service_name == "ChromaDB":
                    response = httpx.get(f"http://{host}:{port}/api/v1/heartbeat", timeout=2)
                    if response.status_code == 200:
                        info = "Heartbeat OK"
                elif service_name == "Qdrant":
                    response = httpx.get(f"http://{host}:{port}/healthz", timeout=2)
                    if response.status_code == 200:
                        info = "Health check passed"
            except:
                pass
                
            table.add_row(service_name, str(port), status, info)
        else:
            status = "❌ Not accessible"
            table.add_row(service_name, str(port), status, "")
    
    console.print(table)
    console.print(f"\n[bold]Summary:[/bold] {passed}/{total} services running")
    console.print(f"[bold]Status:[/bold] {'✅ All services operational' if passed == total else '⚠️  Some services need attention'}\n")
    
    # Check models in Ollama
    if test_port("localhost", 11434):
        console.print("[bold cyan]Ollama Models:[/bold cyan]")
        try:
            response = httpx.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                if models:
                    for model in models:
                        console.print(f"  • {model['name']} ({model['size'] / (1024**3):.1f}GB)")
                else:
                    console.print("  No models loaded")
        except Exception as e:
            console.print(f"  Error checking models: {e}")

if __name__ == "__main__":
    main() 