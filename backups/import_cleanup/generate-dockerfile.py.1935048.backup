#!/usr/bin/env python3
"""
SutazAI Dockerfile Template Generator
Generates service-specific Dockerfiles from master templates
Author: DevOps Manager - Deduplication Operation
Date: August 10, 2025
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List

# Service configuration mapping
SERVICE_CONFIG = {
    "agentgpt": {
        "port": 8515,
        "description": "AgentGPT autonomous agent service",
        "type": "python-agent",
        "base_image": "nodejs-agent-master"  # AgentGPT is Node.js based
    },
    "autogpt": {
        "port": 8501,
        "description": "AutoGPT autonomous agent service", 
        "type": "python-agent",
        "base_image": "python-agent-master"
    },
    "crewai": {
        "port": 8502,
        "description": "CrewAI multi-agent collaboration system",
        "type": "python-agent",
        "base_image": "python-agent-master"
    },
    "langchain": {
        "port": 8503,
        "description": "LangChain agent framework service",
        "type": "python-agent", 
        "base_image": "python-agent-master"
    },
    "langflow": {
        "port": 8504,
        "description": "LangFlow visual workflow designer",
        "type": "python-agent",
        "base_image": "python-agent-master"
    },
    "llamaindex": {
        "port": 8505,
        "description": "LlamaIndex data framework service",
        "type": "python-agent",
        "base_image": "python-agent-master"
    },
    "ollama-integration": {
        "port": 8090,
        "description": "Ollama model integration service",
        "type": "python-agent",
        "base_image": "python-agent-master"
    },
    "hardware-resource-optimizer": {
        "port": 11110,
        "description": "Hardware resource optimization service",
        "type": "python-agent",
        "base_image": "python-agent-master"
    }
}

def load_template(template_path: Path) -> str:
    """Load Dockerfile template content."""
    try:
        with open(template_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Template not found: {template_path}")

def generate_dockerfile(service_name: str, template_content: str, config: Dict) -> str:
    """Generate Dockerfile content from template."""
    
    # Replace template variables
    dockerfile_content = template_content.replace("{{SERVICE_NAME}}", service_name)
    dockerfile_content = dockerfile_content.replace("{{SERVICE_PORT}}", str(config["port"]))
    dockerfile_content = dockerfile_content.replace("{{SERVICE_DESC}}", config["description"])
    
    # Handle base image selection
    base_image = f"sutazai-{config['base_image']}:latest"
    dockerfile_content = dockerfile_content.replace("sutazai-python-agent-master:latest", base_image)
    
    return dockerfile_content

def create_service_dockerfile(service_name: str, output_dir: Path, template_dir: Path):
    """Create Dockerfile for specific service."""
    
    if service_name not in SERVICE_CONFIG:
        raise ValueError(f"Unknown service: {service_name}. Available: {list(SERVICE_CONFIG.keys())}")
    
    config = SERVICE_CONFIG[service_name]
    
    # Determine template based on service type  
    if config["base_image"] == "nodejs-agent-master":
        template_file = template_dir / "Dockerfile.nodejs-agent-template"
    else:
        template_file = template_dir / "Dockerfile.python-agent-template"
    
    # Load and process template
    template_content = load_template(template_file)
    dockerfile_content = generate_dockerfile(service_name, template_content, config)
    
    # Create output directory
    service_dir = output_dir / service_name
    service_dir.mkdir(parents=True, exist_ok=True)
    
    # Write Dockerfile
    dockerfile_path = service_dir / "Dockerfile"
    with open(dockerfile_path, 'w') as f:
        f.write(dockerfile_content)
    
    print(f"Generated: {dockerfile_path}")
    return dockerfile_path

def generate_all_services(output_dir: Path, template_dir: Path):
    """Generate Dockerfiles for all configured services."""
    
    generated_files = []
    
    for service_name in SERVICE_CONFIG.keys():
        try:
            dockerfile_path = create_service_dockerfile(service_name, output_dir, template_dir)
            generated_files.append(str(dockerfile_path))
        except Exception as e:
            print(f"Error generating {service_name}: {e}")
    
    # Create summary report
    summary = {
        "generated_count": len(generated_files),
        "services": list(SERVICE_CONFIG.keys()),
        "generated_files": generated_files,
        "template_dir": str(template_dir),
        "output_dir": str(output_dir)
    }
    
    summary_path = output_dir / "generation-summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Generated {len(generated_files)} Dockerfiles")
    print(f"Summary: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Dockerfiles from templates")
    parser.add_argument("--service", help="Specific service to generate")
    parser.add_argument("--output-dir", default="./generated", help="Output directory")
    parser.add_argument("--template-dir", default=".", help="Template directory")
    parser.add_argument("--all", action="store_true", help="Generate all services")
    
    args = parser.parse_args()
    
    template_dir = Path(args.template_dir)
    output_dir = Path(args.output_dir)
    
    try:
        if args.all:
            generate_all_services(output_dir, template_dir)
        elif args.service:
            create_service_dockerfile(args.service, output_dir, template_dir)
        else:
            print("Specify --service <name> or --all")
            print(f"Available services: {list(SERVICE_CONFIG.keys())}")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())