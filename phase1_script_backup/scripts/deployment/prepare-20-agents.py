#!/usr/bin/env python3
"""
SutazAI Agent Deployment Preparation System
==========================================

Purpose: Prepare and configure critical agents for enterprise deployment
Author: Ultra Code Optimizer
Created: August 10, 2025
Python Version: 3.12+

Usage:
    python3 prepare-20-agents.py --help
    python3 prepare-20-agents.py --verbose --agents-dir /path/to/agents
    python3 prepare-20-agents.py --dry-run --filter "hardware-*"

Requirements:
    - Python 3.12+
    - Write permissions to agents directory
    - Docker environment for validation
"""

import argparse
import json
import logging
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess

DOCKERFILE_TEMPLATE = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install base Python packages
RUN pip install --no-cache-dir \\
    fastapi==0.104.1 \\
    uvicorn==0.24.0 \\
    pydantic==2.5.0 \\
    httpx==0.25.2 \\
    python-dotenv==1.0.0 \\
    redis==5.0.1 \\
    prometheus-client==0.19.0 \\
    psutil==5.9.6 \\
    structlog==23.2.0

# Copy agent-specific requirements if they exist
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Copy application code
COPY . .

# Set agent-specific environment
# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV AGENT_NAME={agent_name}

# Ensure proper permissions
# Create non-root user
RUN groupadd -r agent && useradd -r -g agent agent
RUN chown -R agent:agent /app
USER agent

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
"""

MAIN_PY_TEMPLATE = '''#!/usr/bin/env python3
"""
Main entry point for {agent_name}
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="{agent_title}",
    description="Agent for {description}",
    version="1.0.0"
)

# Request/Response models
class HealthResponse(BaseModel):
    status: str
    agent: str
    timestamp: str
    version: str = "1.0.0"

class TaskRequest(BaseModel):
    type: str = "process"
    data: dict = {{}}
    priority: str = "normal"

class TaskResponse(BaseModel):
    status: str
    agent: str
    result: dict = {{}}
    timestamp: str

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        agent="{agent_name}",
        timestamp=datetime.utcnow().isoformat()
    )

# Main task processing endpoint
@app.post("/task", response_model=TaskResponse)
async def process_task(request: TaskRequest):
    """Process incoming tasks"""
    try:
        logger.info(f"Processing task of type: {{request.type}}")
        
        result = {{
            "message": "Task processed successfully",
            "task_type": request.type,
            "data_keys": list(request.data.keys())
        }}
        
        return TaskResponse(
            status="success",
            agent="{agent_name}",
            result=result,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Error processing task: {{e}}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {{
        "agent": "{agent_name}",
        "status": "running",
        "endpoints": ["/health", "/task", "/docs"]
    }}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
'''

REQUIREMENTS_TEMPLATE = """# Agent-specific requirements for {agent_name}
# Add any additional dependencies below
"""

def create_agent_files(agent_dir: Path, agent_name: str):
    """Create necessary files for an agent"""
    # Create Dockerfile if missing
    dockerfile_path = agent_dir / 'Dockerfile'
    if not dockerfile_path.exists():
        with open(dockerfile_path, 'w') as f:
            f.write(DOCKERFILE_TEMPLATE.format(agent_name=agent_name))
        print(f"  ✓ Created Dockerfile for {agent_name}")
    
    # Create requirements.txt if missing
    requirements_path = agent_dir / 'requirements.txt'
    if not requirements_path.exists():
        with open(requirements_path, 'w') as f:
            f.write(REQUIREMENTS_TEMPLATE.format(agent_name=agent_name))
        print(f"  ✓ Created requirements.txt for {agent_name}")
    
    # Create main.py if missing
    main_path = agent_dir / 'main.py'
    if not main_path.exists():
        agent_title = agent_name.replace('-', ' ').title()
        description = f"{agent_title} operations"
        
        with open(main_path, 'w') as f:
            f.write(MAIN_PY_TEMPLATE.format(
                agent_name=agent_name,
                agent_title=agent_title,
                description=description
            ))
        os.chmod(main_path, 0o755)
        print(f"  ✓ Created main.py for {agent_name}")

class AgentDeploymentError(Exception):
    """Custom exception for agent deployment issues"""
    pass


class AgentPreparationSystem:
    """Enterprise-grade agent preparation system"""
    
    def __init__(self, agents_dir: Path, verbose: bool = False, dry_run: bool = False):
        self.agents_dir = Path(agents_dir)
        self.verbose = verbose
        self.dry_run = dry_run
        self.logger = self._setup_logging()
        
        # Validate Python version
        if sys.version_info < (3, 12):
            raise RuntimeError(f"Python 3.12+ required, got {sys.version_info.major}.{sys.version_info.minor}")
        
        # Validate agents directory
        if not self.agents_dir.exists():
            raise AgentDeploymentError(f"Agents directory not found: {self.agents_dir}")
        
        self.stats = {
            "ready_count": 0,
            "fixed_count": 0,
            "error_count": 0,
            "skipped_count": 0
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup enterprise-grade logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_dir = self.agents_dir.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "agent-preparation.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def get_critical_agents(self, filter_pattern: Optional[str] = None) -> List[str]:
        """Get list of critical agents with optional filtering"""
        critical_agents = [
            'hardware-resource-optimizer',
            'ai-system-architect',
            'ai-agent-orchestrator',
            'infrastructure-devops-manager',
            'deployment-automation-master',
            'senior-ai-engineer',
            'agent-orchestrator',
            'code-generation-improver',
            'agent-creator',
            'docker-specialist',
            'qa-tester',
            'performance-monitoring',
            'git-manager',
            'python-specialist',
            'ci-cd-automation',
            'system-health-monitor',
            'documentation-generator',
            'code-analyzer',
            'database-manager',
            'api-endpoint-creator'
        ]
        
        if filter_pattern:
            import fnmatch
            critical_agents = [agent for agent in critical_agents 
                             if fnmatch.fnmatch(agent, filter_pattern)]
            self.logger.info(f"Filtered agents with pattern '{filter_pattern}': {len(critical_agents)} agents")
        
        return critical_agents
    
    def validate_agent_directory(self, agent_dir: Path) -> Tuple[bool, Dict[str, bool]]:
        """Validate agent directory and check required files"""
        required_files = {
            'Dockerfile': (agent_dir / 'Dockerfile').exists(),
            'requirements.txt': (agent_dir / 'requirements.txt').exists(),
            'main.py': (agent_dir / 'main.py').exists()
        }
        
        all_present = all(required_files.values())
        return all_present, required_files
    
    def create_dockerfile(self, agent_dir: Path, agent_name: str) -> bool:
        """Create standardized Dockerfile for agent"""
        dockerfile_path = agent_dir / 'Dockerfile'
        
        if dockerfile_path.exists() and not self.dry_run:
            self.logger.warning(f"Dockerfile already exists for {agent_name}, skipping")
            return False
        
        dockerfile_content = DOCKERFILE_TEMPLATE.format(agent_name=agent_name)
        
        if not self.dry_run:
            with open(dockerfile_path, 'w', encoding='utf-8') as f:
                f.write(dockerfile_content)
            self.logger.info(f"Created Dockerfile for {agent_name}")
        else:
            self.logger.info(f"[DRY RUN] Would create Dockerfile for {agent_name}")
        
        return True
    
    def create_requirements_file(self, agent_dir: Path, agent_name: str) -> bool:
        """Create requirements.txt for agent"""
        requirements_path = agent_dir / 'requirements.txt'
        
        if requirements_path.exists() and not self.dry_run:
            self.logger.warning(f"requirements.txt already exists for {agent_name}, skipping")
            return False
        
        requirements_content = REQUIREMENTS_TEMPLATE.format(agent_name=agent_name)
        
        if not self.dry_run:
            with open(requirements_path, 'w', encoding='utf-8') as f:
                f.write(requirements_content)
            self.logger.info(f"Created requirements.txt for {agent_name}")
        else:
            self.logger.info(f"[DRY RUN] Would create requirements.txt for {agent_name}")
        
        return True
    
    def create_main_file(self, agent_dir: Path, agent_name: str) -> bool:
        """Create main.py for agent"""
        main_path = agent_dir / 'main.py'
        
        if main_path.exists() and not self.dry_run:
            self.logger.warning(f"main.py already exists for {agent_name}, skipping")
            return False
        
        agent_title = agent_name.replace('-', ' ').title()
        description = f"{agent_title} operations"
        
        main_content = MAIN_PY_TEMPLATE.format(
            agent_name=agent_name,
            agent_title=agent_title,
            description=description
        )
        
        if not self.dry_run:
            with open(main_path, 'w', encoding='utf-8') as f:
                f.write(main_content)
            os.chmod(main_path, 0o755)
            self.logger.info(f"Created main.py for {agent_name}")
        else:
            self.logger.info(f"[DRY RUN] Would create main.py for {agent_name}")
        
        return True
    
    def process_agent(self, agent_name: str) -> Dict[str, any]:
        """Process individual agent with comprehensive error handling"""
        agent_dir = self.agents_dir / agent_name
        result = {
            "agent": agent_name,
            "status": "pending",
            "actions": [],
            "errors": []
        }
        
        try:
            if not agent_dir.exists():
                result["status"] = "error"
                result["errors"].append(f"Agent directory not found: {agent_dir}")
                self.logger.error(f"Agent directory not found: {agent_name}")
                self.stats["error_count"] += 1
                return result
            
            # Validate current state
            is_ready, file_status = self.validate_agent_directory(agent_dir)
            
            if is_ready:
                result["status"] = "ready"
                self.logger.info(f"Agent {agent_name} is already ready")
                self.stats["ready_count"] += 1
                return result
            
            # Create missing files
            files_created = 0
            
            if not file_status['Dockerfile']:
                if self.create_dockerfile(agent_dir, agent_name):
                    result["actions"].append("Created Dockerfile")
                    files_created += 1
            
            if not file_status['requirements.txt']:
                if self.create_requirements_file(agent_dir, agent_name):
                    result["actions"].append("Created requirements.txt")
                    files_created += 1
            
            if not file_status['main.py']:
                if self.create_main_file(agent_dir, agent_name):
                    result["actions"].append("Created main.py")
                    files_created += 1
            
            if files_created > 0:
                result["status"] = "fixed"
                self.stats["fixed_count"] += 1
                self.logger.info(f"Fixed agent {agent_name}: {files_created} files created")
            else:
                result["status"] = "ready"
                self.stats["ready_count"] += 1
            
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            self.logger.error(f"Error processing agent {agent_name}: {e}")
            if self.verbose:
                self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.stats["error_count"] += 1
        
        return result
    
    def validate_deployment_readiness(self) -> Dict[str, any]:
        """Validate overall deployment readiness"""
        validation_results = {
            "docker_available": False,
            "compose_available": False,
            "disk_space_mb": 0,
            "warnings": [],
            "errors": []
        }
        
        try:
            # Check Docker availability
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            validation_results["docker_available"] = result.returncode == 0
            
            # Check Docker Compose availability
            result = subprocess.run(
                ["docker-compose", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            validation_results["compose_available"] = result.returncode == 0
            
            # Check disk space
            disk_usage = shutil.disk_usage(self.agents_dir)
            validation_results["disk_space_mb"] = disk_usage.free / (1024 * 1024)
            
            if validation_results["disk_space_mb"] < 5000:  # Less than 5GB
                validation_results["warnings"].append("Low disk space available")
            
        except Exception as e:
            validation_results["errors"].append(f"Validation error: {e}")
        
        return validation_results
    
    def generate_deployment_report(self, results: List[Dict], validation: Dict) -> str:
        """Generate comprehensive deployment report"""
        report_dir = self.agents_dir.parent / "logs"
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"agent-deployment-report-{timestamp}.json"
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "agents_dir": str(self.agents_dir),
            "dry_run": self.dry_run,
            "statistics": self.stats,
            "validation": validation,
            "agent_results": results,
            "next_steps": [
                "docker-compose build --parallel",
                "docker-compose up -d",
                "curl http://localhost:PORT/health for each agent"
            ]
        }
        
        if not self.dry_run:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)
        
        return str(report_path)
    
    def run_preparation(self, filter_pattern: Optional[str] = None) -> int:
        """Main preparation execution with comprehensive error handling"""
        try:
            self.logger.info("Starting SutazAI Agent Deployment Preparation")
            self.logger.info(f"Agents directory: {self.agents_dir}")
            self.logger.info(f"Dry run mode: {self.dry_run}")
            
            # Get agents to process
            critical_agents = self.get_critical_agents(filter_pattern)
            self.logger.info(f"Processing {len(critical_agents)} critical agents")
            
            # Process each agent
            results = []
            for agent_name in critical_agents:
                self.logger.info(f"Processing agent: {agent_name}")
                result = self.process_agent(agent_name)
                results.append(result)
            
            # Validate deployment readiness
            validation_results = self.validate_deployment_readiness()
            
            # Generate report
            report_path = self.generate_deployment_report(results, validation_results)
            
            # Log summary
            total_agents = len(critical_agents)
            self.logger.info("=== DEPLOYMENT PREPARATION SUMMARY ===")
            self.logger.info(f"Total agents processed: {total_agents}")
            self.logger.info(f"Ready agents: {self.stats['ready_count']}")
            self.logger.info(f"Fixed agents: {self.stats['fixed_count']}")
            self.logger.info(f"Error agents: {self.stats['error_count']}")
            self.logger.info(f"Report generated: {report_path}")
            
            if self.stats['error_count'] > 0:
                self.logger.warning(f"{self.stats['error_count']} agents had errors")
                return 1
            
            self.logger.info("Agent preparation completed successfully")
            return 0
            
        except Exception as e:
            self.logger.error(f"Agent preparation failed: {e}")
            if self.verbose:
                self.logger.error(f"Traceback: {traceback.format_exc()}")
            return 1


def create_argument_parser() -> argparse.ArgumentParser:
    """Create standardized argument parser"""
    parser = argparse.ArgumentParser(
        description="SutazAI Agent Deployment Preparation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --verbose
    %(prog)s --dry-run --filter "hardware-*"
    %(prog)s --agents-dir /custom/path/agents
    %(prog)s --validate-only
        """
    )
    
    parser.add_argument(
        "--agents-dir",
        type=Path,
        default=Path("/opt/sutazaiapp/agents"),
        help="Path to agents directory (default: /opt/sutazaiapp/agents)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="Filter agents by name pattern (supports wildcards)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate deployment readiness, don't modify files"
    )
    
    return parser


def main():
    """Main entry point with enterprise-grade error handling"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Initialize preparation system
        prep_system = AgentPreparationSystem(
            agents_dir=args.agents_dir,
            verbose=args.verbose,
            dry_run=args.dry_run or args.validate_only
        )
        
        # Run preparation
        if args.validate_only:
            validation_results = prep_system.validate_deployment_readiness()
            print(json.dumps(validation_results, indent=2, default=str))
            return 0 if not validation_results.get("errors") else 1
        else:
            return prep_system.run_preparation(args.filter)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1
    
    # List of 20 critical agents to prepare
    critical_agents = [
        'hardware-resource-optimizer',
        'ai-system-architect',
        'ai-agent-orchestrator',
        'infrastructure-devops-manager',
        'deployment-automation-master',
        'senior-ai-engineer',
        'agent-orchestrator',
        'code-generation-improver',
        'agent-creator',
        'docker-specialist',
        'qa-tester',
        'performance-monitoring',
        'git-manager',
        'python-specialist',
        'ci-cd-automation',
        'system-health-monitor',
        'documentation-generator',
        'code-analyzer',
        'database-manager',
        'api-endpoint-creator'
    ]
    
    print("Preparing 20 critical agents for deployment...\n")
    
    fixed_count = 0
    ready_count = 0
    
    for agent_name in critical_agents:
        agent_dir = agents_dir / agent_name
        
        if not agent_dir.exists():
            print(f"⚠ Agent directory not found: {agent_name}")
            continue
        
        print(f"Processing {agent_name}:")
        
        # Check what's missing
        dockerfile_exists = (agent_dir / 'Dockerfile').exists()
        requirements_exists = (agent_dir / 'requirements.txt').exists()
        main_exists = (agent_dir / 'main.py').exists()
        
        if dockerfile_exists and requirements_exists and main_exists:
            print(f"  ✓ Agent is ready")
            ready_count += 1
        else:
            create_agent_files(agent_dir, agent_name)
            fixed_count += 1
        
        print()
    
    print(f"\n✅ Summary:")
    print(f"  - {ready_count} agents were already ready")
    print(f"  - {fixed_count} agents were fixed")
    print(f"  - Total: {ready_count + fixed_count}/20 agents ready for deployment")
    
    print("\nNext steps:")
    print("1. Run: docker-compose -f docker-compose.agents-20.yml build")
    print("2. Run: docker-compose -f docker-compose.agents-20.yml up -d")

if __name__ == "__main__":
    main()