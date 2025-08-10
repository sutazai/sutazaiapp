#!/usr/bin/env python3
"""
Update AI Agent configurations with authentication credentials
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import structlog

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()

# Configuration
BASE_DIR = Path("/opt/sutazaiapp")
AGENTS_DIR = BASE_DIR / "agents"
AUTH_DIR = BASE_DIR / "auth"
CONFIGS_DIR = AUTH_DIR / "agent-configs"

SERVICE_ACCOUNT_MANAGER_URL = "http://localhost:10055"
JWT_SERVICE_URL = "http://localhost:10054"
KONG_PROXY_URL = "http://localhost:10051"

# AI Agents that need authentication updates
AI_AGENTS = [
    'adversarial-attack-detector', 'agent-creator', 'agent-debugger', 'agent-orchestrator',
    'agentgpt-autonomous-executor', 'agentgpt', 'agentzero-coordinator', 'ai-agent-debugger',
    'ai-product-manager', 'ai-qa-team-lead', 'ai-scrum-master', 'ai-senior-backend-developer',
    'ai-senior-engineer', 'ai-senior-frontend-developer', 'ai-senior-full-stack-developer',
    'ai-system-architect', 'ai-system-validator', 'ai-testing-qa-validator', 'aider',
    'attention-optimizer', 'autogen', 'autogpt', 'automated-incident-responder',
    'autonomous-task-executor', 'awesome-code-ai', 'babyagi', 'bias-and-fairness-auditor',
    'bigagi-system-manager', 'browser-automation-orchestrator', 'causal-inference-expert',
    'cicd-pipeline-orchestrator', 'code-improver', 'code-quality-gateway-sonarqube',
    'codebase-team-lead', 'cognitive-architecture-designer', 'cognitive-load-monitor',
    'compute-scheduler-and-optimizer', 'container-orchestrator-k3s', 'container-vulnerability-scanner-trivy',
    'context-framework', 'cpu-only-hardware-optimizer', 'crewai', 'data-analysis-engineer',
    'data-drift-detector', 'data-lifecycle-manager', 'data-pipeline-engineer', 'data-version-controller-dvc',
    'deep-learning-brain-architect', 'deep-learning-brain-manager', 'deep-local-brain-builder',
    'deploy-automation-master', 'deployment-automation-master', 'devika', 'dify-automation-specialist',
    'distributed-computing-architect', 'distributed-tracing-analyzer-jaeger', 'document-knowledge-manager',
    'edge-computing-optimizer', 'edge-inference-proxy', 'emergency-shutdown-coordinator',
    'energy-consumption-optimize', 'episodic-memory-engineer', 'ethical-governor',
    'evolution-strategy-trainer', 'experiment-tracker', 'explainability-and-transparency-agent',
    'explainable-ai-specialist', 'federated-learning-coordinator', 'finrobot', 'flowiseai-flow-manager',
    'fsdp', 'garbage-collector-coordinator', 'garbage-collector', 'genetic-algorithm-tuner',
    'goal-setting-and-planning-agent', 'gpt-engineer', 'gpu-hardware-optimizer'
]

class AgentAuthUpdater:
    """Updates agent configurations with authentication"""
    
    def __init__(self):
        self.service_accounts = {}
        
    async def fetch_service_accounts(self):
        """Fetch all service account details"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{SERVICE_ACCOUNT_MANAGER_URL}/service-accounts")
                response.raise_for_status()
                accounts = response.json()
                
                for account in accounts:
                    self.service_accounts[account['name']] = account
                    
                logger.info("Fetched service accounts", count=len(self.service_accounts))
                
        except Exception as e:
            logger.error("Failed to fetch service accounts", error=str(e))
            
    def generate_auth_config(self, agent_name: str) -> Dict:
        """Generate authentication configuration for agent"""
        account = self.service_accounts.get(agent_name)
        
        if not account:
            logger.warning("No service account found for agent", agent=agent_name)
            return {}
            
        return {
            "authentication": {
                "enabled": True,
                "provider": "keycloak",
                "keycloak_url": "http://keycloak:8080",
                "realm": "sutazai",
                "client_id": account['client_id'],
                "client_secret": "[STORED_IN_VAULT]",
                "jwt_service_url": JWT_SERVICE_URL,
                "kong_proxy_url": KONG_PROXY_URL,
                "scopes": account['scopes'],
                "token_refresh_threshold": 300  # Refresh when 5min remaining
            },
            "endpoints": {
                "token": "/auth/token",
                "validate": "/auth/validate", 
                "revoke": "/auth/revoke",
                "userinfo": "/auth/userinfo"
            },
            "headers": {
                "authorization": "Bearer {ACCESS_TOKEN}",
                "x-client-id": account['client_id'],
                "x-service-name": agent_name
            },
            "retry": {
                "max_attempts": 3,
                "backoff_factor": 2,
                "timeout": 30
            }
        }
        
    def update_python_agent(self, agent_path: Path, auth_config: Dict):
        """Update Python-based agent with authentication"""
        app_py = agent_path / "app.py"
        main_py = agent_path / "main.py"
        
        # Check which file exists
        target_file = app_py if app_py.exists() else main_py if main_py.exists() else None
        
        if not target_file:
            logger.warning("No Python main file found", agent=agent_path.name)
            return
            
        try:
            # Read existing file
            with open(target_file, 'r') as f:
                content = f.read()
                
            # Check if authentication is already added
            if "SutazAI Authentication" in content:
                logger.info("Authentication already added", agent=agent_path.name)
                return
                
            # Add authentication imports and setup
            auth_setup = f'''
# SutazAI Authentication Setup
import os
import httpx
from datetime import datetime, timedelta
from typing import Optional

class SutazAIAuth:
    """SutazAI Authentication Client"""
    
    def __init__(self):
        self.config = {json.dumps(auth_config, indent=12)}
        self.access_token = None
        self.token_expires_at = None
        self.client_id = self.config["authentication"]["client_id"]
        self.jwt_service_url = self.config["authentication"]["jwt_service_url"]
        
    async def get_access_token(self) -> Optional[str]:
        """Get valid access token"""
        if self.access_token and self.token_expires_at:
            # Check if token needs refresh
            now = datetime.utcnow()
            expires_at = datetime.fromisoformat(self.token_expires_at.replace('Z', '+00:00'))
            
            if now < expires_at - timedelta(seconds=self.config["authentication"]["token_refresh_threshold"]):
                return self.access_token
                
        # Request new token
        return await self.refresh_token()
        
    async def refresh_token(self) -> Optional[str]:
        """Refresh access token"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{{self.jwt_service_url}}/auth/token",
                    json={{
                        "service_name": "{agent_path.name}",
                        "scopes": self.config["authentication"]["scopes"],
                        "expires_in": 3600
                    }}
                )
                
                if response.status_code == 200:
                    token_data = response.json()
                    self.access_token = token_data["access_token"]
                    self.token_expires_at = token_data["issued_at"]
                    return self.access_token
                    
        except Exception as e:
            print(f"Failed to refresh token: {{e}}")
            
        return None
        
    async def make_authenticated_request(self, method: str, url: str, **kwargs):
        """Make authenticated request"""
        token = await self.get_access_token()
        if not token:
            raise Exception("Failed to get access token")
            
        headers = kwargs.get('headers', {{}})
        headers.update({{
            'Authorization': f'Bearer {{token}}',
            'X-Client-ID': self.client_id,
            'X-Service-Name': '{agent_path.name}'
        }})
        kwargs['headers'] = headers
        
        async with httpx.AsyncClient() as client:
            return await client.request(method, url, **kwargs)

# Initialize authentication
auth_client = SutazAIAuth()
'''
            
            # Insert authentication setup after imports
            import_end = content.find('\n\n')
            if import_end == -1:
                import_end = content.find('\napp = ')
                
            if import_end != -1:
                new_content = content[:import_end] + auth_setup + content[import_end:]
                
                # Write updated file
                with open(target_file, 'w') as f:
                    f.write(new_content)
                    
                logger.info("Updated Python agent with authentication", agent=agent_path.name)
            else:
                logger.warning("Could not find insertion point", agent=agent_path.name)
                
        except Exception as e:
            logger.error("Failed to update Python agent", agent=agent_path.name, error=str(e))
            
    def update_agent_config_file(self, agent_path: Path, auth_config: Dict):
        """Create/update agent configuration file"""
        config_file = agent_path / "auth_config.json"
        
        try:
            with open(config_file, 'w') as f:
                json.dump(auth_config, f, indent=2)
                
            logger.info("Created auth config file", agent=agent_path.name, file=config_file)
            
        except Exception as e:
            logger.error("Failed to create config file", agent=agent_path.name, error=str(e))
            
    def update_docker_compose_env(self, agent_name: str, auth_config: Dict):
        """Update docker-compose.yml with authentication environment variables"""
        compose_file = BASE_DIR / "docker-compose.yml"
        
        if not compose_file.exists():
            logger.warning("docker-compose.yml not found")
            return
            
        try:
            # Read docker-compose file
            with open(compose_file, 'r') as f:
                content = f.read()
                
            # Add authentication environment variables for the service
            env_vars = f'''
      # Authentication
      SUTAZAI_AUTH_ENABLED: "true"
      SUTAZAI_CLIENT_ID: "{auth_config['authentication']['client_id']}"
      SUTAZAI_JWT_SERVICE_URL: "{auth_config['authentication']['jwt_service_url']}"
      SUTAZAI_KONG_PROXY_URL: "{auth_config['authentication']['kong_proxy_url']}"
      SUTAZAI_KEYCLOAK_URL: "{auth_config['authentication']['keycloak_url']}"
      SUTAZAI_REALM: "{auth_config['authentication']['realm']}"'''
            
            # Find the service section and add environment variables
            service_pattern = f"  {agent_name}:"
            if service_pattern in content:
                # Find environment section or create one
                lines = content.split('\n')
                service_start = -1
                
                for i, line in enumerate(lines):
                    if line.strip() == f"{agent_name}:":
                        service_start = i
                        break
                        
                if service_start != -1:
                    # Find environment section or insertion point
                    env_section = -1
                    next_service = -1
                    
                    for i in range(service_start + 1, len(lines)):
                        if lines[i].strip().startswith('environment:'):
                            env_section = i
                            break
                        elif lines[i].startswith('  ') and lines[i].strip().endswith(':'):
                            next_service = i
                            break
                            
                    if env_section != -1:
                        # Add to existing environment section
                        lines.insert(env_section + 1, env_vars)
                    else:
                        # Create new environment section
                        insert_pos = next_service if next_service != -1 else len(lines)
                        lines.insert(insert_pos, "    environment:" + env_vars)
                        
                    # Write updated file
                    with open(compose_file, 'w') as f:
                        f.write('\n'.join(lines))
                        
                    logger.info("Updated docker-compose.yml", agent=agent_name)
                    
        except Exception as e:
            logger.error("Failed to update docker-compose.yml", agent=agent_name, error=str(e))
            
    async def update_all_agents(self):
        """Update all AI agents with authentication"""
        await self.fetch_service_accounts()
        
        # Create output directory
        CONFIGS_DIR.mkdir(exist_ok=True)
        
        updated_count = 0
        
        for agent_name in AI_AGENTS:
            logger.info("Processing agent", agent=agent_name)
            
            # Generate authentication configuration
            auth_config = self.generate_auth_config(agent_name)
            if not auth_config:
                continue
                
            # Save configuration file
            config_file = CONFIGS_DIR / f"{agent_name}.json"
            with open(config_file, 'w') as f:
                json.dump(auth_config, f, indent=2)
                
            # Update agent directory if it exists
            agent_path = AGENTS_DIR / agent_name
            if agent_path.exists():
                # Update Python agent files
                self.update_python_agent(agent_path, auth_config)
                
                # Create agent config file
                self.update_agent_config_file(agent_path, auth_config)
                
                updated_count += 1
            else:
                logger.warning("Agent directory not found", agent=agent_name, path=agent_path)
                
        logger.info("Agent authentication update completed", 
                   total=len(AI_AGENTS), 
                   updated=updated_count)
                   
        return updated_count

async def main():
    """Main function"""
    updater = AgentAuthUpdater()
    
    try:
        updated_count = await updater.update_all_agents()
        
        print(f"✓ Updated {updated_count} agents with authentication")
        print(f"✓ Generated {len(AI_AGENTS)} configuration files")
        print(f"✓ Configuration files saved to: {CONFIGS_DIR}")
        print()
        print("Next steps:")
        print("1. Review generated configurations")
        print("2. Restart agent services to apply changes")
        print("3. Test authenticated API access")
        print("4. Monitor authentication logs")
        
    except Exception as e:
        logger.error("Agent update failed", error=str(e))
        return 1
        
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))