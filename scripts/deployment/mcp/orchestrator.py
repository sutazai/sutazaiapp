#!/usr/bin/env python3
"""
MCP Container Orchestrator - Manages MCP server deployment with proper isolation and mesh integration
"""

import os
import sys
import json
import subprocess
import time
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import docker
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mcp-orchestrator')

class MCPOrchestrator:
    """Orchestrates MCP server deployment with container isolation and mesh integration"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.base_dir = Path('/opt/sutazaiapp')
        self.mcp_config_path = self.base_dir / '.mcp.json'
        self.port_registry_path = self.base_dir / 'config/ports/mcp_ports.json'
        self.compose_path = self.base_dir / 'docker/docker-compose.mcp-orchestrated.yml'
        
        # Port allocation range for MCP services
        self.port_range = (11100, 11200)
        self.allocated_ports = {}
        self.mcp_servers = {}
        
        # Network configuration
        self.network_name = 'sutazai-network'
        self.container_prefix = 'sutazai-mcp'
        
        # Load MCP configuration
        self.load_mcp_config()
        
    def load_mcp_config(self):
        """Load MCP server configuration from .mcp.json"""
        try:
            with open(self.mcp_config_path, 'r') as f:
                config = json.load(f)
                self.mcp_servers = config.get('mcpServers', {})
                logger.info(f"Loaded {len(self.mcp_servers)} MCP server configurations")
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            sys.exit(1)
    
    def allocate_port(self, service_name: str) -> int:
        """Allocate a unique port for an MCP service"""
        # Check existing port registry
        if self.port_registry_path.exists():
            with open(self.port_registry_path, 'r') as f:
                self.allocated_ports = json.load(f)
        
        # Return existing port if already allocated
        if service_name in self.allocated_ports:
            return self.allocated_ports[service_name]
        
        # Find next available port
        used_ports = set(self.allocated_ports.values())
        for port in range(self.port_range[0], self.port_range[1]):
            if port not in used_ports:
                self.allocated_ports[service_name] = port
                self.save_port_registry()
                return port
        
        raise RuntimeError(f"No available ports in range {self.port_range}")
    
    def save_port_registry(self):
        """Save port allocations to registry file"""
        os.makedirs(self.port_registry_path.parent, exist_ok=True)
        with open(self.port_registry_path, 'w') as f:
            json.dump(self.allocated_ports, f, indent=2)
        logger.info(f"Saved port registry with {len(self.allocated_ports)} allocations")
    
    def cleanup_existing_containers(self):
        """Clean up any existing MCP containers"""
        logger.info("Cleaning up existing MCP containers...")
        
        # Stop and remove orphaned containers
        orphan_patterns = ['tender_', 'optimistic_', 'hungry_', 'vigilant_', 
                          'mcp/', 'sutazai-mcp-']
        
        for container in self.docker_client.containers.list(all=True):
            name = container.name
            image = container.image.tags[0] if container.image.tags else ''
            
            # Check if it's an orphaned MCP container
            if any(pattern in name for pattern in orphan_patterns) or \
               any(pattern in image for pattern in orphan_patterns):
                try:
                    logger.info(f"Removing container: {name}")
                    container.stop(timeout=5)
                    container.remove(force=True)
                except Exception as e:
                    logger.warning(f"Failed to remove {name}: {e}")
        
        # Kill any rogue processes
        subprocess.run("pkill -f 'docker run.*mcp/' 2>/dev/null || true", shell=True)
        subprocess.run("pkill -9 -f 'npm exec.*mcp' 2>/dev/null || true", shell=True)
        
        logger.info("Cleanup complete")
    
    def ensure_network(self):
        """Ensure the Docker network exists"""
        try:
            network = self.docker_client.networks.get(self.network_name)
            logger.info(f"Using existing network: {self.network_name}")
        except docker.errors.NotFound:
            network = self.docker_client.networks.create(
                self.network_name,
                driver='bridge',
                labels={'managed_by': 'mcp-orchestrator'}
            )
            logger.info(f"Created network: {self.network_name}")
        return network
    
    def create_mcp_wrapper_image(self, mcp_name: str, config: Dict) -> str:
        """Create a Docker image for an MCP server with STDIO-to-HTTP bridge"""
        image_name = f"{self.container_prefix}-{mcp_name}:latest"
        
        # Create temporary build directory
        build_dir = Path(f"/tmp/mcp-build-{mcp_name}")
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # Create wrapper script
        wrapper_script = build_dir / "wrapper.py"
        wrapper_content = '''#!/usr/bin/env python3
import sys
import json
import asyncio
import subprocess
from aiohttp import web
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-wrapper")

class MCPBridge:
    """Bridge between STDIO MCP and HTTP API"""
    
    def __init__(self, command, args):
        self.command = command
        self.args = args
        self.process = None
        
    async def start(self):
        """Start the MCP process"""
        cmd = [self.command] + self.args
        logger.info(f"Starting MCP: {' '.join(cmd)}")
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
    async def handle_request(self, request):
        """Handle HTTP requests and forward to STDIO"""
        if not self.process:
            return web.json_response({"error": "MCP not started"}, status=500)
        
        try:
            data = await request.json()
            
            # Forward to STDIO
            input_data = json.dumps(data) + "\\n"
            self.process.stdin.write(input_data.encode())
            await self.process.stdin.drain()
            
            # Read response
            response_line = await self.process.stdout.readline()
            response_data = json.loads(response_line.decode())
            
            return web.json_response(response_data)
            
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def health_check(self, request):
        """Health check endpoint"""
        if self.process and self.process.returncode is None:
            return web.json_response({"status": "healthy"})
        return web.json_response({"status": "unhealthy"}, status=503)

async def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "echo"
    args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    bridge = MCPBridge(command, args)
    await bridge.start()
    
    app = web.Application()
    app.router.add_post('/api/mcp', bridge.handle_request)
    app.router.add_get('/health', bridge.health_check)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()
    
    logger.info("MCP Bridge running on port 8080")
    
    # Keep running
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
'''
        wrapper_script.write_text(wrapper_content)
        
        # Create Dockerfile
        dockerfile = build_dir / "Dockerfile"
        
        # Determine base image and requirements based on MCP type
        if 'npx' in str(config.get('command', '')):
            base_image = "node:20-alpine"
            install_cmd = f"RUN npm install -g {config['args'][0]}" if config.get('args') else ""
        else:
            base_image = "python:3.11-slim"
            install_cmd = "RUN pip install aiohttp"
        
        dockerfile_content = f'''FROM {base_image}

WORKDIR /app

# Install dependencies
RUN apk add --no-cache python3 py3-pip || apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install aiohttp

{install_cmd}

# Copy wrapper
COPY wrapper.py /app/wrapper.py
RUN chmod +x /app/wrapper.py

# Run as non-root
RUN adduser -D -u 1000 mcp || useradd -m -u 1000 mcp
USER mcp

EXPOSE 8080

ENTRYPOINT ["python3", "/app/wrapper.py"]
'''
        dockerfile.write_text(dockerfile_content)
        
        # Build image
        try:
            logger.info(f"Building image: {image_name}")
            self.docker_client.images.build(
                path=str(build_dir),
                tag=image_name,
                rm=True,
                forcerm=True
            )
            return image_name
        except Exception as e:
            logger.error(f"Failed to build image for {mcp_name}: {e}")
            raise
        finally:
            # Cleanup
            subprocess.run(f"rm -rf {build_dir}", shell=True)
    
    def deploy_mcp_server(self, name: str, config: Dict) -> Optional[str]:
        """Deploy a single MCP server as a container"""
        container_name = f"{self.container_prefix}-{name}"
        
        try:
            # Check if container already exists
            try:
                existing = self.docker_client.containers.get(container_name)
                if existing.status == 'running':
                    logger.info(f"Container {container_name} already running")
                    return container_name
                existing.remove(force=True)
            except docker.errors.NotFound:
                pass
            
            # Allocate port
            port = self.allocate_port(name)
            
            # Build wrapper image
            image_name = self.create_mcp_wrapper_image(name, config)
            
            # Prepare command
            cmd = []
            if config.get('command'):
                cmd.append(config['command'])
            if config.get('args'):
                cmd.extend(config['args'])
            
            # Create container
            container = self.docker_client.containers.run(
                image=image_name,
                name=container_name,
                command=cmd if cmd else None,
                detach=True,
                network=self.network_name,
                ports={'8080/tcp': port},
                environment={
                    'MCP_NAME': name,
                    'MCP_PORT': str(port),
                    'MESH_URL': 'http://sutazai-backend:8000'
                },
                labels={
                    'managed_by': 'mcp-orchestrator',
                    'mcp_name': name,
                    'mcp_port': str(port)
                },
                restart_policy={'Name': 'unless-stopped'},
                mem_limit='512m',
                cpu_quota=50000  # 0.5 CPU
            )
            
            logger.info(f"Deployed {container_name} on port {port}")
            return container_name
            
        except Exception as e:
            logger.error(f"Failed to deploy {name}: {e}")
            return None
    
    def generate_compose_file(self):
        """Generate docker-compose file for all MCP services"""
        services = {}
        
        for name, config in self.mcp_servers.items():
            port = self.allocate_port(name)
            container_name = f"{self.container_prefix}-{name}"
            
            # Prepare command
            cmd = []
            if config.get('command'):
                cmd.append(config['command'])
            if config.get('args'):
                cmd.extend(config['args'])
            
            services[f"mcp-{name}"] = {
                'image': f"{self.container_prefix}-{name}:latest",
                'container_name': container_name,
                'command': cmd if cmd else None,
                'ports': [f"{port}:8080"],
                'environment': {
                    'MCP_NAME': name,
                    'MCP_PORT': str(port),
                    'MESH_URL': 'http://sutazai-backend:8000'
                },
                'networks': ['sutazai-network'],
                'labels': {
                    'managed_by': 'mcp-orchestrator',
                    'mcp_name': name,
                    'mcp_port': str(port)
                },
                'restart': 'unless-stopped',
                'deploy': {
                    'resources': {
                        'limits': {
                            'cpus': '0.5',
                            'memory': '512M'
                        },
                        'reservations': {
                            'cpus': '0.1',
                            'memory': '128M'
                        }
                    }
                },
                'healthcheck': {
                    'test': ['CMD', 'curl', '-f', 'http://localhost:8080/health'],
                    'interval': '30s',
                    'timeout': '10s',
                    'retries': 3,
                    'start_period': '40s'
                }
            }
        
        compose = {
            'version': '3.8',
            'services': services,
            'networks': {
                'sutazai-network': {
                    'external': True
                }
            }
        }
        
        # Save compose file
        os.makedirs(self.compose_path.parent, exist_ok=True)
        with open(self.compose_path, 'w') as f:
            yaml.dump(compose, f, default_flow_style=False)
        
        logger.info(f"Generated compose file: {self.compose_path}")
        return self.compose_path
    
    def orchestrate_deployment(self, sequential: bool = True):
        """Orchestrate the deployment of all MCP servers"""
        logger.info("Starting MCP deployment orchestration...")
        
        # Clean up existing containers
        self.cleanup_existing_containers()
        
        # Ensure network exists
        self.ensure_network()
        
        # Deploy servers
        deployed = []
        failed = []
        
        for name, config in self.mcp_servers.items():
            logger.info(f"Deploying {name}...")
            
            if sequential:
                # Sequential deployment to prevent conflicts
                time.sleep(2)
            
            result = self.deploy_mcp_server(name, config)
            if result:
                deployed.append(name)
            else:
                failed.append(name)
        
        # Generate compose file for future use
        self.generate_compose_file()
        
        # Report results
        logger.info(f"Deployment complete: {len(deployed)} succeeded, {len(failed)} failed")
        if failed:
            logger.error(f"Failed deployments: {', '.join(failed)}")
        
        return deployed, failed
    
    def health_check_all(self) -> Dict[str, bool]:
        """Check health of all deployed MCP servers"""
        health_status = {}
        
        for name in self.mcp_servers.keys():
            container_name = f"{self.container_prefix}-{name}"
            try:
                container = self.docker_client.containers.get(container_name)
                if container.status == 'running':
                    # Check HTTP health endpoint
                    port = self.allocated_ports.get(name)
                    if port:
                        result = subprocess.run(
                            f"curl -s -f http://localhost:{port}/health",
                            shell=True,
                            capture_output=True
                        )
                        health_status[name] = result.returncode == 0
                    else:
                        health_status[name] = False
                else:
                    health_status[name] = False
            except docker.errors.NotFound:
                health_status[name] = False
        
        return health_status
    
    def test_multi_client_access(self):
        """Test that multiple clients can access MCP servers"""
        logger.info("Testing multi-client access...")
        
        test_results = []
        
        # Test parallel access to different servers
        import concurrent.futures
        
        def test_server(name, port):
            try:
                # Simulate multiple clients
                for client_id in range(3):
                    result = subprocess.run(
                        f"curl -s -X POST http://localhost:{port}/api/mcp "
                        f"-H 'Content-Type: application/json' "
                        f"-d '{{\"client_id\": {client_id}, \"test\": true}}'",
                        shell=True,
                        capture_output=True,
                        timeout=5
                    )
                    if result.returncode != 0:
                        return False
                return True
            except Exception as e:
                logger.error(f"Test failed for {name}: {e}")
                return False
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            for name, port in list(self.allocated_ports.items())[:5]:  # Test first 5
                futures[executor.submit(test_server, name, port)] = name
            
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                result = future.result()
                test_results.append((name, result))
                logger.info(f"Multi-client test for {name}: {'PASSED' if result else 'FAILED'}")
        
        return test_results


def main():
    """Main entry point"""
    orchestrator = MCPOrchestrator()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='MCP Container Orchestrator')
    parser.add_argument('--cleanup', action='store_true', help='Clean up existing containers')
    parser.add_argument('--deploy', action='store_true', help='Deploy all MCP servers')
    parser.add_argument('--health', action='store_true', help='Check health of all servers')
    parser.add_argument('--test', action='store_true', help='Test multi-client access')
    parser.add_argument('--compose', action='store_true', help='Generate docker-compose file')
    parser.add_argument('--sequential', action='store_true', default=True, 
                       help='Deploy servers sequentially (default: True)')
    
    args = parser.parse_args()
    
    if args.cleanup:
        orchestrator.cleanup_existing_containers()
    
    if args.deploy:
        deployed, failed = orchestrator.orchestrate_deployment(sequential=args.sequential)
        print(f"\nDeployed: {', '.join(deployed)}")
        if failed:
            print(f"Failed: {', '.join(failed)}")
            sys.exit(1)
    
    if args.compose:
        compose_path = orchestrator.generate_compose_file()
        print(f"Generated: {compose_path}")
    
    if args.health:
        health = orchestrator.health_check_all()
        print("\nHealth Status:")
        for name, status in health.items():
            print(f"  {name}: {'✓' if status else '✗'}")
    
    if args.test:
        results = orchestrator.test_multi_client_access()
        print("\nMulti-client Access Test:")
        for name, passed in results:
            print(f"  {name}: {'✓' if passed else '✗'}")
    
    if not any([args.cleanup, args.deploy, args.health, args.test, args.compose]):
        # Default action: deploy
        deployed, failed = orchestrator.orchestrate_deployment()
        
        # Health check
        health = orchestrator.health_check_all()
        print("\nDeployment Summary:")
        for name in deployed:
            status = '✓' if health.get(name) else '✗'
            port = orchestrator.allocated_ports.get(name, 'N/A')
            print(f"  {name}: Port {port} [{status}]")


if __name__ == '__main__':
    main()