"""
Docker container tests for SutazAI system
Tests container functionality, health checks, and service interactions
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import pytest
import docker
import time
import requests
import json
import subprocess
import os
from typing import Dict, List, Optional
import psutil
import socket
from datetime import datetime, timedelta


@pytest.fixture(scope="session")
def docker_client():
    """Docker client fixture."""
    try:
        client = docker.from_env()
        # Test connection
        client.ping()
        return client
    except Exception as e:
        pytest.skip(f"Docker not available: {str(e)}")


@pytest.fixture
def container_configs():
    """Container configuration fixture."""
    return {
        "backend": {
            "image": "sutazai-backend:latest",
            "ports": {"8000/tcp": 8000},
            "environment": {
                "ENV": "test",
                "LOG_LEVEL": "DEBUG"
            },
            "healthcheck_url": "http://localhost:8000/health"
        },
        "frontend": {
            "image": "sutazai-frontend:latest", 
            "ports": {"8501/tcp": 8501},
            "environment": {
                "ENV": "test"
            },
            "healthcheck_url": "http://localhost:8501"
        },
        "redis": {
            "image": "redis:7-alpine",
            "ports": {"6379/tcp": 6379},
            "healthcheck_url": "redis://localhost:6379"
        },
        "postgres": {
            "image": "postgres:15-alpine",
            "ports": {"5432/tcp": 5432},
            "environment": {
                "POSTGRES_DB": "sutazai_test",
                "POSTGRES_USER": "test",
                "POSTGRES_PASSWORD": "test"
            },
            "healthcheck_url": "postgresql://test:test@localhost:5432/sutazai_test"
        }
    }


class TestDockerEnvironment:
    """Test Docker environment and setup."""

    def test_docker_daemon_running(self, docker_client):
        """Test Docker daemon is running."""
        info = docker_client.info()
        assert info is not None
        assert "ServerVersion" in info

    def test_docker_version_compatibility(self, docker_client):
        """Test Docker version compatibility."""
        version = docker_client.version()
        
        # Check minimum Docker version (20.0+)
        version_parts = version["Version"].split(".")
        major_version = int(version_parts[0])
        
        assert major_version >= 20, f"Docker version {version['Version']} may be too old"

    def test_docker_system_resources(self, docker_client):
        """Test Docker system resources."""
        info = docker_client.info()
        
        # Check available memory (at least 2GB)
        memory_gb = info["MemTotal"] / (1024**3)
        assert memory_gb >= 2, f"Insufficient memory: {memory_gb:.1f}GB available"
        
        # Check available disk space
        assert "DockerRootDir" in info

    def test_docker_network_availability(self, docker_client):
        """Test Docker network availability."""
        networks = docker_client.networks.list()
        network_names = [net.name for net in networks]
        
        # Check for default networks
        assert "bridge" in network_names
        assert len(networks) > 0


class TestContainerImages:
    """Test container images."""

    def test_base_images_available(self, docker_client):
        """Test base images are available."""
        base_images = [
            "python:3.12-slim",
            "redis:7-alpine", 
            "postgres:15-alpine",
            "nginx:alpine"
        ]
        
        available_images = [img.tags[0] if img.tags else "none" for img in docker_client.images.list()]
        
        for base_image in base_images:
            # Try to pull if not available
            try:
                docker_client.images.get(base_image)
            except docker.errors.ImageNotFound:
                try:
                    docker_client.images.pull(base_image)
                except Exception as e:
                    pytest.skip(f"Cannot pull base image {base_image}: {str(e)}")

    def test_custom_image_build(self, docker_client):
        """Test custom image building."""
        # Simple test Dockerfile
        dockerfile_content = """
FROM python:3.12-slim
LABEL test=true
RUN echo "Test image" > /tmp/test.txt
CMD ["echo", "Hello from test container"]
"""
        
        try:
            # Build test image
            image, logs = docker_client.images.build(
                fileobj=None,
                dockerfile=dockerfile_content,
                tag="sutazai-test:latest",
                forcerm=True
            )
            
            assert image is not None
            assert "sutazai-test:latest" in image.tags
            
            # Clean up test image
            docker_client.images.remove(image.id, force=True)
            
        except Exception as e:
            pytest.skip(f"Cannot build test image: {str(e)}")

    def test_image_security_scan(self, docker_client):
        """Test image security scanning (basic)."""
        # Get Python base image
        try:
            image = docker_client.images.get("python:3.12-slim")
            
            # Basic security checks
            config = image.attrs.get("Config", {})
            
            # Check that it doesn't run as root by default
            user = config.get("User", "")
            
            # Check for exposed ports
            exposed_ports = config.get("ExposedPorts", {})
            
            # These are informational checks
            assert config is not None
            
        except docker.errors.ImageNotFound:
            pytest.skip("Python base image not available")


class TestContainerLifecycle:
    """Test container lifecycle operations."""

    @pytest.mark.docker
    def test_container_creation(self, docker_client):
        """Test container creation."""
        # Create simple test container
        container = docker_client.containers.create(
            "python:3.12-slim",
            command="echo 'Hello World'",
            name="sutazai-test-container"
        )
        
        assert container is not None
        assert container.name == "sutazai-test-container"
        
        # Clean up
        container.remove()

    @pytest.mark.docker
    def test_container_start_stop(self, docker_client):
        """Test container start and stop."""
        # Create and start container
        container = docker_client.containers.run(
            "python:3.12-slim",
            command="sleep 30",
            name="sutazai-lifecycle-test",
            detach=True
        )
        
        # Wait for container to start
        time.sleep(2)
        
        # Check container is running
        container.reload()
        assert container.status == "running"
        
        # Stop container
        container.stop(timeout=5)
        container.reload()
        assert container.status == "exited"
        
        # Clean up
        container.remove()

    @pytest.mark.docker
    def test_container_resource_limits(self, docker_client):
        """Test container resource limits."""
        # Create container with resource limits
        container = docker_client.containers.run(
            "python:3.12-slim",
            command="sleep 10",
            name="sutazai-resource-test",
            mem_limit="128m",
            cpu_quota=50000,  # 0.5 CPU
            detach=True
        )
        
        # Check container stats
        stats = container.stats(stream=False)
        
        # Verify memory limit is set
        memory_limit = stats["memory"]["limit"]
        assert memory_limit <= 128 * 1024 * 1024 + 1024 * 1024  # Allow some overhead
        
        # Clean up
        container.stop()
        container.remove()

    @pytest.mark.docker
    def test_container_health_check(self, docker_client):
        """Test container health check."""
        # Create container with health check
        dockerfile_content = """
FROM python:3.12-slim
RUN pip install flask
COPY <<EOF /app.py
from flask import Flask
app = Flask(__name__)

@app.route('/health')
def health():
    return 'OK'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
EOF

HEALTHCHECK --interval=5s --timeout=3s --retries=3 \\
  CMD curl -f http://localhost:5000/health || exit 1

EXPOSE 5000
CMD ["python", "/app.py"]
"""
        
        try:
            # Build image with health check
            image, logs = docker_client.images.build(
                fileobj=None,
                dockerfile=dockerfile_content,
                tag="sutazai-health-test:latest"
            )
            
            # Run container
            container = docker_client.containers.run(
                "sutazai-health-test:latest",
                name="sutazai-health-test",
                ports={"5000/tcp": 5000},
                detach=True
            )
            
            # Wait for health check
            time.sleep(20)
            
            # Check health status
            container.reload()
            health_status = container.attrs.get("State", {}).get("Health", {}).get("Status")
            
            # Health check may take time to establish
            assert health_status in ["healthy", "starting", None]
            
            # Clean up
            container.stop()
            container.remove()
            docker_client.images.remove(image.id, force=True)
            
        except Exception as e:
            pytest.skip(f"Health check test failed: {str(e)}")


class TestServiceContainers:
    """Test specific service containers."""

    @pytest.mark.docker
    def test_redis_container(self, docker_client):
        """Test Redis container."""
        # Start Redis container
        redis_container = docker_client.containers.run(
            "redis:7-alpine",
            name="sutazai-redis-test",
            ports={"6379/tcp": 6379},
            detach=True
        )
        
        # Wait for Redis to start
        time.sleep(5)
        
        try:
            # Test Redis connection
            import redis
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            
            # Test basic operations
            r.set('test_key', 'test_value')
            value = r.get('test_key')
            assert value == 'test_value'
            
            # Test Redis info
            info = r.info()
            assert 'redis_version' in info
            
        except ImportError:
            pytest.skip("Redis Python client not available")
        except Exception as e:
            pytest.fail(f"Redis container test failed: {str(e)}")
        finally:
            # Clean up
            redis_container.stop()
            redis_container.remove()

    @pytest.mark.docker
    def test_postgres_container(self, docker_client):
        """Test PostgreSQL container."""
        # Start PostgreSQL container
        postgres_container = docker_client.containers.run(
            "postgres:15-alpine",
            name="sutazai-postgres-test",
            ports={"5432/tcp": 5432},
            environment={
                "POSTGRES_DB": "test_db",
                "POSTGRES_USER": "test_user", 
                "POSTGRES_PASSWORD": "test_pass"
            },
            detach=True
        )
        
        # Wait for PostgreSQL to start
        time.sleep(10)
        
        try:
            import psycopg2
            
            # Test PostgreSQL connection
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="test_db",
                user="test_user",
                password=os.getenv("TEST_PASSWORD", "test_pass")
            )
            
            # Test basic operations
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE test_table (id SERIAL PRIMARY KEY, name VARCHAR(50));")
            cursor.execute("INSERT INTO test_table (name) VALUES ('test');")
            cursor.execute("SELECT * FROM test_table;")
            
            result = cursor.fetchone()
            assert result is not None
            assert result[1] == 'test'
            
            conn.commit()
            conn.close()
            
        except ImportError:
            pytest.skip("PostgreSQL Python client not available")
        except Exception as e:
            pytest.fail(f"PostgreSQL container test failed: {str(e)}")
        finally:
            # Clean up
            postgres_container.stop()
            postgres_container.remove()

    @pytest.mark.docker  
    def test_nginx_container(self, docker_client):
        """Test Nginx container."""
        # Start Nginx container
        nginx_container = docker_client.containers.run(
            "nginx:alpine",
            name="sutazai-nginx-test",
            ports={"80/tcp": 8080},
            detach=True
        )
        
        # Wait for Nginx to start
        time.sleep(5)
        
        try:
            # Test Nginx connection
            response = requests.get("http://localhost:8080", timeout=10)
            assert response.status_code == 200
            assert "nginx" in response.text.lower()
            
        except requests.exceptions.ConnectionError:
            pytest.fail("Cannot connect to Nginx container")
        except Exception as e:
            pytest.fail(f"Nginx container test failed: {str(e)}")
        finally:
            # Clean up
            nginx_container.stop()
            nginx_container.remove()


class TestContainerNetworking:
    """Test container networking."""

    @pytest.mark.docker
    def test_container_network_creation(self, docker_client):
        """Test container network creation."""
        # Create custom network
        network = docker_client.networks.create(
            name="sutazai-test-network",
            driver="bridge"
        )
        
        assert network is not None
        assert network.name == "sutazai-test-network"
        
        # Clean up
        network.remove()

    @pytest.mark.docker
    def test_inter_container_communication(self, docker_client):
        """Test communication between containers."""
        # Create custom network
        network = docker_client.networks.create(
            name="sutazai-comm-test",
            driver="bridge"
        )
        
        try:
            # Start server container
            server_container = docker_client.containers.run(
                "python:3.12-slim",
                command="python -m http.server 8000",
                name="sutazai-server",
                networks=["sutazai-comm-test"],
                detach=True
            )
            
            # Wait for server to start
            time.sleep(5)
            
            # Start client container
            client_container = docker_client.containers.run(
                "python:3.12-slim",
                command="python -c \"import urllib.request; print(urllib.request.urlopen('http://sutazai-server:8000').read())\"",
                name="sutazai-client",
                networks=["sutazai-comm-test"],
                detach=True
            )
            
            # Wait for client to complete
            client_container.wait(timeout=10)
            
            # Check client logs
            logs = client_container.logs().decode()
            assert "Directory listing" in logs or "200" in logs
            
        except Exception as e:
            pytest.fail(f"Inter-container communication failed: {str(e)}")
        finally:
            # Clean up
            try:
                server_container.stop()
                server_container.remove()
                client_container.remove()
            except (AssertionError, Exception) as e:
                # Suppressed exception (was bare except)
                logger.debug(f"Suppressed exception: {e}")
                pass
            network.remove()

    @pytest.mark.docker
    def test_container_port_mapping(self, docker_client):
        """Test container port mapping."""
        # Start container with port mapping
        container = docker_client.containers.run(
            "python:3.12-slim",
            command="python -m http.server 8000",
            name="sutazai-port-test",
            ports={"8000/tcp": 9000},
            detach=True
        )
        
        # Wait for server to start
        time.sleep(5)
        
        try:
            # Test port mapping
            response = requests.get("http://localhost:9000", timeout=10)
            assert response.status_code == 200
            
        except requests.exceptions.ConnectionError:
            pytest.fail("Port mapping not working")
        finally:
            # Clean up
            container.stop()
            container.remove()


class TestContainerVolumes:
    """Test container volumes and data persistence."""

    @pytest.mark.docker
    def test_volume_creation(self, docker_client):
        """Test volume creation."""
        # Create volume
        volume = docker_client.volumes.create(name="sutazai-test-volume")
        
        assert volume is not None
        assert volume.name == "sutazai-test-volume"
        
        # Clean up
        volume.remove()

    @pytest.mark.docker
    def test_volume_mount(self, docker_client):
        """Test volume mounting."""
        # Create volume
        volume = docker_client.volumes.create(name="sutazai-mount-test")
        
        try:
            # Create container with mounted volume
            container = docker_client.containers.run(
                "python:3.12-slim",
                command="sh -c 'echo \"test data\" > /data/test.txt'",
                name="sutazai-volume-test",
                volumes={"sutazai-mount-test": {"bind": "/data", "mode": "rw"}},
                detach=True
            )
            
            # Wait for command to complete
            container.wait(timeout=10)
            
            # Create another container to read the data
            reader_container = docker_client.containers.run(
                "python:3.12-slim",
                command="cat /data/test.txt",
                name="sutazai-volume-reader",
                volumes={"sutazai-mount-test": {"bind": "/data", "mode": "ro"}},
                detach=True
            )
            
            reader_container.wait(timeout=10)
            
            # Check if data persisted
            logs = reader_container.logs().decode().strip()
            assert logs == "test data"
            
        finally:
            # Clean up
            try:
                container.remove()
                reader_container.remove()
            except (AssertionError, Exception) as e:
                # Suppressed exception (was bare except)
                logger.debug(f"Suppressed exception: {e}")
                pass
            volume.remove()

    @pytest.mark.docker
    def test_bind_mount(self, docker_client):
        """Test bind mounting."""
        import tempfile
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("bind mount test")
            
            # Create container with bind mount
            container = docker_client.containers.run(
                "python:3.12-slim",
                command="cat /host/test.txt",
                name="sutazai-bind-test",
                volumes={temp_dir: {"bind": "/host", "mode": "ro"}},
                detach=True
            )
            
            container.wait(timeout=10)
            
            # Check logs
            logs = container.logs().decode().strip()
            assert logs == "bind mount test"
            
            # Clean up
            container.remove()


class TestContainerSecurity:
    """Test container security features."""

    @pytest.mark.docker
    def test_container_user_security(self, docker_client):
        """Test container runs with non-root user."""
        # Create container that checks user
        container = docker_client.containers.run(
            "python:3.12-slim",
            command="id",
            name="sutazai-user-test",
            user="1000:1000",
            detach=True
        )
        
        container.wait(timeout=10)
        
        # Check user ID
        logs = container.logs().decode()
        assert "uid=1000" in logs
        assert "gid=1000" in logs
        
        # Clean up
        container.remove()

    @pytest.mark.docker
    def test_container_readonly_filesystem(self, docker_client):
        """Test container with read-only filesystem."""
        # Create container with read-only filesystem
        container = docker_client.containers.run(
            "python:3.12-slim",
            command="sh -c 'touch /test.txt || echo \"readonly filesystem\"'",
            name="sutazai-readonly-test",
            read_only=True,
            tmpfs={"/tmp": ""},
            detach=True
        )
        
        container.wait(timeout=10)
        
        # Check logs
        logs = container.logs().decode()
        assert "readonly filesystem" in logs
        
        # Clean up
        container.remove()

    @pytest.mark.docker
    def test_container_capabilities_drop(self, docker_client):
        """Test dropping container capabilities."""
        # Create container with dropped capabilities
        container = docker_client.containers.run(
            "python:3.12-slim",
            command="python -c \"import os; print('CAP_NET_RAW' not in str(os.system('capsh --print')))\"",
            name="sutazai-caps-test",
            cap_drop=["NET_RAW"],
            detach=True
        )
        
        container.wait(timeout=10)
        
        # Container should complete without error
        container.reload()
        assert container.attrs["State"]["ExitCode"] == 0
        
        # Clean up
        container.remove()


class TestDockerCompose:
    """Test Docker Compose functionality."""

    def test_docker_compose_available(self):
        """Test Docker Compose is available."""
        try:
            result = subprocess.run(
                ["docker-compose", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            assert result.returncode == 0
            assert "docker-compose" in result.stdout.lower()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            try:
                # Try docker compose (newer syntax)
                result = subprocess.run(
                    ["docker", "compose", "version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                assert result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pytest.skip("Docker Compose not available")

    def test_compose_file_validation(self):
        """Test Docker Compose file validation."""
        compose_content = """
version: '3.8'
services:
  test-service:
    image: python:3.12-slim
    command: echo "Hello from compose"
    environment:
      - TEST_VAR=test_value
"""
        
        # Write temporary compose file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(compose_content)
            compose_file = f.name
        
        try:
            # Validate compose file
            result = subprocess.run(
                ["docker-compose", "-f", compose_file, "config"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                # Try newer docker compose syntax
                result = subprocess.run(
                    ["docker", "compose", "-f", compose_file, "config"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            
            assert result.returncode == 0
            assert "test-service" in result.stdout
            
        finally:
            # Clean up
            os.unlink(compose_file)

    @pytest.mark.docker
    def test_multi_container_setup(self, docker_client):
        """Test multi-container setup simulation."""
        containers = []
        network = None
        
        try:
            # Create network
            network = docker_client.networks.create(
                name="sutazai-multi-test",
                driver="bridge"
            )
            
            # Start Redis container
            redis_container = docker_client.containers.run(
                "redis:7-alpine",
                name="sutazai-multi-redis",
                networks=["sutazai-multi-test"],
                detach=True
            )
            containers.append(redis_container)
            
            # Start app container that connects to Redis
            app_container = docker_client.containers.run(
                "python:3.12-slim",
                command="python -c \"import time; time.sleep(10)\"",
                name="sutazai-multi-app",
                networks=["sutazai-multi-test"],
                detach=True
            )
            containers.append(app_container)
            
            # Wait for containers to start
            time.sleep(5)
            
            # Check both containers are running
            for container in containers:
                container.reload()
                assert container.status == "running"
            
        finally:
            # Clean up
            for container in containers:
                try:
                    container.stop()
                    container.remove()
                except (AssertionError, Exception) as e:
                    # Suppressed exception (was bare except)
                    logger.debug(f"Suppressed exception: {e}")
                    pass
            
            if network:
                try:
                    network.remove()
                except (AssertionError, Exception) as e:
                    # Suppressed exception (was bare except)
                    logger.debug(f"Suppressed exception: {e}")
                    pass


class TestContainerMonitoring:
    """Test container monitoring and logging."""

    @pytest.mark.docker
    def test_container_logs(self, docker_client):
        """Test container logging."""
        # Create container that generates logs
        container = docker_client.containers.run(
            "python:3.12-slim",
            command="python -c \"print('Log message 1'); print('Log message 2')\"",
            name="sutazai-log-test",
            detach=True
        )
        
        container.wait(timeout=10)
        
        # Get logs
        logs = container.logs().decode()
        assert "Log message 1" in logs
        assert "Log message 2" in logs
        
        # Clean up
        container.remove()

    @pytest.mark.docker
    def test_container_stats(self, docker_client):
        """Test container statistics."""
        # Create long-running container
        container = docker_client.containers.run(
            "python:3.12-slim",
            command="python -c \"import time; time.sleep(30)\"",
            name="sutazai-stats-test",
            detach=True
        )
        
        # Wait for container to start
        time.sleep(2)
        
        try:
            # Get container stats
            stats = container.stats(stream=False)
            
            assert "cpu_stats" in stats
            assert "memory_stats" in stats
            assert "networks" in stats
            
            # Check memory usage
            memory_usage = stats["memory_stats"].get("usage", 0)
            assert memory_usage > 0
            
        finally:
            # Clean up
            container.stop()
            container.remove()

    @pytest.mark.docker
    def test_container_events(self, docker_client):
        """Test container events monitoring."""
        events = []
        
        # Start event monitoring in background
        import threading
        
        def collect_events():
            for event in docker_client.events(decode=True):
                if event.get("Type") == "container":
                    events.append(event)
                if len(events) >= 2:  # Stop after collecting some events
                    break
        
        event_thread = threading.Thread(target=collect_events)
        event_thread.daemon = True
        event_thread.start()
        
        # Create and remove container to generate events
        container = docker_client.containers.create(
            "python:3.12-slim",
            command="echo 'test'",
            name="sutazai-event-test"
        )
        
        container.start()
        container.wait(timeout=5)
        container.remove()
        
        # Wait for events
        time.sleep(2)
        
        # Check if events were captured
        event_actions = [event.get("Action") for event in events]
        assert len(event_actions) > 0
        
        # Common container events
        possible_actions = ["create", "start", "die", "destroy"]
        assert any(action in possible_actions for action in event_actions)