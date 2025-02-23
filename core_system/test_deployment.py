import json
import os
import subprocess
import sys

import pytest
import requests
from dotenv import load_dotenv

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

class TestDeployment:
    @pytest.fixture(scope="class")
    def env_config(self):
        """Validate environment configuration"""
        required_vars = [
            "DB_HOST", "DB_PORT", "DB_USER", "DB_NAME",
            "REDIS_HOST", "REDIS_PORT", 
            "SECRET_KEY", "PROJECT_NAME"
        ]
        
        for var in required_vars:
            assert os.getenv(var), f"Environment variable {var} is not set"
        
        return {var: os.getenv(var) for var in required_vars}

    def test_docker_compose_syntax(self):
        """Validate docker-compose file syntax"""
        try:
            result = subprocess.run(
                ['docker-compose', 'config'], 
                capture_output=True, 
                text=True, 
                check=True
            )
            assert result.returncode == 0, "Docker Compose configuration is invalid"
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Docker Compose config validation failed: {e.stderr}")

    def test_environment_file_security(self):
        """Check .env file permissions"""
        env_path = '.env'
        assert os.path.exists(env_path), ".env file does not exist"
        
        # Check file permissions (should be readable only by owner)
        mode = os.stat(env_path).st_mode
        assert oct(mode & 0o777) == '0o600', "Insecure .env file permissions"

    def test_required_services_ports(self):
        """Verify critical service ports are available"""
        test_ports = {
            "postgres": 5432,
            "redis": 6379,
            "application": 8000
        }
        
        for service, port in test_ports.items():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                assert result == 0, f"{service.capitalize()} port {port} is not available"
            except Exception as e:
                pytest.fail(f"Error checking {service} port: {e}")

    def test_docker_build(self):
        """Test Docker image build process"""
        try:
            result = subprocess.run(
                ['docker', 'build', '-t', 'sutazai_test', '.'], 
                capture_output=True, 
                text=True
            )
            assert result.returncode == 0, f"Docker build failed: {result.stderr}"
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Docker build error: {e}")
        finally:
            # Clean up test image
            subprocess.run(['docker', 'rmi', 'sutazai_test'], capture_output=True)

    def test_dependency_installation(self):
        """Verify dependencies can be installed"""
        try:
            result = subprocess.run(
                ['pip', 'install', '-r', 'requirements-prod.txt'], 
                capture_output=True, 
                text=True,
                check=True
            )
            assert result.returncode == 0, "Dependency installation failed"
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Dependency installation error: {e.stderr}")

    def test_health_endpoint(self, env_config):
        """Test application health endpoint"""
        try:
            response = requests.get(
                f"http://localhost:{env_config.get('PORT', 8000)}/health", 
                timeout=5
            )
            assert response.status_code == 200, "Health endpoint not responding"
            
            # Optional: Check health response content
            health_data = response.json()
            assert isinstance(health_data, dict), "Invalid health endpoint response"
            assert health_data.get('status') == 'healthy', "Application not in healthy state"
        
        except requests.RequestException as e:
            pytest.fail(f"Health check failed: {e}")

    def test_configuration_validation(self, env_config):
        """Validate critical configuration parameters"""
        # Check database configuration
        assert env_config['DB_HOST'] != 'db', "Default database host not replaced"
        assert len(env_config['SECRET_KEY']) >= 32, "Weak secret key"
        
        # Check project naming
        assert env_config['PROJECT_NAME'] == 'SutazAI', "Incorrect project name"

def pytest_configure(config):
    """Custom pytest configuration"""
    config.addinivalue_line(
        "markers", 
        "deployment: mark test as a deployment validation test"
    )

if __name__ == '__main__':
    pytest.main([__file__]) 