#!/usr/bin/env python3
"""
Purpose: Test fixtures and mock violation creation for hygiene system testing
Usage: python -m pytest tests/hygiene/test_fixtures.py
Requirements: pytest, tempfile
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import unittest
import tempfile
import shutil
from pathlib import Path
import json
import os

class TestFixtureCreation(unittest.TestCase):
    """Test creation of test fixtures for violation testing"""
    
    def setUp(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.fixtures_dir = self.temp_dir / "fixtures"
        self.fixtures_dir.mkdir()
        
    def tearDown(self):
        """Cleanup test environment"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            
    def create_rule_13_violations(self) -> Path:
        """Create Rule 13 violation fixtures (junk files)"""
        rule_13_dir = self.fixtures_dir / "rule_13_junk_files"
        rule_13_dir.mkdir()
        
        # Create various types of junk files
        junk_files = {
            "backup_file.backup": "Original file backup content",
            "old_script.py.bak": "#!/usr/bin/env python3\nprint('old script')",
            "temp_data.tmp": "temporary data for processing",
            "editor_save~": "editor backup file content",
            "config.old": "old configuration settings",
            "document.backup": "document backup content",
            "file.orig": "original file before changes"
        }
        
        for filename, content in junk_files.items():
            filepath = rule_13_dir / filename
            filepath.write_text(content)
            
        return rule_13_dir
        
    def create_rule_12_violations(self) -> Path:
        """Create Rule 12 violation fixtures (multiple deployment scripts)"""
        rule_12_dir = self.fixtures_dir / "rule_12_deploy_scripts"
        rule_12_dir.mkdir()
        
        # Create multiple deployment scripts
        deploy_scripts = {
            "deploy.sh": """#!/bin/bash
echo "Main deployment script"
docker-compose up -d
""",
            "deploy_staging.sh": """#!/bin/bash
echo "Staging deployment"
export ENV=staging
docker-compose -f docker-compose.staging.yml up -d
""",
            "deploy_production.py": """#!/usr/bin/env python3
print("Production deployment script")
import subprocess
subprocess.run(["docker-compose", "up", "-d"])
""",
            "validate_deployment.py": """#!/usr/bin/env python3
import requests
import sys

def validate_deployment():
    try:
        response = requests.get("http://localhost:10206/health")
        return response.status_code == 200
    except (AssertionError, Exception) as e:
        logger.warning(f"Exception caught, returning: {e}")
        return False

if __name__ == "__main__":
    if validate_deployment():
        print("Deployment validation passed")
        sys.exit(0)
    else:
        print("Deployment validation failed")
        sys.exit(1)
""",
            "old_deploy.sh": """#!/bin/bash
# Old deployment script - should be consolidated
echo "Legacy deployment"
"""
        }
        
        for filename, content in deploy_scripts.items():
            filepath = rule_12_dir / filename
            filepath.write_text(content)
            filepath.chmod(0o755)  # Make executable
            
        return rule_12_dir
        
    def create_rule_8_violations(self) -> Path:
        """Create Rule 8 violation fixtures (undocumented Python scripts)"""
        rule_8_dir = self.fixtures_dir / "rule_8_python_scripts"
        rule_8_dir.mkdir()
        
        # Create Python scripts without proper documentation
        undocumented_scripts = {
            "data_processor.py": """import json
import sys

def process_data(input_file, output_file):
    with open(input_file) as f:
        data = json.load(f)
    
    processed = [item for item in data if item.get('active')]
    
    with open(output_file, 'w') as f:
        json.dump(processed, f, indent=2)

if __name__ == "__main__":
    process_data(sys.argv[1], sys.argv[2])
""",
            "utility_functions.py": """def calculate_metrics(data):
    return {
        'count': len(data),
        'average': sum(data) / len(data) if data else 0
    }

def format_output(metrics):
    return f"Count: {metrics['count']}, Average: {metrics['average']:.2f}"
""",
            "batch_operations.py": """import os
import shutil

for file in os.listdir('.'):
    if file.endswith('.tmp'):
        shutil.move(file, 'archive/' + file)
        
print("Batch operations completed")
""",
            "api_client.py": """import requests

class APIClient:
    def __init__(self, base_url):
        self.base_url = base_url
        
    def get_data(self, endpoint):
        response = requests.get(f"{self.base_url}/{endpoint}")
        return response.json()
"""
        }
        
        for filename, content in undocumented_scripts.items():
            filepath = rule_8_dir / filename
            filepath.write_text(content)
            
        return rule_8_dir
        
    def create_rule_11_violations(self) -> Path:
        """Create Rule 11 violation fixtures (Docker structure chaos)"""
        rule_11_dir = self.fixtures_dir / "rule_11_docker_chaos"
        rule_11_dir.mkdir()
        
        # Create disorganized Docker files
        docker_files = {
            "Dockerfile": """FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install -y python3
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]
""",
            "Dockerfile.old": """FROM ubuntu:18.04
RUN apt-get update && apt-get install -y python3
COPY . /app
WORKDIR /app
CMD ["python3", "old_app.py"]
""",
            "Dockerfile.backup": """FROM ubuntu:20.04
# Backup of main Dockerfile
RUN apt-get update
RUN apt-get install -y python3 nodejs
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]
""",
            "docker-compose.yml": """version: '3.8'
services:
  web:
    build: .
    ports:
      - "8080:8080"
  db:
    image: postgres:13
    environment:
      POSTGRES_PASSWORD: password
""",
            "docker-compose.backup.yml": """version: '3.8'
services:
  web:
    build: .
    ports:
      - "8080:8080"
  db:
    image: postgres:12
    environment:
      POSTGRES_PASSWORD: password
""",
            "docker-compose.staging.yml": """version: '3.8'
services:
  web:
    build: .
    ports:
      - "8081:8080"
  db:
    image: postgres:13
    environment:
      POSTGRES_PASSWORD: staging_password
""",
            ".dockerignore": """node_modules
*.log
.git
""",
            ".dockerignore.backup": """node_modules
*.log
.git
*.tmp
"""
        }
        
        for filename, content in docker_files.items():
            filepath = rule_11_dir / filename
            filepath.write_text(content)
            
        return rule_11_dir
        
    def test_create_rule_13_fixtures(self):
        """Test creation of Rule 13 violation fixtures"""
        rule_13_dir = self.create_rule_13_violations()
        
        # Verify fixtures were created
        self.assertTrue(rule_13_dir.exists())
        
        # Check for expected junk files
        expected_files = [
            "backup_file.backup",
            "old_script.py.bak", 
            "temp_data.tmp",
            "editor_save~",
            "config.old"
        ]
        
        for filename in expected_files:
            filepath = rule_13_dir / filename
            self.assertTrue(filepath.exists(), f"Missing fixture: {filename}")
            self.assertGreater(len(filepath.read_text()), 0, 
                             f"Fixture {filename} should have content")
                             
    def test_create_rule_12_fixtures(self):
        """Test creation of Rule 12 violation fixtures"""
        rule_12_dir = self.create_rule_12_violations()
        
        # Verify fixtures were created
        self.assertTrue(rule_12_dir.exists())
        
        # Check for deployment scripts
        expected_scripts = [
            "deploy.sh",
            "deploy_staging.sh",
            "deploy_production.py",
            "validate_deployment.py"
        ]
        
        for script in expected_scripts:
            script_path = rule_12_dir / script
            self.assertTrue(script_path.exists(), f"Missing script: {script}")
            self.assertTrue(os.access(script_path, os.X_OK), 
                           f"Script {script} should be executable")
                           
    def test_create_rule_8_fixtures(self):
        """Test creation of Rule 8 violation fixtures"""
        rule_8_dir = self.create_rule_8_violations()
        
        # Verify fixtures were created
        self.assertTrue(rule_8_dir.exists())
        
        # Check for Python scripts
        expected_scripts = [
            "data_processor.py",
            "utility_functions.py",
            "batch_operations.py",
            "api_client.py"
        ]
        
        for script in expected_scripts:
            script_path = rule_8_dir / script
            self.assertTrue(script_path.exists(), f"Missing script: {script}")
            
            # Verify it's valid Python
            content = script_path.read_text()
            self.assertIn("def " or "import " or "class ", content,
                         f"Script {script} should contain Python code")
                         
    def test_create_rule_11_fixtures(self):
        """Test creation of Rule 11 violation fixtures"""
        rule_11_dir = self.create_rule_11_violations()
        
        # Verify fixtures were created
        self.assertTrue(rule_11_dir.exists())
        
        # Check for Docker files
        expected_files = [
            "Dockerfile",
            "Dockerfile.old",
            "docker-compose.yml",
            "docker-compose.backup.yml"
        ]
        
        for filename in expected_files:
            filepath = rule_11_dir / filename
            self.assertTrue(filepath.exists(), f"Missing Docker file: {filename}")
            
            # Verify Docker file structure
            content = filepath.read_text()
            if filename.startswith("Dockerfile"):
                self.assertIn("FROM", content, f"{filename} should have FROM instruction")
            elif filename.startswith("docker-compose"):
                self.assertIn("version:", content, f"{filename} should have version")

class TestMockViolationDetection(unittest.TestCase):
    """Test mock violation detection with fixtures"""
    
    def setUp(self):
        """Setup test environment with fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_project = self.temp_dir / "test_project"
        self.test_project.mkdir()
        
        # Create fixture creator
        self.fixture_creator = TestFixtureCreation()
        self.fixture_creator.temp_dir = self.temp_dir
        self.fixture_creator.fixtures_dir = self.temp_dir / "fixtures"
        self.fixture_creator.fixtures_dir.mkdir()
        
    def tearDown(self):
        """Cleanup test environment"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            
    def test_detect_rule_13_violations(self):
        """Test detection of Rule 13 violations using fixtures"""
        # Create fixtures
        rule_13_dir = self.fixture_creator.create_rule_13_violations()
        
        # Simulate violation detection
        violations = []
        patterns = ["*.backup", "*.bak", "*.tmp", "*~", "*.old"]
        
        for pattern in patterns:
            violations.extend(list(rule_13_dir.glob(pattern)))
            
        # Should detect violations
        self.assertGreater(len(violations), 0, "Should detect Rule 13 violations")
        
        # Verify specific violations
        violation_names = [v.name for v in violations]
        self.assertIn("backup_file.backup", violation_names)
        self.assertIn("old_script.py.bak", violation_names)
        self.assertIn("temp_data.tmp", violation_names)
        
    def test_detect_rule_12_violations(self):
        """Test detection of Rule 12 violations using fixtures"""
        # Create fixtures
        rule_12_dir = self.fixture_creator.create_rule_12_violations()
        
        # Simulate deployment script detection
        deploy_scripts = []
        patterns = ["deploy*.sh", "*deploy*.py", "validate*deploy*"]
        
        for pattern in patterns:
            deploy_scripts.extend(list(rule_12_dir.glob(pattern)))
            
        # Should detect multiple deployment scripts
        self.assertGreater(len(deploy_scripts), 1, 
                          "Should detect multiple deployment scripts")
                          
    def test_detect_rule_8_violations(self):
        """Test detection of Rule 8 violations using fixtures"""
        # Create fixtures  
        rule_8_dir = self.fixture_creator.create_rule_8_violations()
        
        # Simulate Python script detection
        python_scripts = list(rule_8_dir.glob("*.py"))
        
        # Check for missing documentation headers
        undocumented_scripts = []
        for script in python_scripts:
            content = script.read_text()
            
            # Check if script lacks proper header documentation
            has_purpose = "Purpose:" in content
            has_usage = "Usage:" in content
            has_requirements = "Requirements:" in content
            
            if not (has_purpose and has_usage and has_requirements):
                undocumented_scripts.append(script)
                
        # Should detect undocumented scripts
        self.assertGreater(len(undocumented_scripts), 0,
                          "Should detect undocumented Python scripts")

class TestFixtureManagement(unittest.TestCase):
    """Test fixture management and cleanup"""
    
    def setUp(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """Cleanup test environment"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            
    def test_fixture_metadata_creation(self):
        """Test creation of fixture metadata"""
        fixtures_dir = self.temp_dir / "fixtures"
        fixtures_dir.mkdir()
        
        # Create metadata file
        metadata = {
            "created_at": "2024-01-01T00:00:00Z",
            "purpose": "Test fixtures for hygiene system validation",
            "fixtures": {
                "rule_13": {
                    "description": "Junk files and backup violations",
                    "file_count": 7,
                    "patterns": ["*.backup", "*.bak", "*.tmp", "*~", "*.old"]
                },
                "rule_12": {
                    "description": "Multiple deployment scripts",
                    "file_count": 5,
                    "patterns": ["deploy*.sh", "*deploy*.py"]
                }
            }
        }
        
        metadata_file = fixtures_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))
        
        # Verify metadata file
        self.assertTrue(metadata_file.exists())
        
        # Verify metadata content
        loaded_metadata = json.loads(metadata_file.read_text())
        self.assertEqual(loaded_metadata["purpose"], metadata["purpose"])
        self.assertIn("rule_13", loaded_metadata["fixtures"])
        
    def test_fixture_cleanup_simulation(self):
        """Test fixture cleanup simulation"""
        fixtures_dir = self.temp_dir / "fixtures"
        fixtures_dir.mkdir()
        
        # Create test fixture files
        test_files = [
            "test.backup",
            "old.tmp", 
            "script.py.bak"
        ]
        
        for filename in test_files:
            (fixtures_dir / filename).write_text("test content")
            
        # Simulate cleanup (archive and remove)
        archive_dir = self.temp_dir / "archive"
        archive_dir.mkdir()
        
        cleaned_files = []
        for test_file in fixtures_dir.iterdir():
            if test_file.is_file() and test_file.name != "metadata.json":
                # Archive file
                archive_path = archive_dir / test_file.name
                archive_path.write_text(test_file.read_text())
                
                # Remove original
                test_file.unlink()
                cleaned_files.append(test_file.name)
                
        # Verify cleanup
        self.assertEqual(len(cleaned_files), len(test_files))
        self.assertEqual(len(list(fixtures_dir.iterdir())), 0)
        self.assertEqual(len(list(archive_dir.iterdir())), len(test_files))

if __name__ == "__main__":
    unittest.main()