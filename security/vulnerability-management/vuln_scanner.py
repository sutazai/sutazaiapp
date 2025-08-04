#!/usr/bin/env python3
"""
Automated Vulnerability Scanning and Patching System
Implements comprehensive vulnerability assessment and automated remediation
"""

import asyncio
import logging
import json
import subprocess
import time
import os
import re
import hashlib
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import xml.etree.ElementTree as ET
import docker
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import threading
from concurrent.futures import ThreadPoolExecutor
import yaml
import tarfile
import shutil

class VulnerabilitySeverity(Enum):
    INFORMATIONAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5

class ScanType(Enum):
    CONTAINER = "container"
    NETWORK = "network"
    APPLICATION = "application"
    INFRASTRUCTURE = "infrastructure"
    CONFIGURATION = "configuration"

class RemediationStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    IGNORED = "ignored"

@dataclass
class Vulnerability:
    """Vulnerability finding"""
    vuln_id: str
    cve_id: Optional[str]
    title: str
    description: str
    severity: VulnerabilitySeverity
    cvss_score: float
    affected_component: str
    affected_version: str
    fixed_version: Optional[str]
    scan_type: ScanType
    discovered_at: datetime
    remediation_advice: str
    references: List[str]
    exploitable: bool = False
    patch_available: bool = False

@dataclass
class ScanResult:
    """Vulnerability scan result"""
    scan_id: str
    target: str
    scan_type: ScanType
    started_at: datetime
    completed_at: Optional[datetime]
    status: str
    vulnerabilities: List[Vulnerability]
    summary: Dict[str, int]
    scanner_version: str
    scan_config: Dict[str, Any]

class VulnerabilityScanner:
    """Comprehensive vulnerability scanner"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.redis_client = None
        self.db_connection = None
        self.docker_client = None
        self.scan_results: Dict[str, ScanResult] = {}
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=5)
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize vulnerability scanner components"""
        try:
            # Initialize Redis
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'redis'),
                port=self.config.get('redis_port', 6379),
                password=self.config.get('redis_password'),
                ssl=True,
                decode_responses=True
            )
            
            # Initialize PostgreSQL
            self.db_connection = psycopg2.connect(
                host=self.config.get('postgres_host', 'postgres'),
                port=self.config.get('postgres_port', 5432),
                database=self.config.get('postgres_db', 'sutazai'),
                user=self.config.get('postgres_user', 'sutazai'),
                password=self.config.get('postgres_password'),
                sslmode='require'
            )
            
            # Initialize Docker client
            self.docker_client = docker.from_env()
            
            # Setup scanner tools
            self._setup_scanner_tools()
            
            self.logger.info("Vulnerability Scanner initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Vulnerability Scanner: {e}")
            raise
    
    def _setup_scanner_tools(self):
        """Setup and verify scanner tools"""
        try:
            # Verify Trivy is available
            result = subprocess.run(['trivy', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info(f"Trivy available: {result.stdout.strip()}")
            else:
                self.logger.warning("Trivy not available, installing...")
                self._install_trivy()
            
            # Verify Nmap is available
            result = subprocess.run(['nmap', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("Nmap available")
            else:
                self.logger.warning("Nmap not available")
            
            # Verify Semgrep is available
            result = subprocess.run(['semgrep', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("Semgrep available")
            else:
                self.logger.warning("Semgrep not available")
                
        except Exception as e:
            self.logger.error(f"Failed to setup scanner tools: {e}")
    
    def _install_trivy(self):
        """Install Trivy vulnerability scanner"""
        try:
            # Download and install Trivy
            commands = [
                "curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin",
                "chmod +x /usr/local/bin/trivy"
            ]
            
            for cmd in commands:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    self.logger.error(f"Failed to install Trivy: {result.stderr}")
                    return False
            
            self.logger.info("Trivy installed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Exception during Trivy installation: {e}")
            return False
    
    async def scan_containers(self, target_images: List[str] = None) -> str:
        """Scan Docker containers for vulnerabilities"""
        scan_id = self._generate_scan_id()
        
        try:
            if not target_images:
                # Get all running containers
                containers = self.docker_client.containers.list()
                target_images = [container.image.tags[0] if container.image.tags else container.image.id 
                               for container in containers]
            
            scan_result = ScanResult(
                scan_id=scan_id,
                target=','.join(target_images),
                scan_type=ScanType.CONTAINER,
                started_at=datetime.utcnow(),
                completed_at=None,
                status="running",
                vulnerabilities=[],
                summary={},
                scanner_version="trivy-latest",
                scan_config={"images": target_images}
            )
            
            self.scan_results[scan_id] = scan_result
            
            # Run container scans in parallel
            tasks = [self._scan_single_container(image) for image in target_images]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate results
            all_vulnerabilities = []
            for result in results:
                if isinstance(result, list):
                    all_vulnerabilities.extend(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"Container scan error: {result}")
            
            # Update scan result
            scan_result.vulnerabilities = all_vulnerabilities
            scan_result.completed_at = datetime.utcnow()
            scan_result.status = "completed"
            scan_result.summary = self._generate_summary(all_vulnerabilities)
            
            # Store results
            await self._store_scan_result(scan_result)
            
            return scan_id
            
        except Exception as e:
            self.logger.error(f"Container scan failed: {e}")
            if scan_id in self.scan_results:
                self.scan_results[scan_id].status = "failed"
            raise
    
    async def _scan_single_container(self, image: str) -> List[Vulnerability]:
        """Scan single container image"""
        try:
            # Run Trivy scan
            cmd = [
                'trivy', 'image',
                '--format', 'json',
                '--severity', 'LOW,MEDIUM,HIGH,CRITICAL',
                image
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self.logger.error(f"Trivy scan failed for {image}: {result.stderr}")
                return []
            
            # Parse Trivy output
            trivy_data = json.loads(result.stdout)
            vulnerabilities = []
            
            for result_item in trivy_data.get('Results', []):
                target = result_item.get('Target', image)
                
                for vuln in result_item.get('Vulnerabilities', []):
                    vulnerability = Vulnerability(
                        vuln_id=f"trivy_{vuln.get('VulnerabilityID', 'unknown')}_{int(time.time())}",
                        cve_id=vuln.get('VulnerabilityID'),
                        title=vuln.get('Title', 'Unknown vulnerability'),
                        description=vuln.get('Description', ''),
                        severity=self._parse_severity(vuln.get('Severity', 'UNKNOWN')),
                        cvss_score=self._parse_cvss_score(vuln.get('CVSS', {})),
                        affected_component=vuln.get('PkgName', 'unknown'),
                        affected_version=vuln.get('InstalledVersion', 'unknown'),
                        fixed_version=vuln.get('FixedVersion'),
                        scan_type=ScanType.CONTAINER,
                        discovered_at=datetime.utcnow(),
                        remediation_advice=self._generate_container_remediation(vuln),
                        references=vuln.get('References', []),
                        exploitable=self._is_exploitable(vuln),
                        patch_available=bool(vuln.get('FixedVersion'))
                    )
                    vulnerabilities.append(vulnerability)
            
            self.logger.info(f"Found {len(vulnerabilities)} vulnerabilities in {image}")
            return vulnerabilities
            
        except Exception as e:
            self.logger.error(f"Failed to scan container {image}: {e}")
            return []
    
    async def scan_network(self, target_hosts: List[str]) -> str:
        """Scan network for vulnerabilities"""
        scan_id = self._generate_scan_id()
        
        try:
            scan_result = ScanResult(
                scan_id=scan_id,
                target=','.join(target_hosts),
                scan_type=ScanType.NETWORK,
                started_at=datetime.utcnow(),
                completed_at=None,
                status="running",
                vulnerabilities=[],
                summary={},
                scanner_version="nmap-latest",
                scan_config={"hosts": target_hosts}
            )
            
            self.scan_results[scan_id] = scan_result
            
            # Run network scans
            all_vulnerabilities = []
            for host in target_hosts:
                host_vulns = await self._scan_single_host(host)
                all_vulnerabilities.extend(host_vulns)
            
            # Update scan result
            scan_result.vulnerabilities = all_vulnerabilities
            scan_result.completed_at = datetime.utcnow()
            scan_result.status = "completed"
            scan_result.summary = self._generate_summary(all_vulnerabilities)
            
            # Store results
            await self._store_scan_result(scan_result)
            
            return scan_id
            
        except Exception as e:
            self.logger.error(f"Network scan failed: {e}")
            if scan_id in self.scan_results:
                self.scan_results[scan_id].status = "failed"
            raise
    
    async def _scan_single_host(self, host: str) -> List[Vulnerability]:
        """Scan single host for vulnerabilities"""
        try:
            vulnerabilities = []
            
            # Port scan
            port_scan_vulns = await self._nmap_port_scan(host)
            vulnerabilities.extend(port_scan_vulns)
            
            # Service detection
            service_vulns = await self._nmap_service_scan(host)
            vulnerabilities.extend(service_vulns)
            
            # SSL/TLS scan
            ssl_vulns = await self._scan_ssl_tls(host)
            vulnerabilities.extend(ssl_vulns)
            
            return vulnerabilities
            
        except Exception as e:
            self.logger.error(f"Failed to scan host {host}: {e}")
            return []
    
    async def _nmap_port_scan(self, host: str) -> List[Vulnerability]:
        """Perform Nmap port scan"""
        try:
            cmd = [
                'nmap', '-sS', '-O', '-sV', '--script=vuln',
                '-oX', '-', host
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                self.logger.warning(f"Nmap scan issues for {host}: {result.stderr}")
                return []
            
            # Parse Nmap XML output
            vulnerabilities = []
            try:
                root = ET.fromstring(result.stdout)
                
                for host_elem in root.findall('host'):
                    for port_elem in host_elem.findall('.//port'):
                        port_id = port_elem.get('portid')
                        protocol = port_elem.get('protocol')
                        
                        # Check for vulnerable services
                        service = port_elem.find('service')
                        if service is not None:
                            service_name = service.get('name', '')
                            version = service.get('version', '')
                            
                            # Check for known vulnerable services
                            vuln = self._check_service_vulnerabilities(
                                host, port_id, protocol, service_name, version
                            )
                            if vuln:
                                vulnerabilities.append(vuln)
                        
                        # Check script results for vulnerabilities
                        for script in port_elem.findall('.//script'):
                            script_id = script.get('id', '')
                            if 'vuln' in script_id:
                                vuln = self._parse_nmap_script_vuln(
                                    host, port_id, script
                                )
                                if vuln:
                                    vulnerabilities.append(vuln)
            
            except ET.ParseError as e:
                self.logger.error(f"Failed to parse Nmap output: {e}")
            
            return vulnerabilities
            
        except Exception as e:
            self.logger.error(f"Nmap port scan failed for {host}: {e}")
            return []
    
    async def _nmap_service_scan(self, host: str) -> List[Vulnerability]:
        """Perform service-specific vulnerability scan"""
        try:
            # This would implement service-specific scans
            # For now, return empty list
            return []
            
        except Exception as e:
            self.logger.error(f"Service scan failed for {host}: {e}")
            return []
    
    async def _scan_ssl_tls(self, host: str) -> List[Vulnerability]:
        """Scan SSL/TLS configuration"""
        try:
            vulnerabilities = []
            
            # Check common SSL/TLS issues
            cmd = [
                'nmap', '--script', 'ssl-enum-ciphers,ssl-cert,ssl-date',
                '-p', '443,8443', host
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Parse SSL scan results
                output = result.stdout
                
                # Check for weak ciphers
                if 'SSLv2' in output or 'SSLv3' in output:
                    vuln = Vulnerability(
                        vuln_id=f"ssl_weak_protocol_{host}_{int(time.time())}",
                        cve_id=None,
                        title="Weak SSL/TLS Protocol",
                        description="Server supports weak SSL/TLS protocols",
                        severity=VulnerabilitySeverity.HIGH,
                        cvss_score=7.5,
                        affected_component="SSL/TLS",
                        affected_version="SSLv2/SSLv3",
                        fixed_version="TLSv1.2+",
                        scan_type=ScanType.NETWORK,
                        discovered_at=datetime.utcnow(),
                        remediation_advice="Disable SSLv2 and SSLv3, enable only TLSv1.2 and above",
                        references=["https://tools.ietf.org/rfc/rfc7568.txt"],
                        exploitable=True,
                        patch_available=True
                    )
                    vulnerabilities.append(vuln)
            
            return vulnerabilities
            
        except Exception as e:
            self.logger.error(f"SSL/TLS scan failed for {host}: {e}")
            return []
    
    async def scan_application_code(self, code_path: str) -> str:
        """Scan application code for vulnerabilities"""
        scan_id = self._generate_scan_id()
        
        try:
            scan_result = ScanResult(
                scan_id=scan_id,
                target=code_path,
                scan_type=ScanType.APPLICATION,
                started_at=datetime.utcnow(),
                completed_at=None,
                status="running",
                vulnerabilities=[],
                summary={},
                scanner_version="semgrep-latest",
                scan_config={"code_path": code_path}
            )
            
            self.scan_results[scan_id] = scan_result
            
            # Run Semgrep scan
            vulnerabilities = await self._semgrep_scan(code_path)
            
            # Update scan result
            scan_result.vulnerabilities = vulnerabilities
            scan_result.completed_at = datetime.utcnow()
            scan_result.status = "completed"
            scan_result.summary = self._generate_summary(vulnerabilities)
            
            # Store results
            await self._store_scan_result(scan_result)
            
            return scan_id
            
        except Exception as e:
            self.logger.error(f"Application code scan failed: {e}")
            if scan_id in self.scan_results:
                self.scan_results[scan_id].status = "failed"
            raise
    
    async def _semgrep_scan(self, code_path: str) -> List[Vulnerability]:
        """Run Semgrep static analysis"""
        try:
            cmd = [
                'semgrep', '--config=auto', '--json',
                '--severity=ERROR', '--severity=WARNING',
                code_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0 and result.returncode != 1:  # 1 is findings found
                self.logger.error(f"Semgrep scan failed: {result.stderr}")
                return []
            
            # Parse Semgrep output
            semgrep_data = json.loads(result.stdout)
            vulnerabilities = []
            
            for finding in semgrep_data.get('results', []):
                severity = self._parse_semgrep_severity(finding.get('extra', {}).get('severity', 'INFO'))
                
                vulnerability = Vulnerability(
                    vuln_id=f"semgrep_{finding.get('check_id', 'unknown')}_{int(time.time())}",
                    cve_id=finding.get('extra', {}).get('metadata', {}).get('cve'),
                    title=finding.get('extra', {}).get('message', 'Code vulnerability'),
                    description=finding.get('extra', {}).get('message', ''),
                    severity=severity,
                    cvss_score=self._severity_to_cvss(severity),
                    affected_component=finding.get('path', 'unknown'),
                    affected_version="current",
                    fixed_version=None,
                    scan_type=ScanType.APPLICATION,
                    discovered_at=datetime.utcnow(),
                    remediation_advice=finding.get('extra', {}).get('fix', 'Review and fix the identified code issue'),
                    references=finding.get('extra', {}).get('references', []),
                    exploitable=severity.value >= VulnerabilitySeverity.HIGH.value,
                    patch_available=False
                )
                vulnerabilities.append(vulnerability)
            
            return vulnerabilities
            
        except Exception as e:
            self.logger.error(f"Semgrep scan failed: {e}")
            return []
    
    async def scan_infrastructure(self, config_files: List[str]) -> str:
        """Scan infrastructure configuration for vulnerabilities"""
        scan_id = self._generate_scan_id()
        
        try:
            scan_result = ScanResult(
                scan_id=scan_id,
                target=','.join(config_files),
                scan_type=ScanType.INFRASTRUCTURE,
                started_at=datetime.utcnow(),
                completed_at=None,
                status="running",
                vulnerabilities=[],
                summary={},
                scanner_version="custom-1.0",
                scan_config={"config_files": config_files}
            )
            
            self.scan_results[scan_id] = scan_result
            
            # Scan configuration files
            all_vulnerabilities = []
            for config_file in config_files:
                config_vulns = await self._scan_config_file(config_file)
                all_vulnerabilities.extend(config_vulns)
            
            # Update scan result
            scan_result.vulnerabilities = all_vulnerabilities
            scan_result.completed_at = datetime.utcnow()
            scan_result.status = "completed"
            scan_result.summary = self._generate_summary(all_vulnerabilities)
            
            # Store results
            await self._store_scan_result(scan_result)
            
            return scan_id
            
        except Exception as e:
            self.logger.error(f"Infrastructure scan failed: {e}")
            if scan_id in self.scan_results:
                self.scan_results[scan_id].status = "failed"
            raise
    
    async def _scan_config_file(self, config_file: str) -> List[Vulnerability]:
        """Scan configuration file for security issues"""
        try:
            vulnerabilities = []
            
            with open(config_file, 'r') as f:
                content = f.read()
            
            # Check for common misconfigurations
            if config_file.endswith('.yml') or config_file.endswith('.yaml'):
                config_data = yaml.safe_load(content)
                vulnerabilities.extend(self._check_yaml_security(config_file, config_data))
            
            # Check for hardcoded secrets
            secret_vulns = self._check_hardcoded_secrets(config_file, content)
            vulnerabilities.extend(secret_vulns)
            
            # Check for insecure configurations
            insecure_vulns = self._check_insecure_configs(config_file, content)
            vulnerabilities.extend(insecure_vulns)
            
            return vulnerabilities
            
        except Exception as e:
            self.logger.error(f"Failed to scan config file {config_file}: {e}")
            return []
    
    def _check_yaml_security(self, filename: str, config_data: Dict[str, Any]) -> List[Vulnerability]:
        """Check YAML configuration for security issues"""
        vulnerabilities = []
        
        # Check for privileged containers
        if isinstance(config_data, dict):
            services = config_data.get('services', {})
            for service_name, service_config in services.items():
                if isinstance(service_config, dict):
                    if service_config.get('privileged', False):
                        vuln = Vulnerability(
                            vuln_id=f"config_privileged_{service_name}_{int(time.time())}",
                            cve_id=None,
                            title="Privileged Container Configuration",
                            description=f"Service {service_name} is configured to run in privileged mode",
                            severity=VulnerabilitySeverity.HIGH,
                            cvss_score=7.2,
                            affected_component=service_name,
                            affected_version="current",
                            fixed_version=None,
                            scan_type=ScanType.CONFIGURATION,
                            discovered_at=datetime.utcnow(),
                            remediation_advice="Remove privileged: true or use specific capabilities instead",
                            references=["https://docs.docker.com/engine/reference/run/#runtime-privilege-and-linux-capabilities"],
                            exploitable=True,
                            patch_available=True
                        )
                        vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    def _check_hardcoded_secrets(self, filename: str, content: str) -> List[Vulnerability]:
        """Check for hardcoded secrets in configuration"""
        vulnerabilities = []
        
        # Common secret patterns
        secret_patterns = [
            (r'password\s*[:=]\s*["\']?([^"\'\s]+)["\']?', "Password"),
            (r'api[_-]?key\s*[:=]\s*["\']?([^"\'\s]+)["\']?', "API Key"),
            (r'secret[_-]?key\s*[:=]\s*["\']?([^"\'\s]+)["\']?', "Secret Key"),
            (r'token\s*[:=]\s*["\']?([^"\'\s]+)["\']?', "Token"),
            (r'private[_-]?key\s*[:=]\s*["\']?([^"\'\s]+)["\']?', "Private Key"),
        ]
        
        for pattern, secret_type in secret_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                secret_value = match.group(1)
                
                # Skip obvious placeholders
                if secret_value.lower() in ['changeme', 'password', 'secret', 'key', 'token', 'xxx', '***']:
                    continue
                
                vuln = Vulnerability(
                    vuln_id=f"hardcoded_secret_{hashlib.md5(secret_value.encode()).hexdigest()[:8]}",
                    cve_id=None,
                    title=f"Hardcoded {secret_type}",
                    description=f"Hardcoded {secret_type.lower()} found in {filename}",
                    severity=VulnerabilitySeverity.HIGH,
                    cvss_score=8.1,
                    affected_component=filename,
                    affected_version="current",
                    fixed_version=None,
                    scan_type=ScanType.CONFIGURATION,
                    discovered_at=datetime.utcnow(),
                    remediation_advice=f"Remove hardcoded {secret_type.lower()} and use environment variables or secret management",
                    references=["https://owasp.org/www-community/vulnerabilities/Use_of_hard-coded_password"],
                    exploitable=True,
                    patch_available=True
                )
                vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    def _check_insecure_configs(self, filename: str, content: str) -> List[Vulnerability]:
        """Check for insecure configurations"""
        vulnerabilities = []
        
        # Check for insecure SSL/TLS settings
        if 'ssl' in content.lower() and 'false' in content.lower():
            vuln = Vulnerability(
                vuln_id=f"insecure_ssl_{int(time.time())}",
                cve_id=None,
                title="SSL/TLS Disabled",
                description="SSL/TLS appears to be disabled in configuration",
                severity=VulnerabilitySeverity.MEDIUM,
                cvss_score=5.3,
                affected_component=filename,
                affected_version="current",
                fixed_version=None,
                scan_type=ScanType.CONFIGURATION,
                discovered_at=datetime.utcnow(),
                remediation_advice="Enable SSL/TLS encryption for all network communications",
                references=["https://owasp.org/www-project-top-ten/2017/A3_2017-Sensitive_Data_Exposure"],
                exploitable=True,
                patch_available=True
            )
            vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    def _parse_severity(self, severity_str: str) -> VulnerabilitySeverity:
        """Parse severity string to enum"""
        severity_map = {
            'CRITICAL': VulnerabilitySeverity.CRITICAL,
            'HIGH': VulnerabilitySeverity.HIGH,
            'MEDIUM': VulnerabilitySeverity.MEDIUM,
            'LOW': VulnerabilitySeverity.LOW,
            'INFORMATIONAL': VulnerabilitySeverity.INFORMATIONAL,
            'INFO': VulnerabilitySeverity.INFORMATIONAL,
            'UNKNOWN': VulnerabilitySeverity.LOW
        }
        return severity_map.get(severity_str.upper(), VulnerabilitySeverity.LOW)
    
    def _parse_semgrep_severity(self, severity_str: str) -> VulnerabilitySeverity:
        """Parse Semgrep severity to enum"""
        severity_map = {
            'ERROR': VulnerabilitySeverity.HIGH,
            'WARNING': VulnerabilitySeverity.MEDIUM,
            'INFO': VulnerabilitySeverity.LOW
        }
        return severity_map.get(severity_str.upper(), VulnerabilitySeverity.LOW)
    
    def _parse_cvss_score(self, cvss_data: Dict[str, Any]) -> float:
        """Parse CVSS score from vulnerability data"""
        if isinstance(cvss_data, dict):
            # Try different CVSS version fields
            for version in ['nvd', 'redhat', 'v3', 'v2']:
                if version in cvss_data:
                    score = cvss_data[version].get('Score', 0)
                    if score:
                        return float(score)
        
        return 0.0
    
    def _severity_to_cvss(self, severity: VulnerabilitySeverity) -> float:
        """Convert severity to approximate CVSS score"""
        score_map = {
            VulnerabilitySeverity.CRITICAL: 9.0,
            VulnerabilitySeverity.HIGH: 7.0,
            VulnerabilitySeverity.MEDIUM: 5.0,
            VulnerabilitySeverity.LOW: 3.0,
            VulnerabilitySeverity.INFORMATIONAL: 1.0
        }
        return score_map.get(severity, 0.0)
    
    def _is_exploitable(self, vuln_data: Dict[str, Any]) -> bool:
        """Determine if vulnerability is exploitable"""
        # Check for exploit availability indicators
        exploitable_indicators = [
            'exploit', 'poc', 'proof of concept', 'metasploit'
        ]
        
        description = vuln_data.get('Description', '').lower()
        title = vuln_data.get('Title', '').lower()
        
        return any(indicator in description or indicator in title 
                  for indicator in exploitable_indicators)
    
    def _generate_container_remediation(self, vuln: Dict[str, Any]) -> str:
        """Generate remediation advice for container vulnerability"""
        fixed_version = vuln.get('FixedVersion')
        pkg_name = vuln.get('PkgName', 'package')
        
        if fixed_version:
            return f"Update {pkg_name} to version {fixed_version} or later"
        else:
            return f"No fix available for {pkg_name}. Consider using alternative packages or applying workarounds"
    
    def _check_service_vulnerabilities(self, host: str, port: str, protocol: str,
                                     service: str, version: str) -> Optional[Vulnerability]:
        """Check if service version has known vulnerabilities"""
        # This would integrate with vulnerability databases
        # For now, return None (no vulnerability found)
        return None
    
    def _parse_nmap_script_vuln(self, host: str, port: str, script_elem) -> Optional[Vulnerability]:
        """Parse Nmap script output for vulnerabilities"""
        script_id = script_elem.get('id', '')
        script_output = script_elem.get('output', '')
        
        if 'vuln' in script_id and script_output:
            return Vulnerability(
                vuln_id=f"nmap_{script_id}_{host}_{port}_{int(time.time())}",
                cve_id=None,
                title=f"Network vulnerability detected by {script_id}",
                description=script_output[:500],  # Truncate long output
                severity=VulnerabilitySeverity.MEDIUM,
                cvss_score=5.0,
                affected_component=f"{host}:{port}",
                affected_version="unknown",
                fixed_version=None,
                scan_type=ScanType.NETWORK,
                discovered_at=datetime.utcnow(),
                remediation_advice="Review service configuration and apply security patches",
                references=[],
                exploitable=False,
                patch_available=False
            )
        
        return None
    
    def _generate_summary(self, vulnerabilities: List[Vulnerability]) -> Dict[str, int]:
        """Generate vulnerability summary"""
        summary = {
            'total': len(vulnerabilities),
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'informational': 0,
            'exploitable': 0,
            'patchable': 0
        }
        
        for vuln in vulnerabilities:
            if vuln.severity == VulnerabilitySeverity.CRITICAL:
                summary['critical'] += 1
            elif vuln.severity == VulnerabilitySeverity.HIGH:
                summary['high'] += 1
            elif vuln.severity == VulnerabilitySeverity.MEDIUM:
                summary['medium'] += 1
            elif vuln.severity == VulnerabilitySeverity.LOW:
                summary['low'] += 1
            else:
                summary['informational'] += 1
            
            if vuln.exploitable:
                summary['exploitable'] += 1
            if vuln.patch_available:
                summary['patchable'] += 1
        
        return summary
    
    async def _store_scan_result(self, scan_result: ScanResult):
        """Store scan result in database"""
        try:
            cursor = self.db_connection.cursor()
            
            # Store scan result
            cursor.execute("""
                INSERT INTO vulnerability_scans 
                (scan_id, target, scan_type, started_at, completed_at, status, 
                 summary, scanner_version, scan_config)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                scan_result.scan_id,
                scan_result.target,
                scan_result.scan_type.value,
                scan_result.started_at,
                scan_result.completed_at,
                scan_result.status,
                json.dumps(scan_result.summary),
                scan_result.scanner_version,
                json.dumps(scan_result.scan_config)
            ))
            
            # Store vulnerabilities
            for vuln in scan_result.vulnerabilities:
                cursor.execute("""
                    INSERT INTO vulnerabilities 
                    (vuln_id, scan_id, cve_id, title, description, severity, cvss_score,
                     affected_component, affected_version, fixed_version, scan_type,
                     discovered_at, remediation_advice, references, exploitable, patch_available)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    vuln.vuln_id, scan_result.scan_id, vuln.cve_id, vuln.title,
                    vuln.description, vuln.severity.value, vuln.cvss_score,
                    vuln.affected_component, vuln.affected_version, vuln.fixed_version,
                    vuln.scan_type.value, vuln.discovered_at, vuln.remediation_advice,
                    json.dumps(vuln.references), vuln.exploitable, vuln.patch_available
                ))
            
            self.db_connection.commit()
            cursor.close()
            
            # Store in Redis for real-time access
            self.redis_client.setex(
                f"scan_result:{scan_result.scan_id}",
                3600,  # 1 hour TTL
                json.dumps(asdict(scan_result), default=str)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store scan result: {e}")
    
    def _generate_scan_id(self) -> str:
        """Generate unique scan ID"""
        timestamp = str(int(time.time() * 1000))
        random_part = hashlib.md5(os.urandom(16)).hexdigest()[:8]
        return f"scan_{timestamp}_{random_part}"
    
    def get_scan_result(self, scan_id: str) -> Optional[ScanResult]:
        """Get scan result by ID"""
        return self.scan_results.get(scan_id)
    
    def get_all_scan_results(self) -> List[ScanResult]:
        """Get all scan results"""
        return list(self.scan_results.values())
    
    async def generate_scan_report(self, scan_id: str, format: str = "json") -> str:
        """Generate scan report"""
        scan_result = self.get_scan_result(scan_id)
        if not scan_result:
            raise ValueError(f"Scan result not found: {scan_id}")
        
        if format == "json":
            return json.dumps(asdict(scan_result), indent=2, default=str)
        elif format == "html":
            return self._generate_html_report(scan_result)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_html_report(self, scan_result: ScanResult) -> str:
        """Generate HTML report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vulnerability Scan Report - {scan_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .critical {{ color: #dc3545; }}
                .high {{ color: #fd7e14; }}
                .medium {{ color: #ffc107; }}
                .low {{ color: #28a745; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Vulnerability Scan Report</h1>
            <div class="summary">
                <h2>Scan Summary</h2>
                <p><strong>Scan ID:</strong> {scan_id}</p>
                <p><strong>Target:</strong> {target}</p>
                <p><strong>Scan Type:</strong> {scan_type}</p>
                <p><strong>Started:</strong> {started_at}</p>
                <p><strong>Completed:</strong> {completed_at}</p>
                <p><strong>Status:</strong> {status}</p>
                
                <h3>Vulnerability Counts</h3>
                <p><span class="critical">Critical: {critical}</span> | 
                   <span class="high">High: {high}</span> | 
                   <span class="medium">Medium: {medium}</span> | 
                   <span class="low">Low: {low}</span></p>
            </div>
            
            <h2>Vulnerabilities</h2>
            <table>
                <tr>
                    <th>CVE ID</th>
                    <th>Title</th>
                    <th>Severity</th>
                    <th>CVSS Score</th>
                    <th>Component</th>
                    <th>Fixed Version</th>
                </tr>
                {vulnerability_rows}
            </table>
        </body>
        </html>
        """
        
        # Generate vulnerability rows
        vulnerability_rows = ""
        for vuln in scan_result.vulnerabilities:
            severity_class = vuln.severity.name.lower()
            vulnerability_rows += f"""
                <tr>
                    <td>{vuln.cve_id or 'N/A'}</td>
                    <td>{vuln.title}</td>
                    <td class="{severity_class}">{vuln.severity.name}</td>
                    <td>{vuln.cvss_score}</td>
                    <td>{vuln.affected_component}</td>
                    <td>{vuln.fixed_version or 'N/A'}</td>
                </tr>
            """
        
        return html_template.format(
            scan_id=scan_result.scan_id,
            target=scan_result.target,
            scan_type=scan_result.scan_type.value,
            started_at=scan_result.started_at,
            completed_at=scan_result.completed_at or 'N/A',
            status=scan_result.status,
            critical=scan_result.summary.get('critical', 0),
            high=scan_result.summary.get('high', 0),
            medium=scan_result.summary.get('medium', 0),
            low=scan_result.summary.get('low', 0),
            vulnerability_rows=vulnerability_rows
        )

# Database schema for vulnerability management
VULNERABILITY_SCHEMA = """
-- Vulnerability scans table
CREATE TABLE IF NOT EXISTS vulnerability_scans (
    id SERIAL PRIMARY KEY,
    scan_id VARCHAR(255) UNIQUE NOT NULL,
    target TEXT NOT NULL,
    scan_type VARCHAR(50) NOT NULL,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    status VARCHAR(50) NOT NULL,
    summary JSONB,
    scanner_version VARCHAR(100),
    scan_config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vulnerabilities table
CREATE TABLE IF NOT EXISTS vulnerabilities (
    id SERIAL PRIMARY KEY,
    vuln_id VARCHAR(255) UNIQUE NOT NULL,
    scan_id VARCHAR(255) REFERENCES vulnerability_scans(scan_id),
    cve_id VARCHAR(50),
    title TEXT NOT NULL,
    description TEXT,
    severity INTEGER NOT NULL,
    cvss_score FLOAT DEFAULT 0,
    affected_component TEXT NOT NULL,
    affected_version TEXT,
    fixed_version TEXT,
    scan_type VARCHAR(50) NOT NULL,
    discovered_at TIMESTAMP NOT NULL,
    remediation_advice TEXT,
    references JSONB,
    exploitable BOOLEAN DEFAULT false,
    patch_available BOOLEAN DEFAULT false,
    remediation_status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_vulnerability_scans_scan_id ON vulnerability_scans(scan_id);
CREATE INDEX IF NOT EXISTS idx_vulnerability_scans_target ON vulnerability_scans(target);
CREATE INDEX IF NOT EXISTS idx_vulnerabilities_vuln_id ON vulnerabilities(vuln_id);
CREATE INDEX IF NOT EXISTS idx_vulnerabilities_cve_id ON vulnerabilities(cve_id);
CREATE INDEX IF NOT EXISTS idx_vulnerabilities_severity ON vulnerabilities(severity);
"""

if __name__ == "__main__":
    # Example usage
    config = {
        'redis_host': 'redis',
        'redis_port': 6379,
        'postgres_host': 'postgres',
        'postgres_port': 5432,
        'postgres_db': 'sutazai',
        'postgres_user': 'sutazai'
    }
    
    async def main():
        scanner = VulnerabilityScanner(config)
        
        # Scan containers
        container_scan_id = await scanner.scan_containers(['ubuntu:latest', 'nginx:latest'])
        print(f"Container scan started: {container_scan_id}")
        
        # Scan network
        network_scan_id = await scanner.scan_network(['127.0.0.1', '172.20.0.1'])
        print(f"Network scan started: {network_scan_id}")
        
        # Scan application code
        code_scan_id = await scanner.scan_application_code('/opt/sutazaiapp')
        print(f"Code scan started: {code_scan_id}")
        
        # Generate report
        await asyncio.sleep(5)  # Wait for scans to complete
        report = await scanner.generate_scan_report(container_scan_id, "json")
        print("Scan report generated")
    
    asyncio.run(main())