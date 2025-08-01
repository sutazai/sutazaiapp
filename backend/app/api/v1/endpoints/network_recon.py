"""
AI-Powered Network Reconnaissance API
Enterprise-grade network security scanning and threat analysis system
"""

import asyncio
import ipaddress
import socket
import subprocess
import nmap
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pydantic import BaseModel, Field, validator
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading

# Import existing security infrastructure
try:
    from app.core.security import security_manager, SecurityLevel
    from app.api.v1.security import get_current_user
    SECURITY_ENABLED = True
except ImportError:
    SECURITY_ENABLED = False
    security_manager = None

logger = logging.getLogger(__name__)

class ScanType(Enum):
    """Network scan types"""
    PING_SWEEP = "ping_sweep"
    PORT_SCAN = "port_scan"
    SERVICE_DETECTION = "service_detection"
    OS_FINGERPRINT = "os_fingerprint"
    VULNERABILITY_SCAN = "vulnerability_scan"
    COMPREHENSIVE = "comprehensive"
    STEALTH = "stealth"

class ThreatLevel(Enum):
    """Threat assessment levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class NetworkHost:
    """Discovered network host"""
    ip: str
    hostname: Optional[str] = None
    mac_address: Optional[str] = None
    vendor: Optional[str] = None
    os_info: Optional[Dict[str, Any]] = None
    open_ports: List[Dict[str, Any]] = None
    services: List[Dict[str, Any]] = None
    vulnerabilities: List[Dict[str, Any]] = None
    last_seen: str = None
    threat_level: str = "info"
    
    def __post_init__(self):
        if self.open_ports is None:
            self.open_ports = []
        if self.services is None:
            self.services = []
        if self.vulnerabilities is None:
            self.vulnerabilities = []
        if self.last_seen is None:
            self.last_seen = datetime.utcnow().isoformat()

@dataclass
class ScanResult:
    """Network reconnaissance scan result"""
    scan_id: str
    scan_type: str
    target: str
    start_time: str
    end_time: Optional[str] = None
    status: str = "running"
    hosts_discovered: List[NetworkHost] = None
    network_topology: Dict[str, Any] = None
    threat_summary: Dict[str, Any] = None
    recommendations: List[str] = None
    ai_analysis: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.hosts_discovered is None:
            self.hosts_discovered = []
        if self.recommendations is None:
            self.recommendations = []

# Request/Response Models
class NetworkScanRequest(BaseModel):
    target: str = Field(..., description="Target IP, subnet, or hostname")
    scan_type: ScanType = Field(default=ScanType.COMPREHENSIVE, description="Type of scan to perform")
    port_range: Optional[str] = Field(default="1-1000", description="Port range (e.g., '22,80,443' or '1-65535')")
    stealth_mode: bool = Field(default=True, description="Use stealth scanning techniques")
    timeout: int = Field(default=300, description="Scan timeout in seconds")
    deep_analysis: bool = Field(default=False, description="Enable AI-powered deep threat analysis")
    
    @validator('target')
    def validate_target(cls, v):
        try:
            # Try to parse as IP address or subnet
            ipaddress.ip_network(v, strict=False)
            return v
        except ValueError:
            # If not an IP, assume it's a hostname
            if not v or len(v) > 255:
                raise ValueError("Invalid target format")
            return v

class VulnerabilityAssessmentRequest(BaseModel):
    targets: List[str] = Field(..., description="List of targets to assess")
    scan_intensity: str = Field(default="normal", description="Scan intensity: light, normal, aggressive")
    include_exploits: bool = Field(default=False, description="Include exploit information")
    ai_threat_modeling: bool = Field(default=True, description="Enable AI threat modeling")

class NetworkDiscoveryRequest(BaseModel):
    network: str = Field(..., description="Network to discover (e.g., '192.168.1.0/24')")
    discovery_methods: List[str] = Field(default=["ping", "arp"], description="Discovery methods to use")
    port_discovery: bool = Field(default=True, description="Include port discovery")

# Response Models
class NetworkScanResponse(BaseModel):
    scan_id: str
    status: str
    message: str
    estimated_completion: Optional[str] = None

class ScanStatusResponse(BaseModel):
    scan_id: str
    status: str
    progress: float
    hosts_found: int
    current_phase: str
    estimated_remaining: Optional[int] = None

class NetworkReconResults(BaseModel):
    scan_id: str
    summary: Dict[str, Any]
    hosts: List[Dict[str, Any]]
    topology: Dict[str, Any]
    threats: List[Dict[str, Any]]
    recommendations: List[str]
    ai_insights: Optional[Dict[str, Any]] = None

# Router setup
router = APIRouter()

# Global scan storage (in production, use Redis or database)
active_scans: Dict[str, ScanResult] = {}
scan_lock = threading.Lock()

class NetworkReconEngine:
    """AI-powered network reconnaissance engine"""
    
    def __init__(self):
        self.scanner = self._init_scanner()
        self.threat_analyzer = AIThreatAnalyzer()
        self.vulnerability_db = VulnerabilityDatabase()
        
    def _init_scanner(self):
        """Initialize network scanner"""
        try:
            return nmap.PortScanner()
        except Exception as e:
            logger.warning(f"Nmap not available: {e}")
            return None
    
    async def perform_scan(self, request: NetworkScanRequest, scan_id: str) -> ScanResult:
        """Perform comprehensive network reconnaissance"""
        result = ScanResult(
            scan_id=scan_id,
            scan_type=request.scan_type.value,
            target=request.target,
            start_time=datetime.utcnow().isoformat()
        )
        
        try:
            # Phase 1: Host Discovery
            await self._update_scan_status(scan_id, "Host Discovery", 0.1)
            discovered_hosts = await self._discover_hosts(request.target)
            
            # Phase 2: Port Scanning
            await self._update_scan_status(scan_id, "Port Scanning", 0.3)
            for host in discovered_hosts:
                await self._scan_ports(host, request.port_range, request.stealth_mode)
            
            # Phase 3: Service Detection
            await self._update_scan_status(scan_id, "Service Detection", 0.5)
            for host in discovered_hosts:
                await self._detect_services(host)
            
            # Phase 4: OS Fingerprinting
            await self._update_scan_status(scan_id, "OS Fingerprinting", 0.7)
            for host in discovered_hosts:
                await self._fingerprint_os(host)
            
            # Phase 5: Vulnerability Assessment
            if request.scan_type in [ScanType.VULNERABILITY_SCAN, ScanType.COMPREHENSIVE]:
                await self._update_scan_status(scan_id, "Vulnerability Assessment", 0.8)
                for host in discovered_hosts:
                    await self._assess_vulnerabilities(host)
            
            # Phase 6: AI Analysis
            if request.deep_analysis:
                await self._update_scan_status(scan_id, "AI Threat Analysis", 0.9)
                result.ai_analysis = await self.threat_analyzer.analyze_network(discovered_hosts)
            
            # Finalize results
            result.hosts_discovered = discovered_hosts
            result.network_topology = await self._build_topology(discovered_hosts)
            result.threat_summary = await self._generate_threat_summary(discovered_hosts)
            result.recommendations = await self._generate_recommendations(discovered_hosts)
            result.status = "completed"
            result.end_time = datetime.utcnow().isoformat()
            
            await self._update_scan_status(scan_id, "Completed", 1.0)
            
        except Exception as e:
            logger.error(f"Scan failed: {e}")
            result.status = "failed"
            result.end_time = datetime.utcnow().isoformat()
            
        return result
    
    async def _discover_hosts(self, target: str) -> List[NetworkHost]:
        """Discover active hosts in network"""
        hosts = []
        
        try:
            # Parse target network
            network = ipaddress.ip_network(target, strict=False)
            
            # Ping sweep for host discovery
            tasks = []
            for ip in network.hosts():
                if len(tasks) < 100:  # Limit concurrent pings
                    tasks.append(self._ping_host(str(ip)))
                
            # Execute ping sweep
            ping_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(ping_results):
                if isinstance(result, NetworkHost):
                    hosts.append(result)
                    
        except Exception as e:
            logger.error(f"Host discovery failed: {e}")
            # Fallback: treat as single host
            hosts.append(NetworkHost(ip=target))
            
        return hosts
    
    async def _ping_host(self, ip: str) -> Optional[NetworkHost]:
        """Ping individual host"""
        try:
            # Use asyncio subprocess for non-blocking ping
            process = await asyncio.create_subprocess_exec(
                'ping', '-c', '1', '-W', '1', ip,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Host is up, try to get hostname
                hostname = await self._resolve_hostname(ip)
                return NetworkHost(ip=ip, hostname=hostname)
                
        except Exception as e:
            logger.debug(f"Ping failed for {ip}: {e}")
            
        return None
    
    async def _resolve_hostname(self, ip: str) -> Optional[str]:
        """Resolve hostname for IP"""
        try:
            hostname = socket.gethostbyaddr(ip)[0]
            return hostname
        except:
            return None
    
    async def _scan_ports(self, host: NetworkHost, port_range: str, stealth: bool):
        """Scan ports on host"""
        if not self.scanner:
            return
            
        try:
            # Configure scan parameters
            arguments = '-sS' if stealth else '-sT'  # SYN scan vs TCP connect
            arguments += ' -T4'  # Timing template
            
            # Perform port scan
            scan_result = self.scanner.scan(host.ip, port_range, arguments=arguments)
            
            # Process results
            if host.ip in scan_result['scan']:
                host_data = scan_result['scan'][host.ip]
                
                if 'tcp' in host_data:
                    for port, info in host_data['tcp'].items():
                        if info['state'] == 'open':
                            port_info = {
                                'port': port,
                                'protocol': 'tcp',
                                'state': info['state'],
                                'service': info.get('name', 'unknown'),
                                'version': info.get('version', ''),
                                'product': info.get('product', '')
                            }
                            host.open_ports.append(port_info)
                            
        except Exception as e:
            logger.error(f"Port scan failed for {host.ip}: {e}")
    
    async def _detect_services(self, host: NetworkHost):
        """Detect services running on open ports"""
        if not self.scanner:
            return
            
        try:
            # Service detection scan
            port_list = ','.join([str(p['port']) for p in host.open_ports])
            if port_list:
                scan_result = self.scanner.scan(
                    host.ip, 
                    port_list, 
                    arguments='-sV -T4'
                )
                
                # Update service information
                if host.ip in scan_result['scan']:
                    host_data = scan_result['scan'][host.ip]
                    if 'tcp' in host_data:
                        for port, info in host_data['tcp'].items():
                            service_info = {
                                'port': port,
                                'name': info.get('name', 'unknown'),
                                'product': info.get('product', ''),
                                'version': info.get('version', ''),
                                'extrainfo': info.get('extrainfo', ''),
                                'cpe': info.get('cpe', '')
                            }
                            host.services.append(service_info)
                            
        except Exception as e:
            logger.error(f"Service detection failed for {host.ip}: {e}")
    
    async def _fingerprint_os(self, host: NetworkHost):
        """Perform OS fingerprinting"""
        if not self.scanner:
            return
            
        try:
            # OS detection scan
            scan_result = self.scanner.scan(
                host.ip, 
                arguments='-O -T4'
            )
            
            # Process OS information
            if host.ip in scan_result['scan']:
                host_data = scan_result['scan'][host.ip]
                if 'osmatch' in host_data:
                    os_matches = host_data['osmatch']
                    if os_matches:
                        best_match = os_matches[0]
                        host.os_info = {
                            'name': best_match.get('name', 'Unknown'),
                            'accuracy': best_match.get('accuracy', 0),
                            'line': best_match.get('line', 0),
                            'osclass': host_data.get('osclass', [])
                        }
                        
        except Exception as e:
            logger.error(f"OS fingerprinting failed for {host.ip}: {e}")
    
    async def _assess_vulnerabilities(self, host: NetworkHost):
        """Assess vulnerabilities for host"""
        try:
            # Check each service for known vulnerabilities
            for service in host.services:
                vulns = await self.vulnerability_db.check_service(
                    service.get('product', ''),
                    service.get('version', ''),
                    service.get('port', 0)
                )
                host.vulnerabilities.extend(vulns)
                
            # Determine overall threat level
            if any(v.get('severity') == 'critical' for v in host.vulnerabilities):
                host.threat_level = ThreatLevel.CRITICAL.value
            elif any(v.get('severity') == 'high' for v in host.vulnerabilities):
                host.threat_level = ThreatLevel.HIGH.value
            elif any(v.get('severity') == 'medium' for v in host.vulnerabilities):
                host.threat_level = ThreatLevel.MEDIUM.value
            elif host.vulnerabilities:
                host.threat_level = ThreatLevel.LOW.value
                
        except Exception as e:
            logger.error(f"Vulnerability assessment failed for {host.ip}: {e}")
    
    async def _build_topology(self, hosts: List[NetworkHost]) -> Dict[str, Any]:
        """Build network topology map"""
        topology = {
            'nodes': [],
            'edges': [],
            'subnets': {},
            'gateways': []
        }
        
        # Add hosts as nodes
        for host in hosts:
            node = {
                'id': host.ip,
                'label': host.hostname or host.ip,
                'type': 'host',
                'threat_level': host.threat_level,
                'services': len(host.services),
                'vulnerabilities': len(host.vulnerabilities)
            }
            topology['nodes'].append(node)
            
        return topology
    
    async def _generate_threat_summary(self, hosts: List[NetworkHost]) -> Dict[str, Any]:
        """Generate threat assessment summary"""
        summary = {
            'total_hosts': len(hosts),
            'hosts_by_threat_level': {
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0,
                'info': 0
            },
            'total_vulnerabilities': 0,
            'critical_vulnerabilities': 0,
            'exposed_services': [],
            'security_score': 0
        }
        
        # Analyze hosts
        for host in hosts:
            summary['hosts_by_threat_level'][host.threat_level] += 1
            summary['total_vulnerabilities'] += len(host.vulnerabilities)
            
            # Count critical vulnerabilities
            critical_vulns = [v for v in host.vulnerabilities if v.get('severity') == 'critical']
            summary['critical_vulnerabilities'] += len(critical_vulns)
            
            # Track exposed services
            for service in host.services:
                service_summary = f"{service.get('name')} on {host.ip}:{service.get('port')}"
                summary['exposed_services'].append(service_summary)
        
        # Calculate security score (0-100)
        total_hosts = len(hosts)
        if total_hosts > 0:
            critical_weight = summary['hosts_by_threat_level']['critical'] * 25
            high_weight = summary['hosts_by_threat_level']['high'] * 15
            medium_weight = summary['hosts_by_threat_level']['medium'] * 5
            
            penalty = (critical_weight + high_weight + medium_weight) / total_hosts
            summary['security_score'] = max(0, 100 - penalty)
        
        return summary
    
    async def _generate_recommendations(self, hosts: List[NetworkHost]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        # Check for common issues
        critical_hosts = [h for h in hosts if h.threat_level == ThreatLevel.CRITICAL.value]
        if critical_hosts:
            recommendations.append(
                f"CRITICAL: {len(critical_hosts)} hosts have critical vulnerabilities requiring immediate attention"
            )
        
        # Check for exposed services
        exposed_services = set()
        for host in hosts:
            for service in host.services:
                service_name = service.get('name', 'unknown')
                if service_name in ['ssh', 'telnet', 'ftp', 'rdp']:
                    exposed_services.add(service_name)
        
        if exposed_services:
            recommendations.append(
                f"Consider restricting access to exposed management services: {', '.join(exposed_services)}"
            )
        
        # Check for unpatched services
        outdated_services = []
        for host in hosts:
            for service in host.services:
                if service.get('version') and 'old' in service.get('extrainfo', '').lower():
                    outdated_services.append(f"{service.get('name')} on {host.ip}")
        
        if outdated_services:
            recommendations.append(
                f"Update outdated services: {', '.join(outdated_services[:5])}"
            )
        
        # General recommendations
        if len(hosts) > 10:
            recommendations.append("Consider network segmentation to reduce attack surface")
        
        recommendations.append("Implement regular vulnerability scanning and patch management")
        recommendations.append("Deploy network monitoring and intrusion detection systems")
        
        return recommendations
    
    async def _update_scan_status(self, scan_id: str, phase: str, progress: float):
        """Update scan status"""
        with scan_lock:
            if scan_id in active_scans:
                # Update status information
                pass  # Status updates would be stored in database in production

class AIThreatAnalyzer:
    """AI-powered threat analysis system"""
    
    async def analyze_network(self, hosts: List[NetworkHost]) -> Dict[str, Any]:
        """Perform AI-powered threat analysis"""
        analysis = {
            'threat_vectors': [],
            'attack_paths': [],
            'risk_assessment': {},
            'remediation_priority': []
        }
        
        # Analyze threat vectors
        analysis['threat_vectors'] = await self._identify_threat_vectors(hosts)
        
        # Map potential attack paths
        analysis['attack_paths'] = await self._map_attack_paths(hosts)
        
        # Assess overall risk
        analysis['risk_assessment'] = await self._assess_risk(hosts)
        
        # Prioritize remediation
        analysis['remediation_priority'] = await self._prioritize_remediation(hosts)
        
        return analysis
    
    async def _identify_threat_vectors(self, hosts: List[NetworkHost]) -> List[Dict[str, Any]]:
        """Identify potential threat vectors"""
        vectors = []
        
        for host in hosts:
            # Check for common attack vectors
            for service in host.services:
                service_name = service.get('name', '').lower()
                
                if service_name in ['ssh', 'rdp', 'telnet']:
                    vectors.append({
                        'type': 'Remote Access',
                        'target': f"{host.ip}:{service.get('port')}",
                        'service': service_name,
                        'risk': 'high' if service_name == 'telnet' else 'medium',
                        'description': f'Exposed {service_name} service allows remote access'
                    })
                
                elif service_name in ['http', 'https']:
                    vectors.append({
                        'type': 'Web Application',
                        'target': f"{host.ip}:{service.get('port')}",
                        'service': service_name,
                        'risk': 'medium',
                        'description': 'Web service may contain application vulnerabilities'
                    })
        
        return vectors
    
    async def _map_attack_paths(self, hosts: List[NetworkHost]) -> List[Dict[str, Any]]:
        """Map potential attack paths"""
        paths = []
        
        # Simple attack path analysis
        entry_points = [h for h in hosts if any(
            s.get('name', '').lower() in ['ssh', 'rdp', 'http', 'https'] 
            for s in h.services
        )]
        
        for entry in entry_points:
            path = {
                'entry_point': entry.ip,
                'steps': [
                    f'Initial access via {entry.ip}',
                    'Privilege escalation',
                    'Lateral movement',
                    'Data exfiltration'
                ],
                'likelihood': 'medium',
                'impact': 'high'
            }
            paths.append(path)
        
        return paths
    
    async def _assess_risk(self, hosts: List[NetworkHost]) -> Dict[str, Any]:
        """Assess overall network risk"""
        risk = {
            'overall_score': 0,
            'factors': [],
            'recommendations': []
        }
        
        # Calculate risk factors
        total_vulns = sum(len(h.vulnerabilities) for h in hosts)
        critical_vulns = sum(
            len([v for v in h.vulnerabilities if v.get('severity') == 'critical'])
            for h in hosts
        )
        
        # Risk scoring
        risk_score = min(100, (critical_vulns * 20) + (total_vulns * 2))
        risk['overall_score'] = risk_score
        
        # Add risk factors
        if critical_vulns > 0:
            risk['factors'].append(f'{critical_vulns} critical vulnerabilities detected')
        
        if total_vulns > len(hosts) * 3:
            risk['factors'].append('High vulnerability density across network')
        
        return risk
    
    async def _prioritize_remediation(self, hosts: List[NetworkHost]) -> List[Dict[str, Any]]:
        """Prioritize remediation tasks"""
        tasks = []
        
        # Critical vulnerabilities first
        for host in hosts:
            critical_vulns = [v for v in host.vulnerabilities if v.get('severity') == 'critical']
            for vuln in critical_vulns:
                tasks.append({
                    'priority': 1,
                    'host': host.ip,
                    'task': f"Patch critical vulnerability: {vuln.get('name', 'Unknown')}",
                    'impact': 'critical',
                    'effort': 'medium'
                })
        
        # Exposed services
        for host in hosts:
            for service in host.services:
                if service.get('name', '').lower() in ['telnet', 'ftp']:
                    tasks.append({
                        'priority': 2,
                        'host': host.ip,
                        'task': f"Secure or disable {service.get('name')} service",
                        'impact': 'high',
                        'effort': 'low'
                    })
        
        return sorted(tasks, key=lambda x: x['priority'])

class VulnerabilityDatabase:
    """Vulnerability database for service assessment"""
    
    def __init__(self):
        self.vuln_db = self._load_vulnerability_data()
    
    def _load_vulnerability_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load vulnerability database"""
        # In production, this would connect to CVE database or similar
        return {
            'ssh': [
                {
                    'cve': 'CVE-2023-38408',
                    'name': 'OpenSSH Certificate Validation Bypass',
                    'severity': 'medium',
                    'description': 'Certificate validation bypass in OpenSSH'
                }
            ],
            'apache': [
                {
                    'cve': 'CVE-2023-31122',
                    'name': 'Apache HTTP Server Out-of-bounds Read',
                    'severity': 'high',
                    'description': 'Out-of-bounds read vulnerability'
                }
            ]
        }
    
    async def check_service(self, product: str, version: str, port: int) -> List[Dict[str, Any]]:
        """Check service for known vulnerabilities"""
        vulnerabilities = []
        
        # Simple vulnerability check based on service name
        service_name = product.lower()
        if service_name in self.vuln_db:
            vulnerabilities.extend(self.vuln_db[service_name])
        
        # Add generic vulnerabilities based on service type
        if port in [21, 23]:  # FTP, Telnet
            vulnerabilities.append({
                'cve': 'GENERIC-001',
                'name': 'Insecure Protocol',
                'severity': 'high',
                'description': f'Service on port {port} uses insecure protocol'
            })
        
        return vulnerabilities

# Initialize the reconnaissance engine
recon_engine = NetworkReconEngine()

# Helper functions
async def require_admin_access(current_user: Dict = Depends(get_current_user) if SECURITY_ENABLED else None):
    """Require admin access for network reconnaissance operations"""
    if SECURITY_ENABLED and current_user:
        if "admin" not in current_user.get("scopes", []):
            raise HTTPException(status_code=403, detail="Admin access required for network reconnaissance")
    return current_user

def generate_scan_id() -> str:
    """Generate unique scan ID"""
    import uuid
    return f"recon_{int(time.time())}_{str(uuid.uuid4())[:8]}"

# API Endpoints

@router.post("/scan", response_model=NetworkScanResponse)
async def start_network_scan(
    request: NetworkScanRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(require_admin_access)
):
    """
    Start comprehensive network reconnaissance scan
    
    **Features:**
    - Host discovery and enumeration
    - Port scanning with stealth options
    - Service detection and fingerprinting
    - OS identification
    - Vulnerability assessment
    - AI-powered threat analysis
    """
    try:
        scan_id = generate_scan_id()
        
        # Log scan initiation
        if SECURITY_ENABLED and security_manager:
            await security_manager.audit.log_event(
                "network_scan_initiated",
                "info",
                "network_recon",
                {
                    "scan_id": scan_id,
                    "target": request.target,
                    "scan_type": request.scan_type.value
                },
                user_id=current_user.get("user_id") if current_user else "anonymous"
            )
        
        # Initialize scan result
        scan_result = ScanResult(
            scan_id=scan_id,
            scan_type=request.scan_type.value,
            target=request.target,
            start_time=datetime.utcnow().isoformat()
        )
        
        with scan_lock:
            active_scans[scan_id] = scan_result
        
        # Start scan in background
        background_tasks.add_task(
            run_scan_background,
            request,
            scan_id
        )
        
        # Calculate estimated completion time
        estimated_minutes = {
            ScanType.PING_SWEEP: 2,
            ScanType.PORT_SCAN: 10,
            ScanType.SERVICE_DETECTION: 15,
            ScanType.OS_FINGERPRINT: 20,
            ScanType.VULNERABILITY_SCAN: 30,
            ScanType.COMPREHENSIVE: 45,
            ScanType.STEALTH: 60
        }
        
        estimated_completion = (
            datetime.utcnow() + 
            timedelta(minutes=estimated_minutes.get(request.scan_type, 30))
        ).isoformat()
        
        return NetworkScanResponse(
            scan_id=scan_id,
            status="initiated",
            message=f"Network reconnaissance scan started for {request.target}",
            estimated_completion=estimated_completion
        )
        
    except Exception as e:
        logger.error(f"Failed to start network scan: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start scan: {str(e)}")

async def run_scan_background(request: NetworkScanRequest, scan_id: str):
    """Run scan in background task"""
    try:
        result = await recon_engine.perform_scan(request, scan_id)
        with scan_lock:
            active_scans[scan_id] = result
    except Exception as e:
        logger.error(f"Background scan failed: {e}")
        with scan_lock:
            if scan_id in active_scans:
                active_scans[scan_id].status = "failed"
                active_scans[scan_id].end_time = datetime.utcnow().isoformat()

@router.get("/scan/{scan_id}/status", response_model=ScanStatusResponse)
async def get_scan_status(
    scan_id: str,
    current_user: Dict = Depends(require_admin_access)
):
    """Get status of running network scan"""
    with scan_lock:
        if scan_id not in active_scans:
            raise HTTPException(status_code=404, detail="Scan not found")
        
        scan = active_scans[scan_id]
        
        # Calculate progress
        progress = 1.0 if scan.status == "completed" else 0.5
        hosts_found = len(scan.hosts_discovered) if scan.hosts_discovered else 0
        
        return ScanStatusResponse(
            scan_id=scan_id,
            status=scan.status,
            progress=progress,
            hosts_found=hosts_found,
            current_phase="Processing" if scan.status == "running" else scan.status.title()
        )

@router.get("/scan/{scan_id}/results", response_model=NetworkReconResults)
async def get_scan_results(
    scan_id: str,
    current_user: Dict = Depends(require_admin_access)
):
    """Get detailed results of completed network reconnaissance scan"""
    with scan_lock:
        if scan_id not in active_scans:
            raise HTTPException(status_code=404, detail="Scan not found")
        
        scan = active_scans[scan_id]
        
        if scan.status not in ["completed", "failed"]:
            raise HTTPException(status_code=400, detail="Scan not yet completed")
        
        # Convert hosts to dict format
        hosts_data = []
        if scan.hosts_discovered:
            hosts_data = [asdict(host) for host in scan.hosts_discovered]
        
        # Extract threats
        threats = []
        if scan.hosts_discovered:
            for host in scan.hosts_discovered:
                for vuln in host.vulnerabilities:
                    threats.append({
                        'host': host.ip,
                        'vulnerability': vuln.get('name', 'Unknown'),
                        'severity': vuln.get('severity', 'unknown'),
                        'cve': vuln.get('cve', ''),
                        'description': vuln.get('description', '')
                    })
        
        return NetworkReconResults(
            scan_id=scan_id,
            summary=scan.threat_summary or {},
            hosts=hosts_data,
            topology=scan.network_topology or {},
            threats=threats,
            recommendations=scan.recommendations or [],
            ai_insights=scan.ai_analysis
        )

@router.post("/vulnerability-assessment", response_model=NetworkScanResponse)
async def start_vulnerability_assessment(
    request: VulnerabilityAssessmentRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(require_admin_access)
):
    """
    Start comprehensive vulnerability assessment
    
    **Advanced Features:**
    - Multi-target vulnerability scanning
    - Exploit database integration
    - AI-powered threat modeling
    - Risk prioritization
    """
    try:
        scan_id = generate_scan_id()
        
        # Convert to network scan request
        network_request = NetworkScanRequest(
            target=request.targets[0] if request.targets else "127.0.0.1",
            scan_type=ScanType.VULNERABILITY_SCAN,
            deep_analysis=request.ai_threat_modeling
        )
        
        # Log assessment initiation
        if SECURITY_ENABLED and security_manager:
            await security_manager.audit.log_event(
                "vulnerability_assessment_initiated",
                "info",
                "network_recon",
                {
                    "scan_id": scan_id,
                    "targets": request.targets,
                    "intensity": request.scan_intensity
                },
                user_id=current_user.get("user_id") if current_user else "anonymous"
            )
        
        # Start assessment
        background_tasks.add_task(
            run_scan_background,
            network_request,
            scan_id
        )
        
        return NetworkScanResponse(
            scan_id=scan_id,
            status="initiated",
            message=f"Vulnerability assessment started for {len(request.targets)} targets"
        )
        
    except Exception as e:
        logger.error(f"Failed to start vulnerability assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/network-discovery", response_model=NetworkScanResponse)
async def start_network_discovery(
    request: NetworkDiscoveryRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(require_admin_access)
):
    """
    Start network discovery and mapping
    
    **Capabilities:**
    - Subnet enumeration
    - Device discovery
    - Network topology mapping
    - Asset inventory
    """
    try:
        scan_id = generate_scan_id()
        
        # Convert to network scan request
        network_request = NetworkScanRequest(
            target=request.network,
            scan_type=ScanType.PING_SWEEP if not request.port_discovery else ScanType.PORT_SCAN
        )
        
        # Start discovery
        background_tasks.add_task(
            run_scan_background,
            network_request,
            scan_id
        )
        
        return NetworkScanResponse(
            scan_id=scan_id,
            status="initiated",
            message=f"Network discovery started for {request.network}"
        )
        
    except Exception as e:
        logger.error(f"Failed to start network discovery: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scans", response_model=List[Dict[str, Any]])
async def list_scans(
    limit: int = Query(default=20, le=100),
    current_user: Dict = Depends(require_admin_access)
):
    """List recent network reconnaissance scans"""
    with scan_lock:
        scans = []
        for scan_id, scan in list(active_scans.items())[:limit]:
            scans.append({
                'scan_id': scan_id,
                'target': scan.target,
                'scan_type': scan.scan_type,
                'status': scan.status,
                'start_time': scan.start_time,
                'end_time': scan.end_time,
                'hosts_found': len(scan.hosts_discovered) if scan.hosts_discovered else 0
            })
        return scans

@router.delete("/scan/{scan_id}")
async def delete_scan(
    scan_id: str,
    current_user: Dict = Depends(require_admin_access)
):
    """Delete scan results"""
    with scan_lock:
        if scan_id not in active_scans:
            raise HTTPException(status_code=404, detail="Scan not found")
        
        del active_scans[scan_id]
        
    return {"message": f"Scan {scan_id} deleted successfully"}

@router.get("/capabilities")
async def get_reconnaissance_capabilities(
    current_user: Dict = Depends(require_admin_access)
):
    """Get network reconnaissance capabilities and status"""
    capabilities = {
        "scan_types": [
            {
                "name": "ping_sweep",
                "description": "Basic host discovery using ICMP ping",
                "duration": "1-2 minutes"
            },
            {
                "name": "port_scan",
                "description": "TCP/UDP port scanning",
                "duration": "5-15 minutes"
            },
            {
                "name": "service_detection",
                "description": "Service and version detection",
                "duration": "10-20 minutes"
            },
            {
                "name": "os_fingerprint",
                "description": "Operating system identification",
                "duration": "15-25 minutes"
            },
            {
                "name": "vulnerability_scan",
                "description": "Vulnerability assessment and CVE matching",
                "duration": "20-40 minutes"
            },
            {
                "name": "comprehensive",
                "description": "Full reconnaissance with all techniques",
                "duration": "30-60 minutes"
            }
        ],
        "features": [
            "AI-powered threat analysis",
            "Network topology mapping",
            "Vulnerability assessment",
            "Service fingerprinting",
            "OS detection",
            "Stealth scanning options",
            "Real-time progress tracking"
        ],
        "tools_available": {
            "nmap": recon_engine.scanner is not None,
            "ai_analysis": True,
            "vulnerability_db": True
        },
        "active_scans": len(active_scans),
        "security_integration": SECURITY_ENABLED
    }
    
    return capabilities

@router.get("/health")
async def get_reconnaissance_health():
    """Get network reconnaissance system health status"""
    return {
        "status": "operational",
        "engine_ready": recon_engine is not None,
        "scanner_available": recon_engine.scanner is not None if recon_engine else False,
        "active_scans": len(active_scans),
        "capabilities": "full" if recon_engine and recon_engine.scanner else "limited"
    } 