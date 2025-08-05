#!/usr/bin/env python3
"""
Network Security Analyzer for SutazAI System
Comprehensive analysis of network segmentation and security
"""

import subprocess
import json
import re
from datetime import datetime
import socket
import ipaddress

class NetworkSecurityAnalyzer:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'network_analysis': {},
            'firewall_rules': {},
            'docker_networks': {},
            'security_issues': [],
            'recommendations': []
        }
    
    def analyze_docker_networks(self):
        """Analyze Docker network configuration"""
        try:
            result = subprocess.run(
                ['docker', 'network', 'ls', '--format', 'json'],
                capture_output=True, text=True, check=True
            )
            
            networks = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    networks.append(json.loads(line))
            
            network_details = {}
            for network in networks:
                network_id = network['ID']
                network_name = network['Name']
                
                # Get detailed network info
                inspect_result = subprocess.run(
                    ['docker', 'network', 'inspect', network_id],
                    capture_output=True, text=True, check=True
                )
                
                network_info = json.loads(inspect_result.stdout)[0]
                network_details[network_name] = {
                    'id': network_id,
                    'driver': network_info.get('Driver'),
                    'scope': network_info.get('Scope'),
                    'ipam': network_info.get('IPAM', {}),
                    'containers': network_info.get('Containers', {}),
                    'options': network_info.get('Options', {}),
                    'labels': network_info.get('Labels', {})
                }
            
            self.results['docker_networks'] = network_details
            return network_details
            
        except subprocess.CalledProcessError as e:
            print(f"Error analyzing Docker networks: {e}")
            return {}
    
    def analyze_firewall_rules(self):
        """Analyze system firewall configuration"""
        firewall_info = {
            'iptables': {},
            'ufw': {},
            'firewalld': {}
        }
        
        # Check iptables
        try:
            result = subprocess.run(
                ['iptables', '-L', '-n', '-v'],
                capture_output=True, text=True, check=True
            )
            firewall_info['iptables']['rules'] = result.stdout
            firewall_info['iptables']['active'] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            firewall_info['iptables']['active'] = False
        
        # Check UFW
        try:
            result = subprocess.run(
                ['ufw', 'status', 'verbose'],
                capture_output=True, text=True, check=True
            )
            firewall_info['ufw']['status'] = result.stdout
            firewall_info['ufw']['active'] = 'Status: active' in result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            firewall_info['ufw']['active'] = False
        
        # Check firewalld
        try:
            result = subprocess.run(
                ['firewall-cmd', '--state'],
                capture_output=True, text=True, check=True
            )
            firewall_info['firewalld']['active'] = 'running' in result.stdout
            
            if firewall_info['firewalld']['active']:
                zones_result = subprocess.run(
                    ['firewall-cmd', '--list-all-zones'],
                    capture_output=True, text=True, check=True
                )
                firewall_info['firewalld']['zones'] = zones_result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            firewall_info['firewalld']['active'] = False
        
        self.results['firewall_rules'] = firewall_info
        return firewall_info
    
    def analyze_network_interfaces(self):
        """Analyze network interfaces and routing"""
        network_info = {}
        
        # Get network interfaces
        try:
            result = subprocess.run(
                ['ip', 'addr', 'show'],
                capture_output=True, text=True, check=True
            )
            network_info['interfaces'] = result.stdout
        except subprocess.CalledProcessError:
            pass
        
        # Get routing table
        try:
            result = subprocess.run(
                ['ip', 'route', 'show'],
                capture_output=True, text=True, check=True
            )
            network_info['routes'] = result.stdout
        except subprocess.CalledProcessError:
            pass
        
        # Get listening ports
        try:
            result = subprocess.run(
                ['netstat', '-tlnp'],
                capture_output=True, text=True, check=True
            )
            network_info['listening_ports'] = result.stdout
        except subprocess.CalledProcessError:
            pass
        
        self.results['network_analysis'] = network_info
        return network_info
    
    def identify_security_issues(self):
        """Identify network security issues"""
        issues = []
        
        # Check Docker network security
        docker_networks = self.results.get('docker_networks', {})
        
        # Issue 1: Default bridge network usage
        for network_name, network_info in docker_networks.items():
            if network_name == 'bridge' and network_info.get('containers'):
                issues.append({
                    'severity': 'MEDIUM',
                    'category': 'Network Segmentation',
                    'issue': 'Containers using default bridge network',
                    'description': 'Default bridge network provides less isolation',
                    'recommendation': 'Use custom bridge networks for better isolation'
                })
        
        # Issue 2: Host network mode
        for network_name, network_info in docker_networks.items():
            if network_name == 'host' and network_info.get('containers'):
                issues.append({
                    'severity': 'HIGH',
                    'category': 'Network Security',
                    'issue': 'Container using host network mode',
                    'description': 'Host network mode bypasses Docker network isolation',
                    'recommendation': 'Use bridge or custom networks instead of host mode'
                })
        
        # Check firewall configuration
        firewall_info = self.results.get('firewall_rules', {})
        
        # Issue 3: No active firewall
        if not any(firewall_info.get(fw, {}).get('active', False) for fw in ['iptables', 'ufw', 'firewalld']):
            issues.append({
                'severity': 'HIGH',
                'category': 'Firewall',
                'issue': 'No active firewall detected',
                'description': 'System has no active firewall protection',
                'recommendation': 'Configure and enable a firewall (UFW, iptables, or firewalld)'
            })
        
        # Issue 4: Analyze listening ports
        network_info = self.results.get('network_analysis', {})
        listening_ports = network_info.get('listening_ports', '')
        
        exposed_ports = []
        for line in listening_ports.split('\n'):
            if '0.0.0.0:' in line:
                match = re.search(r'0\.0\.0\.0:(\d+)', line)
                if match:
                    port = int(match.group(1))
                    if port > 1024:  # Non-privileged ports
                        exposed_ports.append(port)
        
        if len(exposed_ports) > 10:
            issues.append({
                'severity': 'MEDIUM',
                'category': 'Port Exposure',
                'issue': f'{len(exposed_ports)} ports exposed on all interfaces',
                'description': 'Many services exposed on 0.0.0.0 (all interfaces)',
                'recommendation': 'Bind services to localhost (127.0.0.1) when possible'
            })
        
        # Issue 5: Check for common insecure ports
        insecure_ports = [21, 23, 80, 135, 139, 445, 1433, 3306, 5432]
        exposed_insecure = [port for port in exposed_ports if port in insecure_ports]
        
        if exposed_insecure:
            issues.append({
                'severity': 'HIGH',
                'category': 'Insecure Services',
                'issue': f'Insecure services exposed: {exposed_insecure}',
                'description': 'Services known for security vulnerabilities are exposed',
                'recommendation': 'Secure or disable these services, use encrypted alternatives'
            })
        
        self.results['security_issues'] = issues
        return issues
    
    def test_network_connectivity(self):
        """Test internal network connectivity and isolation"""
        connectivity_tests = {}
        
        # Test Docker network connectivity
        docker_networks = self.results.get('docker_networks', {})
        
        for network_name, network_info in docker_networks.items():
            if network_name in ['bridge', 'host', 'none']:
                continue
                
            containers = network_info.get('containers', {})
            if len(containers) > 1:
                # Test if containers can communicate within the network
                container_ids = list(containers.keys())
                connectivity_tests[network_name] = {
                    'containers': len(containers),
                    'isolated': False  # Would need actual connectivity tests
                }
        
        # Test external connectivity
        test_hosts = ['8.8.8.8', '1.1.1.1']
        external_connectivity = {}
        
        for host in test_hosts:
            try:
                result = subprocess.run(
                    ['ping', '-c', '1', '-W', '3', host],
                    capture_output=True, text=True, check=True
                )
                external_connectivity[host] = True
            except subprocess.CalledProcessError:
                external_connectivity[host] = False
        
        connectivity_tests['external'] = external_connectivity
        return connectivity_tests
    
    def generate_network_security_recommendations(self):
        """Generate network security recommendations"""
        recommendations = []
        
        issues = self.results.get('security_issues', [])
        
        # Priority recommendations based on issues found
        if any(issue['category'] == 'Firewall' for issue in issues):
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'Firewall Configuration',
                'title': 'Enable and Configure Firewall',
                'description': 'System lacks active firewall protection',
                'implementation': [
                    'Install and enable UFW: sudo ufw enable',
                    'Configure default deny policy: sudo ufw default deny incoming',
                    'Allow only necessary ports: sudo ufw allow <port>',
                    'Enable logging: sudo ufw logging on'
                ]
            })
        
        if any(issue['category'] == 'Network Segmentation' for issue in issues):
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Network Segmentation',
                'title': 'Implement Network Segmentation',
                'description': 'Improve container network isolation',
                'implementation': [
                    'Create custom bridge networks for different service tiers',
                    'Use Docker network policies to restrict inter-container communication',
                    'Implement network namespaces for additional isolation',
                    'Deploy network monitoring for traffic analysis'
                ]
            })
        
        if any(issue['category'] == 'Port Exposure' for issue in issues):
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Port Security',
                'title': 'Reduce Port Exposure',
                'description': 'Minimize attack surface by reducing exposed ports',
                'implementation': [
                    'Bind services to localhost (127.0.0.1) when possible',
                    'Use reverse proxy for external access',
                    'Implement port-based access controls',
                    'Regular port scanning and monitoring'
                ]
            })
        
        # Always recommend these best practices
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Network Monitoring',
            'title': 'Implement Network Monitoring',
            'description': 'Deploy comprehensive network monitoring',
            'implementation': [
                'Deploy network intrusion detection system (Suricata/Snort)',
                'Implement traffic analysis and logging',
                'Configure network anomaly detection',
                'Set up network security alerting'
            ]
        })
        
        self.results['recommendations'] = recommendations
        return recommendations
    
    def calculate_network_security_score(self):
        """Calculate network security score"""
        issues = self.results.get('security_issues', [])
        
        if not issues:
            return 10.0
        
        severity_weights = {
            'CRITICAL': 3.0,
            'HIGH': 2.0,
            'MEDIUM': 1.0,
            'LOW': 0.5
        }
        
        total_deduction = sum(severity_weights.get(issue['severity'], 1.0) for issue in issues)
        
        # Base score of 10, deduct based on issues
        score = max(0.0, 10.0 - total_deduction)
        return round(score, 1)
    
    def run_comprehensive_analysis(self):
        """Run complete network security analysis"""
        print("=" * 60)
        print("Network Security Analysis")
        print("=" * 60)
        
        # Analyze Docker networks
        print("[*] Analyzing Docker networks...")
        docker_networks = self.analyze_docker_networks()
        print(f"[+] Found {len(docker_networks)} Docker networks")
        
        # Analyze firewall configuration
        print("[*] Analyzing firewall configuration...")
        firewall_info = self.analyze_firewall_rules()
        active_firewalls = [fw for fw, info in firewall_info.items() if info.get('active')]
        print(f"[+] Active firewalls: {active_firewalls or 'None'}")
        
        # Analyze network interfaces
        print("[*] Analyzing network interfaces...")
        network_info = self.analyze_network_interfaces()
        
        # Test network connectivity
        print("[*] Testing network connectivity...")
        connectivity = self.test_network_connectivity()
        
        # Identify security issues
        print("[*] Identifying security issues...")
        issues = self.identify_security_issues()
        print(f"[+] Found {len(issues)} security issues")
        
        # Generate recommendations
        recommendations = self.generate_network_security_recommendations()
        
        # Calculate security score
        security_score = self.calculate_network_security_score()
        self.results['security_score'] = security_score
        
        print(f"\n[*] Network security analysis complete!")
        print(f"[*] Security Score: {security_score}/10.0")
        print(f"[*] Issues found: {len(issues)}")
        
        return self.results
    
    def save_results(self, filename='network_security_analysis.json'):
        """Save analysis results to file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"[*] Results saved to {filename}")

def main():
    analyzer = NetworkSecurityAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    analyzer.save_results('/opt/sutazaiapp/network_security_analysis.json')
    
    # Print summary
    print("\n" + "=" * 60)
    print("NETWORK SECURITY ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Security Score: {results['security_score']}/10.0")
    print(f"Docker Networks: {len(results['docker_networks'])}")
    print(f"Security Issues: {len(results['security_issues'])}")
    
    # Group issues by severity
    severity_counts = {}
    for issue in results['security_issues']:
        sev = issue['severity']
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = severity_counts.get(severity, 0)
        if count > 0:
            print(f"  - {severity}: {count}")
    
    if results['recommendations']:
        print(f"\nTop Recommendations:")
        for i, rec in enumerate(results['recommendations'][:3], 1):
            print(f"{i}. [{rec['priority']}] {rec['title']}")
            print(f"   {rec['description']}")

if __name__ == "__main__":
    main()