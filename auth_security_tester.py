#!/usr/bin/env python3
"""
Authentication and Authorization Security Tester
Comprehensive testing of authentication mechanisms in SutazAI system
"""

import requests
import json
import base64
import time
from datetime import datetime
import jwt
from urllib.parse import urlencode
import subprocess

class AuthSecurityTester:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'auth_services': {},
            'vulnerabilities': [],
            'test_results': {},
            'security_score': 0.0
        }
        
        # Define auth service endpoints
        self.auth_endpoints = {
            'keycloak': {
                'base_url': 'http://localhost:10050',
                'admin_url': 'http://localhost:10050/admin/',
                'auth_url': 'http://localhost:10050/realms/sutazai/protocol/openid-connect/auth',
                'token_url': 'http://localhost:10050/realms/sutazai/protocol/openid-connect/token'
            },
            'kong': {
                'proxy_url': 'http://localhost:10051',
                'admin_url': 'http://localhost:10052'
            },
            'jwt_service': {
                'url': 'http://localhost:10054'
            },
            'vault': {
                'url': 'http://localhost:10053'
            }
        }
    
    def test_service_availability(self):
        """Test if authentication services are available"""
        print("[*] Testing authentication service availability...")
        
        for service_name, endpoints in self.auth_endpoints.items():
            service_status = {
                'available': False,
                'response_time': None,
                'status_code': None,
                'error': None
            }
            
            try:
                # Test primary endpoint
                primary_url = endpoints.get('base_url') or endpoints.get('url')
                if primary_url:
                    start_time = time.time()
                    response = requests.get(primary_url, timeout=5)
                    response_time = time.time() - start_time
                    
                    service_status.update({
                        'available': True,
                        'response_time': round(response_time, 3),
                        'status_code': response.status_code
                    })
            
            except requests.exceptions.RequestException as e:
                service_status['error'] = str(e)
            
            self.results['auth_services'][service_name] = service_status
            print(f"[+] {service_name}: {'Available' if service_status['available'] else 'Unavailable'}")
    
    def test_keycloak_security(self):
        """Test Keycloak authentication security"""
        print("[*] Testing Keycloak security...")
        
        keycloak_tests = {
            'admin_console_access': False,
            'default_credentials': False,
            'realm_access': False,
            'user_registration': False,
            'token_validation': False
        }
        
        base_url = self.auth_endpoints['keycloak']['base_url']
        
        # Test 1: Admin console accessibility
        try:
            admin_response = requests.get(f"{base_url}/admin/", timeout=5)
            keycloak_tests['admin_console_access'] = admin_response.status_code == 200
            
            if admin_response.status_code == 200:
                self.results['vulnerabilities'].append({
                    'service': 'keycloak',
                    'severity': 'HIGH',
                    'type': 'UNPROTECTED_ADMIN',
                    'description': 'Keycloak admin console accessible without authentication',
                    'recommendation': 'Secure admin console with proper authentication'
                })
        except:
            pass
        
        # Test 2: Default credentials
        try:
            token_response = requests.post(
                f"{base_url}/realms/master/protocol/openid-connect/token",
                data={
                    'grant_type': 'password',
                    'client_id': 'admin-cli',
                    'username': 'admin',
                    'password': 'admin'
                },
                timeout=5
            )
            
            if token_response.status_code == 200:
                keycloak_tests['default_credentials'] = True
                self.results['vulnerabilities'].append({
                    'service': 'keycloak',
                    'severity': 'CRITICAL',
                    'type': 'DEFAULT_CREDENTIALS',
                    'description': 'Keycloak using default admin credentials',
                    'recommendation': 'Change default admin password immediately'
                })
        except:
            pass
        
        # Test 3: Realm configuration access
        try:
            realm_response = requests.get(f"{base_url}/realms/sutazai", timeout=5)
            keycloak_tests['realm_access'] = realm_response.status_code == 200
        except:
            pass
        
        # Test 4: User registration endpoint
        try:
            register_response = requests.post(
                f"{base_url}/realms/sutazai/login-actions/registration",
                data={'username': 'testuser', 'password': 'testpass'},
                timeout=5
            )
            keycloak_tests['user_registration'] = register_response.status_code != 404
        except:
            pass
        
        self.results['test_results']['keycloak'] = keycloak_tests
    
    def test_kong_security(self):
        """Test Kong API Gateway security"""
        print("[*] Testing Kong API Gateway security...")
        
        kong_tests = {
            'admin_api_access': False,
            'proxy_bypass': False,
            'rate_limiting': False,
            'auth_plugins': False
        }
        
        admin_url = self.auth_endpoints['kong']['admin_url']
        proxy_url = self.auth_endpoints['kong']['proxy_url']
        
        # Test 1: Admin API access
        try:
            admin_response = requests.get(f"{admin_url}/", timeout=5)
            kong_tests['admin_api_access'] = admin_response.status_code == 200
            
            if admin_response.status_code == 200:
                self.results['vulnerabilities'].append({
                    'service': 'kong',
                    'severity': 'HIGH',
                    'type': 'EXPOSED_ADMIN_API',
                    'description': 'Kong Admin API accessible without authentication',
                    'recommendation': 'Secure Admin API with authentication and network restrictions'
                })
        except:
            pass
        
        # Test 2: Check for authentication plugins
        try:
            plugins_response = requests.get(f"{admin_url}/plugins", timeout=5)
            if plugins_response.status_code == 200:
                plugins_data = plugins_response.json()
                auth_plugins = [p for p in plugins_data.get('data', []) 
                              if p.get('name') in ['jwt', 'oauth2', 'key-auth', 'basic-auth']]
                kong_tests['auth_plugins'] = len(auth_plugins) > 0
                
                if not auth_plugins:
                    self.results['vulnerabilities'].append({
                        'service': 'kong',
                        'severity': 'MEDIUM',
                        'type': 'NO_AUTH_PLUGINS',
                        'description': 'No authentication plugins configured in Kong',
                        'recommendation': 'Configure authentication plugins for API protection'
                    })
        except:
            pass
        
        # Test 3: Rate limiting
        try:
            # Make multiple rapid requests to test rate limiting
            responses = []
            for i in range(10):
                response = requests.get(f"{proxy_url}/", timeout=2)
                responses.append(response.status_code)
                time.sleep(0.1)
            
            # Check if any requests were rate limited (429 status)
            kong_tests['rate_limiting'] = 429 in responses
            
            if not kong_tests['rate_limiting']:
                self.results['vulnerabilities'].append({
                    'service': 'kong',
                    'severity': 'MEDIUM',
                    'type': 'NO_RATE_LIMITING',
                    'description': 'No rate limiting configured',
                    'recommendation': 'Configure rate limiting to prevent abuse'
                })
        except:
            pass
        
        self.results['test_results']['kong'] = kong_tests
    
    def test_jwt_service_security(self):
        """Test JWT service security"""
        print("[*] Testing JWT service security...")
        
        jwt_tests = {
            'service_accessible': False,
            'weak_secret': False,
            'token_validation': False,
            'algorithm_confusion': False
        }
        
        jwt_url = self.auth_endpoints['jwt_service']['url']
        
        # Test 1: Service accessibility
        try:
            health_response = requests.get(f"{jwt_url}/health", timeout=5)
            jwt_tests['service_accessible'] = health_response.status_code == 200
        except:
            pass
        
        # Test 2: Try to generate token with common weak secrets
        weak_secrets = ['secret', 'jwt_secret', 'sutazai', '123456', 'password']
        
        for secret in weak_secrets:
            try:
                # Create a test JWT token
                test_payload = {
                    'user': 'test',
                    'role': 'admin',
                    'exp': int(time.time()) + 3600
                }
                
                test_token = jwt.encode(test_payload, secret, algorithm='HS256')
                
                # Try to validate with the service
                validation_response = requests.post(
                    f"{jwt_url}/validate",
                    json={'token': test_token},
                    timeout=5
                )
                
                if validation_response.status_code == 200:
                    jwt_tests['weak_secret'] = True
                    self.results['vulnerabilities'].append({
                        'service': 'jwt_service',
                        'severity': 'HIGH',
                        'type': 'WEAK_JWT_SECRET',
                        'description': f'JWT service accepts tokens signed with weak secret: {secret}',
                        'recommendation': 'Use strong, randomly generated JWT secrets'
                    })
                    break
            except:
                continue
        
        # Test 3: Algorithm confusion (none algorithm)
        try:
            none_payload = {
                'user': 'admin',
                'role': 'superadmin',
                'exp': int(time.time()) + 3600
            }
            
            # Create unsigned JWT (algorithm: none)
            header = base64.urlsafe_b64encode(json.dumps({'alg': 'none', 'typ': 'JWT'}).encode()).decode().rstrip('=')
            payload = base64.urlsafe_b64encode(json.dumps(none_payload).encode()).decode().rstrip('=')
            none_token = f"{header}.{payload}."
            
            validation_response = requests.post(
                f"{jwt_url}/validate",
                json={'token': none_token},
                timeout=5
            )
            
            if validation_response.status_code == 200:
                jwt_tests['algorithm_confusion'] = True
                self.results['vulnerabilities'].append({
                    'service': 'jwt_service',
                    'severity': 'CRITICAL',
                    'type': 'ALGORITHM_CONFUSION',
                    'description': 'JWT service accepts unsigned tokens (algorithm: none)',
                    'recommendation': 'Reject tokens with "none" algorithm'
                })
        except:
            pass
        
        self.results['test_results']['jwt_service'] = jwt_tests
    
    def test_vault_security(self):
        """Test HashiCorp Vault security"""
        print("[*] Testing Vault security...")
        
        vault_tests = {
            'vault_accessible': False,
            'vault_sealed': True,
            'auth_methods': [],
            'policies': []
        }
        
        vault_url = self.auth_endpoints['vault']['url']
        
        # Test 1: Vault accessibility and status
        try:
            health_response = requests.get(f"{vault_url}/v1/sys/health", timeout=5)
            vault_tests['vault_accessible'] = health_response.status_code == 200
            
            if health_response.status_code == 200:
                health_data = health_response.json()
                vault_tests['vault_sealed'] = health_data.get('sealed', True)
                
                if not vault_tests['vault_sealed']:
                    # Vault is unsealed, check for security issues
                    self.results['vulnerabilities'].append({
                        'service': 'vault',
                        'severity': 'MEDIUM',
                        'type': 'VAULT_UNSEALED',
                        'description': 'Vault is unsealed and accessible',
                        'recommendation': 'Ensure Vault is properly secured and access is restricted'
                    })
        except:
            pass
        
        # Test 2: Check auth methods (if accessible)
        try:
            auth_response = requests.get(f"{vault_url}/v1/sys/auth", timeout=5)
            if auth_response.status_code == 200:
                auth_data = auth_response.json()
                vault_tests['auth_methods'] = list(auth_data.keys())
        except:
            pass
        
        self.results['test_results']['vault'] = vault_tests
    
    def test_session_security(self):
        """Test session management security"""
        print("[*] Testing session security...")
        
        session_tests = {
            'session_fixation': False,
            'session_hijacking': False,
            'secure_cookies': False,
            'session_timeout': False
        }
        
        # Test against services that might use sessions
        test_urls = [
            'http://localhost:10050',  # Keycloak
            'http://localhost:10002',  # Neo4j
        ]
        
        for url in test_urls:
            try:
                # Test 1: Check for secure cookie attributes
                response = requests.get(url, timeout=5)
                
                set_cookie_headers = response.headers.get('Set-Cookie', '')
                if set_cookie_headers:
                    if 'Secure' not in set_cookie_headers:
                        session_tests['secure_cookies'] = False
                    if 'HttpOnly' not in set_cookie_headers:
                        session_tests['secure_cookies'] = False
                    if 'SameSite' not in set_cookie_headers:
                        session_tests['secure_cookies'] = False
                
            except:
                continue
        
        if not session_tests['secure_cookies']:
            self.results['vulnerabilities'].append({
                'service': 'general',
                'severity': 'MEDIUM',
                'type': 'INSECURE_COOKIES',
                'description': 'Cookies lack security attributes (Secure, HttpOnly, SameSite)',
                'recommendation': 'Configure cookies with proper security attributes'
            })
        
        self.results['test_results']['session_security'] = session_tests
    
    def calculate_auth_security_score(self):
        """Calculate authentication security score"""
        vulnerabilities = self.results['vulnerabilities']
        
        if not vulnerabilities:
            return 10.0
        
        severity_weights = {
            'CRITICAL': 3.0,
            'HIGH': 2.0,
            'MEDIUM': 1.0,
            'LOW': 0.5
        }
        
        total_deduction = sum(severity_weights.get(vuln['severity'], 1.0) for vuln in vulnerabilities)
        
        # Base score of 10, deduct based on vulnerabilities
        score = max(0.0, 10.0 - total_deduction)
        return round(score, 1)
    
    def generate_auth_recommendations(self):
        """Generate authentication security recommendations"""
        recommendations = []
        
        vulnerabilities = self.results['vulnerabilities']
        
        # Critical recommendations
        critical_vulns = [v for v in vulnerabilities if v['severity'] == 'CRITICAL']
        if critical_vulns:
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'Authentication',
                'title': 'Fix Critical Authentication Vulnerabilities',
                'description': f'Found {len(critical_vulns)} critical authentication issues',
                'actions': [v['recommendation'] for v in critical_vulns]
            })
        
        # Service-specific recommendations
        if not self.results['auth_services'].get('keycloak', {}).get('available'):
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Authentication Infrastructure',
                'title': 'Deploy Authentication Service',
                'description': 'No centralized authentication service detected',
                'actions': [
                    'Deploy and configure Keycloak or similar identity provider',
                    'Integrate all services with centralized authentication',
                    'Implement Single Sign-On (SSO)'
                ]
            })
        
        # General security improvements
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Authentication Security',
            'title': 'Implement Authentication Best Practices',
            'description': 'Enhance overall authentication security',
            'actions': [
                'Enable multi-factor authentication (MFA)',
                'Implement strong password policies',
                'Configure session timeouts',
                'Enable audit logging for authentication events',
                'Implement account lockout policies'
            ]
        })
        
        return recommendations
    
    def run_comprehensive_auth_test(self):
        """Run complete authentication security test"""
        print("=" * 60)
        print("Authentication & Authorization Security Test")
        print("=" * 60)
        
        # Test service availability
        self.test_service_availability()
        
        # Test individual services
        if self.results['auth_services'].get('keycloak', {}).get('available'):
            self.test_keycloak_security()
        
        if self.results['auth_services'].get('kong', {}).get('available'):
            self.test_kong_security()
        
        if self.results['auth_services'].get('jwt_service', {}).get('available'):
            self.test_jwt_service_security()
        
        if self.results['auth_services'].get('vault', {}).get('available'):
            self.test_vault_security()
        
        # Test session security
        self.test_session_security()
        
        # Calculate security score
        security_score = self.calculate_auth_security_score()
        self.results['security_score'] = security_score
        
        # Generate recommendations
        recommendations = self.generate_auth_recommendations()
        self.results['recommendations'] = recommendations
        
        print(f"\n[*] Authentication security test complete!")
        print(f"[*] Security Score: {security_score}/10.0")
        print(f"[*] Vulnerabilities found: {len(self.results['vulnerabilities'])}")
        
        return self.results
    
    def save_results(self, filename='auth_security_test_results.json'):
        """Save test results to file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"[*] Results saved to {filename}")

def main():
    tester = AuthSecurityTester()
    results = tester.run_comprehensive_auth_test()
    tester.save_results('/opt/sutazaiapp/auth_security_test_results.json')
    
    # Print summary
    print("\n" + "=" * 60)
    print("AUTHENTICATION SECURITY TEST SUMMARY")
    print("=" * 60)
    print(f"Security Score: {results['security_score']}/10.0")
    
    # Service availability
    available_services = [name for name, info in results['auth_services'].items() if info.get('available')]
    print(f"Available Auth Services: {available_services}")
    
    # Vulnerabilities by severity
    vulnerabilities = results['vulnerabilities']
    severity_counts = {}
    for vuln in vulnerabilities:
        sev = vuln['severity']
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    print(f"Vulnerabilities Found: {len(vulnerabilities)}")
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = severity_counts.get(severity, 0)
        if count > 0:
            print(f"  - {severity}: {count}")
    
    # Top recommendations
    if results.get('recommendations'):
        print(f"\nTop Recommendations:")
        for i, rec in enumerate(results['recommendations'][:3], 1):
            print(f"{i}. [{rec['priority']}] {rec['title']}")
            print(f"   {rec['description']}")

if __name__ == "__main__":
    main()