"""
Advanced Web Honeypot for HTTP/HTTPS Attack Detection
Comprehensive web application honeypot for detecting and analyzing web-based attacks
"""

import asyncio
import logging
import json
import ssl
import re
import base64
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import aiohttp
from aiohttp import web, web_request, web_response
import jinja2
import random
import string
import urllib.parse

# Import honeypot infrastructure
from security.honeypot_infrastructure import BaseHoneypot, HoneypotType, HoneypotEvent

logger = logging.getLogger(__name__)

class WebHoneypotServer:
    """Advanced web honeypot server with multiple attack detection capabilities"""
    
    def __init__(self, honeypot_id: str, port: int, database, intelligence_engine, 
                 ssl_context=None, enable_https=False):
        self.honeypot_id = honeypot_id
        self.port = port
        self.database = database
        self.intelligence_engine = intelligence_engine
        self.ssl_context = ssl_context
        self.enable_https = enable_https
        self.app = None
        self.site = None
        self.runner = None
        self.is_running = False
        
        # Attack detection patterns
        self.attack_patterns = self._load_attack_patterns()
        
        # Fake applications and content
        self.fake_apps = self._create_fake_applications()
        
        # Session tracking
        self.sessions = {}
        
        # Template engine for dynamic content
        self.template_env = jinja2.Environment(
            loader=jinja2.DictLoader(self._get_templates())
        )
        
    def _load_attack_patterns(self) -> Dict[str, List[str]]:
        """Load attack detection patterns"""
        return {
            'sql_injection': [
                r'union\s+select', r'or\s+1\s*=\s*1', r'drop\s+table', r'exec\s+xp_',
                r'sp_executesql', r'information_schema', r'sysobjects', r'sys\.tables',
                r'admin[\'\"]\s*--', r'\'\s*or\s*[\'\"]\s*1[\'\"]\s*=\s*[\'\"]\s*1',
                r'waitfor\s+delay', r'benchmark\s*\(', r'pg_sleep\s*\(', r'sleep\s*\(',
                r'load_file\s*\(', r'into\s+outfile', r'into\s+dumpfile'
            ],
            'xss': [
                r'<script[^>]*>', r'javascript\s*:', r'on\w+\s*=', r'eval\s*\(',
                r'document\.cookie', r'document\.write', r'window\.location',
                r'alert\s*\(', r'confirm\s*\(', r'prompt\s*\(',
                r'<iframe[^>]*>', r'<object[^>]*>', r'<embed[^>]*>',
                r'expression\s*\(', r'vbscript\s*:', r'<link[^>]*javascript'
            ],
            'command_injection': [
                r';\s*cat\s+', r'\|\s*wget\s+', r'&&\s*curl\s+', r'nc\s+-e',
                r'/bin/sh', r'/bin/bash', r'bash\s+-i', r'python\s+-c',
                r'perl\s+-e', r'ruby\s+-e', r'php\s+-r', r'exec\s*\(',
                r'system\s*\(', r'shell_exec\s*\(', r'passthru\s*\('
            ],
            'path_traversal': [
                r'\.\./.*\.\./.*\.\./', r'\.\.\\.*\.\.\\.*\.\.\\',
                r'%2e%2e%2f', r'%2e%2e/', r'%252e%252e%252f',
                r'/etc/passwd', r'/etc/shadow', r'c:\\windows\\system32',
                r'\.\.%2f', r'\.\.%5c', r'%c0%ae%c0%ae/'
            ],
            'rfi_lfi': [
                r'http://.*\.(txt|php|asp)', r'ftp://.*\.(txt|php|asp)',
                r'file:///', r'php://filter', r'php://input', r'data://',
                r'expect://', r'zip://', r'phar://'
            ],
            'xxe': [
                r'<!ENTITY.*SYSTEM', r'<!ENTITY.*PUBLIC', r'ENTITY.*file://',
                r'ENTITY.*http://', r'ENTITY.*ftp://', r'%.*ENTITY'
            ],
            'ldap_injection': [
                r'\(\|\(', r'\)\(', r'\*\)\(', r'\(\*\)',
                r'&\(objectClass=\*\)', r'\|\(mail=\*\)'
            ],
            'nosql_injection': [
                r'\$ne\s*:', r'\$gt\s*:', r'\$lt\s*:', r'\$regex\s*:',
                r'\$where\s*:', r'\$exists\s*:', r'true\s*,\s*true'
            ],
            'ssti': [
                r'\{\{.*\}\}', r'\{\%.*\%\}', r'\$\{.*\}',
                r'<%.*%>', r'#\{.*\}', r'\{\{7\*7\}\}',
                r'\{\{config\}\}', r'\{\{request\}\}'
            ]
        }
    
    def _create_fake_applications(self) -> Dict[str, Dict[str, Any]]:
        """Create fake web applications for the honeypot"""
        return {
            'admin': {
                'name': 'SutazAI Admin Panel',
                'paths': ['/admin', '/admin/', '/admin/login', '/admin/dashboard'],
                'requires_auth': True,
                'login_form': True
            },
            'api': {
                'name': 'SutazAI API',
                'paths': ['/api', '/api/', '/api/v1', '/api/v1/', '/api/docs'],
                'requires_auth': True,
                'api_endpoints': True
            },
            'wordpress': {
                'name': 'SutazAI Blog',
                'paths': ['/wp-admin', '/wp-login.php', '/wp-config.php', '/wp-content'],
                'requires_auth': True,
                'wordpress_like': True
            },
            'phpmyadmin': {
                'name': 'Database Administration',
                'paths': ['/phpmyadmin', '/pma', '/phpMyAdmin', '/dbadmin'],
                'requires_auth': True,
                'database_admin': True
            },
            'ftp': {
                'name': 'File Manager',
                'paths': ['/filemanager', '/files', '/uploads', '/ftp'],
                'requires_auth': True,
                'file_manager': True
            }
        }
    
    def _get_templates(self) -> Dict[str, str]:
        """Get Jinja2 templates for fake applications"""
        return {
            'base.html': '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} - SutazAI</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { border-bottom: 1px solid #eee; padding-bottom: 20px; margin-bottom: 20px; }
        .login-form { max-width: 400px; margin: 0 auto; }
        .form-group { margin-bottom: 15px; }
        .form-control { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
        .btn { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #0056b3; }
        .alert { padding: 10px; margin: 10px 0; border-radius: 4px; }
        .alert-danger { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; font-size: 12px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
        </div>
        {% block content %}{% endblock %}
        <div class="footer">
            <p>&copy; 2024 SutazAI System. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
            ''',
            
            'login.html': '''
{% extends "base.html" %}
{% block content %}
<div class="login-form">
    <h2>Login Required</h2>
    {% if error %}
    <div class="alert alert-danger">{{ error }}</div>
    {% endif %}
    <form method="POST" action="{{ action }}">
        <div class="form-group">
            <label for="username">Username:</label>
            <input type="text" id="username" name="username" class="form-control" required>
        </div>
        <div class="form-group">
            <label for="password">Password:</label>
            <input type="password" id="password" name="password" class="form-control" required>
        </div>
        <button type="submit" class="btn">Login</button>
    </form>
</div>
{% endblock %}
            ''',
            
            'dashboard.html': '''
{% extends "base.html" %}
{% block content %}
<h2>Dashboard</h2>
<div class="row">
    <div class="col">
        <h3>System Status</h3>
        <p>All systems operational</p>
        <ul>
            <li>AI Agents: {{ agent_count }} active</li>
            <li>Database: Connected</li>
            <li>Cache: {{ cache_status }}</li>
            <li>Queue: {{ queue_length }} jobs</li>
        </ul>
    </div>
</div>
<h3>Recent Activity</h3>
<table border="1" style="width: 100%; border-collapse: collapse;">
    <tr>
        <th>Time</th>
        <th>User</th>
        <th>Action</th>
        <th>Status</th>
    </tr>
    {% for activity in activities %}
    <tr>
        <td>{{ activity.time }}</td>
        <td>{{ activity.user }}</td>
        <td>{{ activity.action }}</td>
        <td>{{ activity.status }}</td>
    </tr>
    {% endfor %}
</table>
{% endblock %}
            ''',
            
            'api_docs.html': '''
{% extends "base.html" %}
{% block content %}
<h2>API Documentation</h2>
<h3>Authentication</h3>
<p>All API endpoints require authentication via Bearer token.</p>
<pre>Authorization: Bearer &lt;your-token&gt;</pre>

<h3>Endpoints</h3>
<h4>GET /api/v1/agents</h4>
<p>List all available AI agents</p>
<pre>{
  "agents": [
    {"id": "agent-001", "type": "senior-engineer", "status": "active"},
    {"id": "agent-002", "type": "qa-lead", "status": "idle"}
  ]
}</pre>

<h4>POST /api/v1/coordinator/task</h4>
<p>Submit a task to the coordinator</p>
<pre>{
  "task": "Analyze this code",
  "priority": "high",
  "agent_type": "senior-engineer"
}</pre>

<h4>GET /api/v1/status</h4>
<p>Get system status</p>
<pre>{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": "24d 6h 32m"
}</pre>
{% endblock %}
            ''',
            
            'error.html': '''
{% extends "base.html" %}
{% block content %}
<h2>Error {{ error_code }}</h2>
<p>{{ error_message }}</p>
<p><a href="/">Return to home</a></p>
{% endblock %}
            '''
        }
    
    async def start(self):
        """Start the web honeypot server"""
        self.app = web.Application()
        self._setup_routes()
        
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        # Configure SSL if enabled
        ssl_context = None
        if self.enable_https and self.ssl_context:
            ssl_context = self.ssl_context
        
        self.site = web.TCPSite(
            self.runner, 
            '0.0.0.0', 
            self.port, 
            ssl_context=ssl_context
        )
        
        await self.site.start()
        self.is_running = True
        
        protocol = "HTTPS" if self.enable_https else "HTTP"
        logger.info(f"{protocol} honeypot started on port {self.port}")
    
    async def stop(self):
        """Stop the web honeypot server"""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        
        self.is_running = False
        logger.info(f"Web honeypot stopped on port {self.port}")
    
    def _setup_routes(self):
        """Setup web application routes"""
        # Root paths
        self.app.router.add_get('/', self.handle_root)
        self.app.router.add_get('/index.html', self.handle_root)
        self.app.router.add_get('/index.php', self.handle_root)
        
        # Admin paths
        self.app.router.add_get('/admin', self.handle_admin)
        self.app.router.add_get('/admin/', self.handle_admin)
        self.app.router.add_get('/admin/login', self.handle_admin_login)
        self.app.router.add_post('/admin/login', self.handle_admin_login_post)
        self.app.router.add_get('/admin/dashboard', self.handle_admin_dashboard)
        
        # API paths
        self.app.router.add_get('/api', self.handle_api_root)
        self.app.router.add_get('/api/', self.handle_api_root)
        self.app.router.add_get('/api/v1', self.handle_api_v1)
        self.app.router.add_get('/api/v1/', self.handle_api_v1)
        self.app.router.add_get('/api/v1/agents', self.handle_api_agents)
        self.app.router.add_post('/api/v1/coordinator/task', self.handle_api_task)
        self.app.router.add_get('/api/v1/status', self.handle_api_status)
        self.app.router.add_get('/api/docs', self.handle_api_docs)
        
        # WordPress-like paths
        self.app.router.add_get('/wp-admin', self.handle_wp_admin)
        self.app.router.add_get('/wp-login.php', self.handle_wp_login)
        self.app.router.add_post('/wp-login.php', self.handle_wp_login_post)
        self.app.router.add_get('/wp-config.php', self.handle_wp_config)
        
        # Database admin paths
        self.app.router.add_get('/phpmyadmin', self.handle_phpmyadmin)
        self.app.router.add_get('/pma', self.handle_phpmyadmin)
        self.app.router.add_get('/phpMyAdmin', self.handle_phpmyadmin)
        self.app.router.add_post('/phpmyadmin/index.php', self.handle_phpmyadmin_login)
        
        # File manager paths
        self.app.router.add_get('/filemanager', self.handle_filemanager)
        self.app.router.add_get('/files', self.handle_filemanager)
        self.app.router.add_get('/uploads', self.handle_filemanager)
        
        # Common attack targets
        self.app.router.add_get('/.env', self.handle_env_file)
        self.app.router.add_get('/config.php', self.handle_config_file)
        self.app.router.add_get('/config.json', self.handle_config_file)
        self.app.router.add_get('/backup.sql', self.handle_backup_file)
        self.app.router.add_get('/database.sql', self.handle_backup_file)
        
        # Catch-all for other requests
        self.app.router.add_route('*', '/{path:.*}', self.handle_catchall)
    
    async def _log_interaction(self, request: web_request.Request, event_type: str, 
                              payload: str = "", severity: str = "medium", **kwargs):
        """Log web interaction"""
        try:
            # Get client information
            client_ip = request.remote
            user_agent = request.headers.get('User-Agent', '')
            
            # Create event
            event_id = f"{self.honeypot_id}_{int(datetime.utcnow().timestamp())}_{random.randint(1000, 9999)}"
            
            event = HoneypotEvent(
                id=event_id,
                timestamp=datetime.utcnow(),
                honeypot_id=self.honeypot_id,
                honeypot_type=HoneypotType.HTTPS.value if self.enable_https else HoneypotType.HTTP.value,
                source_ip=client_ip,
                source_port=0,  # Not available in HTTP
                destination_port=self.port,
                event_type=event_type,
                payload=payload,
                severity=severity,
                user_agent=user_agent,
                **kwargs
            )
            
            # Analyze for threats
            analysis = await self.intelligence_engine.analyze_event(event)
            event.threat_indicators = analysis['indicators']
            
            # Store event
            self.database.store_event(event)
            
            # Update attacker profile
            profile = self.database.get_attacker_profile(client_ip)
            if not profile:
                from security.honeypot_infrastructure import AttackerProfile
                profile = AttackerProfile(
                    source_ip=client_ip,
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    total_attempts=1,
                    honeypots_hit={self.honeypot_id},
                    attack_patterns=[analysis['attack_classification']],
                    credentials_tried=[],
                    user_agents={user_agent} if user_agent else set(),
                    threat_score=0.0
                )
            else:
                profile.last_seen = datetime.utcnow()
                profile.total_attempts += 1
                profile.honeypots_hit.add(self.honeypot_id)
                if analysis['attack_classification'] not in profile.attack_patterns:
                    profile.attack_patterns.append(analysis['attack_classification'])
                if user_agent:
                    profile.user_agents.add(user_agent)
            
            # Recalculate threat score
            profile.threat_score = self.intelligence_engine.calculate_attacker_threat_score(profile)
            self.database.update_attacker_profile(profile)
            
            logger.info(f"Web honeypot interaction: {event_type} from {client_ip}")
            
        except Exception as e:
            logger.error(f"Error logging web interaction: {e}")
    
    def _detect_attacks(self, request: web_request.Request) -> Tuple[List[str], str]:
        """Detect attacks in HTTP request"""
        attacks_detected = []
        max_severity = "low"
        
        # Get request data
        path = request.path_qs
        query_string = request.query_string
        user_agent = request.headers.get('User-Agent', '')
        
        # Combine all request data for analysis
        request_data = f"{path} {query_string} {user_agent}".lower()
        
        # Check for attack patterns
        for attack_type, patterns in self.attack_patterns.items():
            for pattern in patterns:
                if re.search(pattern, request_data, re.IGNORECASE):
                    attacks_detected.append(attack_type)
                    
                    # Determine severity
                    if attack_type in ['sql_injection', 'command_injection', 'xxe']:
                        max_severity = 'high'
                    elif attack_type in ['xss', 'path_traversal', 'rfi_lfi']:
                        if max_severity != 'high':
                            max_severity = 'medium'
                    break
        
        # Check for suspicious user agents
        suspicious_agents = [
            'sqlmap', 'nmap', 'nikto', 'gobuster', 'dirb', 'wfuzz', 
            'burp', 'owasp', 'metasploit', 'hydra', 'medusa'
        ]
        
        for agent in suspicious_agents:
            if agent in user_agent.lower():
                attacks_detected.append('automated_scanner')
                max_severity = 'high'
                break
        
        return attacks_detected, max_severity
    
    async def handle_root(self, request: web_request.Request) -> web_response.Response:
        """Handle root path requests"""
        attacks, severity = self._detect_attacks(request)
        
        await self._log_interaction(
            request, "web_root_access", 
            payload=f"Path: {request.path_qs}",
            severity=severity,
            attack_vector=attacks[0] if attacks else None
        )
        
        template = self.template_env.get_template('base.html')
        content = template.render(
            title="SutazAI Management Portal",
            content='''
            <h2>Welcome to SutazAI Management Portal</h2>
            <p>This is the central management interface for the SutazAI system.</p>
            <h3>Quick Links</h3>
            <ul>
                <li><a href="/admin">Administrative Panel</a></li>
                <li><a href="/api">API Documentation</a></li>
                <li><a href="/wp-admin">Content Management</a></li>
            </ul>
            <h3>System Information</h3>
            <p>Version: 1.2.3<br>
            Status: Operational<br>
            Uptime: 15 days, 4 hours</p>
            '''
        )
        
        return web_response.Response(text=content, content_type='text/html')
    
    async def handle_admin(self, request: web_request.Request) -> web_response.Response:
        """Handle admin panel access"""
        attacks, severity = self._detect_attacks(request)
        
        await self._log_interaction(
            request, "admin_panel_access", 
            payload=f"Path: {request.path_qs}",
            severity="medium" if not attacks else severity,
            attack_vector=attacks[0] if attacks else None
        )
        
        # Redirect to login
        return web_response.Response(
            status=302,
            headers={'Location': '/admin/login'}
        )
    
    async def handle_admin_login(self, request: web_request.Request) -> web_response.Response:
        """Handle admin login page"""
        attacks, severity = self._detect_attacks(request)
        
        await self._log_interaction(
            request, "admin_login_page", 
            payload=f"Path: {request.path_qs}",
            severity=severity,
            attack_vector=attacks[0] if attacks else None
        )
        
        template = self.template_env.get_template('login.html')
        content = template.render(
            title="Admin Login",
            action="/admin/login"
        )
        
        return web_response.Response(text=content, content_type='text/html')
    
    async def handle_admin_login_post(self, request: web_request.Request) -> web_response.Response:
        """Handle admin login POST"""
        try:
            post_data = await request.post()
            username = post_data.get('username', '')
            password = post_data.get('password', '')
            
            attacks, severity = self._detect_attacks(request)
            
            await self._log_interaction(
                request, "admin_login_attempt", 
                payload=f"Username: {username}, Password: {'*' * len(password)}",
                severity="high",
                attack_vector=attacks[0] if attacks else 'credential_harvesting',
                credentials={"username": username, "password": password}
            )
            
            # Always reject login but make it look realistic
            template = self.template_env.get_template('login.html')
            content = template.render(
                title="Admin Login",
                action="/admin/login",
                error="Invalid username or password"
            )
            
            return web_response.Response(text=content, content_type='text/html')
            
        except Exception as e:
            logger.error(f"Error handling admin login POST: {e}")
            return web_response.Response(status=500, text="Internal Server Error")
    
    async def handle_admin_dashboard(self, request: web_request.Request) -> web_response.Response:
        """Handle admin dashboard (fake)"""
        attacks, severity = self._detect_attacks(request)
        
        await self._log_interaction(
            request, "admin_dashboard_access", 
            payload=f"Path: {request.path_qs}",
            severity="high",  # Unauthorized access attempt
            attack_vector=attacks[0] if attacks else 'unauthorized_access'
        )
        
        # Return fake dashboard with realistic content
        template = self.template_env.get_template('dashboard.html')
        content = template.render(
            title="Admin Dashboard",
            agent_count=random.randint(5, 15),
            cache_status="Active",
            queue_length=random.randint(0, 50),
            activities=[
                {
                    'time': (datetime.utcnow() - timedelta(minutes=random.randint(1, 60))).strftime('%H:%M'),
                    'user': random.choice(['admin', 'operator', 'system']),
                    'action': random.choice(['Login', 'Task Created', 'Agent Started', 'Config Updated']),
                    'status': random.choice(['Success', 'Completed', 'Running'])
                } for _ in range(5)
            ]
        )
        
        return web_response.Response(text=content, content_type='text/html')
    
    async def handle_api_root(self, request: web_request.Request) -> web_response.Response:
        """Handle API root access"""
        attacks, severity = self._detect_attacks(request)
        
        await self._log_interaction(
            request, "api_root_access", 
            payload=f"Path: {request.path_qs}",
            severity=severity,
            attack_vector=attacks[0] if attacks else None
        )
        
        return web_response.json_response({
            "name": "SutazAI API",
            "version": "1.0.0",
            "status": "operational",
            "endpoints": [
                "/api/v1/agents",
                "/api/v1/coordinator",
                "/api/v1/status"
            ],
            "documentation": "/api/docs"
        })
    
    async def handle_api_v1(self, request: web_request.Request) -> web_response.Response:
        """Handle API v1 root"""
        attacks, severity = self._detect_attacks(request)
        
        await self._log_interaction(
            request, "api_v1_access", 
            payload=f"Path: {request.path_qs}",
            severity=severity,
            attack_vector=attacks[0] if attacks else None
        )
        
        return web_response.json_response({
            "version": "1.0.0",
            "endpoints": {
                "agents": "/api/v1/agents",
                "coordinator": "/api/v1/coordinator",
                "status": "/api/v1/status"
            }
        })
    
    async def handle_api_agents(self, request: web_request.Request) -> web_response.Response:
        """Handle API agents endpoint"""
        attacks, severity = self._detect_attacks(request)
        
        await self._log_interaction(
            request, "api_agents_access", 
            payload=f"Path: {request.path_qs}",
            severity="high" if attacks else "medium",  # API access should be authenticated
            attack_vector=attacks[0] if attacks else 'unauthorized_api_access'
        )
        
        # Return fake agent data
        agents = [
            {"id": f"agent-{i:03d}", "type": random.choice([
                "ai-senior-engineer", "ai-qa-team-lead", "ai-system-architect",
                "ai-senior-backend-developer", "ai-senior-frontend-developer"
            ]), "status": random.choice(["active", "idle", "busy"])}
            for i in range(1, random.randint(5, 15))
        ]
        
        return web_response.json_response({
            "agents": agents,
            "total": len(agents),
            "active": len([a for a in agents if a["status"] == "active"])
        })
    
    async def handle_api_task(self, request: web_request.Request) -> web_response.Response:
        """Handle API task submission"""
        try:
            body = await request.text()
            attacks, severity = self._detect_attacks(request)
            
            await self._log_interaction(
                request, "api_task_submission", 
                payload=f"Body: {body[:500]}",
                severity="high",  # Unauthorized API usage
                attack_vector=attacks[0] if attacks else 'unauthorized_api_access'
            )
            
            return web_response.json_response({
                "error": "Authentication required",
                "code": 401,
                "message": "Valid API token required"
            }, status=401)
            
        except Exception as e:
            logger.error(f"Error handling API task: {e}")
            return web_response.json_response({"error": "Bad Request"}, status=400)
    
    async def handle_api_status(self, request: web_request.Request) -> web_response.Response:
        """Handle API status endpoint"""
        attacks, severity = self._detect_attacks(request)
        
        await self._log_interaction(
            request, "api_status_access", 
            payload=f"Path: {request.path_qs}",
            severity=severity,
            attack_vector=attacks[0] if attacks else None
        )
        
        return web_response.json_response({
            "status": "healthy",
            "version": "1.0.0",
            "uptime": f"{random.randint(1, 30)}d {random.randint(0, 23)}h {random.randint(0, 59)}m",
            "agents": {
                "total": random.randint(5, 15),
                "active": random.randint(3, 10),
                "idle": random.randint(0, 5)
            },
            "system": {
                "cpu_usage": f"{random.randint(10, 80)}%",
                "memory_usage": f"{random.randint(30, 90)}%",
                "disk_usage": f"{random.randint(20, 70)}%"
            }
        })
    
    async def handle_api_docs(self, request: web_request.Request) -> web_response.Response:
        """Handle API documentation"""
        attacks, severity = self._detect_attacks(request)
        
        await self._log_interaction(
            request, "api_docs_access", 
            payload=f"Path: {request.path_qs}",
            severity=severity,
            attack_vector=attacks[0] if attacks else None
        )
        
        template = self.template_env.get_template('api_docs.html')
        content = template.render(title="API Documentation")
        
        return web_response.Response(text=content, content_type='text/html')
    
    async def handle_wp_admin(self, request: web_request.Request) -> web_response.Response:
        """Handle WordPress admin access"""
        attacks, severity = self._detect_attacks(request)
        
        await self._log_interaction(
            request, "wordpress_admin_access", 
            payload=f"Path: {request.path_qs}",
            severity="medium",
            attack_vector=attacks[0] if attacks else None
        )
        
        return web_response.Response(
            status=302,
            headers={'Location': '/wp-login.php'}
        )
    
    async def handle_wp_login(self, request: web_request.Request) -> web_response.Response:
        """Handle WordPress login page"""
        attacks, severity = self._detect_attacks(request)
        
        await self._log_interaction(
            request, "wordpress_login_page", 
            payload=f"Path: {request.path_qs}",
            severity=severity,
            attack_vector=attacks[0] if attacks else None
        )
        
        # Return WordPress-like login page
        wp_login_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Log In ‹ SutazAI Blog — WordPress</title>
    <style>
        body { background: #f1f1f1; font-family: sans-serif; }
        .login { width: 320px; margin: 100px auto; }
        .login h1 { text-align: center; margin-bottom: 20px; }
        .login form { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,.13); }
        .login label { display: block; margin-bottom: 5px; }
        .login input[type="text"], .login input[type="password"] { width: 100%; padding: 10px; margin-bottom: 15px; border: 1px solid #ddd; }
        .login input[type="submit"] { background: #0073aa; color: white; padding: 10px 20px; border: none; border-radius: 3px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="login">
        <h1>SutazAI Blog</h1>
        <form method="post" action="/wp-login.php">
            <label for="user_login">Username or Email Address</label>
            <input type="text" name="log" id="user_login" required>
            <label for="user_pass">Password</label>
            <input type="password" name="pwd" id="user_pass" required>
            <input type="submit" name="wp-submit" value="Log In">
        </form>
    </div>
</body>
</html>
        '''
        
        return web_response.Response(text=wp_login_html, content_type='text/html')
    
    async def handle_wp_login_post(self, request: web_request.Request) -> web_response.Response:
        """Handle WordPress login POST"""
        try:
            post_data = await request.post()
            username = post_data.get('log', '')
            password = post_data.get('pwd', '')
            
            attacks, severity = self._detect_attacks(request)
            
            await self._log_interaction(
                request, "wordpress_login_attempt", 
                payload=f"Username: {username}, Password: {'*' * len(password)}",
                severity="high",
                attack_vector=attacks[0] if attacks else 'credential_harvesting',
                credentials={"username": username, "password": password}
            )
            
            # Return error page
            error_html = '''
<!DOCTYPE html>
<html>
<head><title>WordPress Error</title></head>
<body>
    <h1>ERROR</h1>
    <p><strong>ERROR:</strong> Invalid username or password.</p>
    <p><a href="/wp-login.php">&larr; Back to login</a></p>
</body>
</html>
            '''
            
            return web_response.Response(text=error_html, content_type='text/html')
            
        except Exception as e:
            logger.error(f"Error handling WordPress login POST: {e}")
            return web_response.Response(status=500, text="Internal Server Error")
    
    async def handle_wp_config(self, request: web_request.Request) -> web_response.Response:
        """Handle WordPress config file access"""
        attacks, severity = self._detect_attacks(request)
        
        await self._log_interaction(
            request, "wordpress_config_access", 
            payload=f"Path: {request.path_qs}",
            severity="high",  # Sensitive file access
            attack_vector=attacks[0] if attacks else 'sensitive_file_access'
        )
        
        return web_response.Response(status=403, text="Forbidden")
    
    async def handle_phpmyadmin(self, request: web_request.Request) -> web_response.Response:
        """Handle phpMyAdmin access"""
        attacks, severity = self._detect_attacks(request)
        
        await self._log_interaction(
            request, "phpmyadmin_access", 
            payload=f"Path: {request.path_qs}",
            severity="medium",
            attack_vector=attacks[0] if attacks else None
        )
        
        # Return phpMyAdmin-like login page
        pma_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>phpMyAdmin</title>
    <style>
        body { font-family: sans-serif; background: #f4f4f4; margin: 0; padding: 20px; }
        .container { max-width: 400px; margin: 50px auto; background: white; padding: 30px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .logo { text-align: center; margin-bottom: 30px; font-size: 24px; color: #0066cc; }
        input[type="text"], input[type="password"] { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; }
        input[type="submit"] { background: #0066cc; color: white; padding: 10px 20px; border: none; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <div class="logo">phpMyAdmin</div>
        <form method="post" action="/phpmyadmin/index.php">
            <label>Username:</label>
            <input type="text" name="pma_username" required>
            <label>Password:</label>
            <input type="password" name="pma_password" required>
            <label>Server:</label>
            <input type="text" name="pma_servername" value="localhost">
            <input type="submit" value="Go">
        </form>
    </div>
</body>
</html>
        '''
        
        return web_response.Response(text=pma_html, content_type='text/html')
    
    async def handle_phpmyadmin_login(self, request: web_request.Request) -> web_response.Response:
        """Handle phpMyAdmin login POST"""
        try:
            post_data = await request.post()
            username = post_data.get('pma_username', '')
            password = post_data.get('pma_password', '')
            server = post_data.get('pma_servername', '')
            
            attacks, severity = self._detect_attacks(request)
            
            await self._log_interaction(
                request, "phpmyadmin_login_attempt", 
                payload=f"Username: {username}, Server: {server}",
                severity="high",
                attack_vector=attacks[0] if attacks else 'database_access_attempt',
                credentials={"username": username, "password": password, "server": server}
            )
            
            # Return access denied
            return web_response.Response(
                text="Access denied for user",
                status=403
            )
            
        except Exception as e:
            logger.error(f"Error handling phpMyAdmin login: {e}")
            return web_response.Response(status=500, text="Internal Server Error")
    
    async def handle_filemanager(self, request: web_request.Request) -> web_response.Response:
        """Handle file manager access"""
        attacks, severity = self._detect_attacks(request)
        
        await self._log_interaction(
            request, "filemanager_access", 
            payload=f"Path: {request.path_qs}",
            severity="medium",
            attack_vector=attacks[0] if attacks else None
        )
        
        # Return fake file manager
        fm_html = '''
<!DOCTYPE html>
<html>
<head><title>File Manager</title></head>
<body>
    <h1>File Manager</h1>
    <p>Access denied. Please log in.</p>
    <form method="post">
        <input type="text" name="username" placeholder="Username">
        <input type="password" name="password" placeholder="Password">
        <input type="submit" value="Login">
    </form>
</body>
</html>
        '''
        
        return web_response.Response(text=fm_html, content_type='text/html')
    
    async def handle_env_file(self, request: web_request.Request) -> web_response.Response:
        """Handle .env file access"""
        attacks, severity = self._detect_attacks(request)
        
        await self._log_interaction(
            request, "env_file_access", 
            payload=f"Path: {request.path_qs}",
            severity="high",  # Very sensitive file
            attack_vector=attacks[0] if attacks else 'sensitive_file_access'
        )
        
        return web_response.Response(status=404, text="Not Found")
    
    async def handle_config_file(self, request: web_request.Request) -> web_response.Response:
        """Handle config file access"""
        attacks, severity = self._detect_attacks(request)
        
        await self._log_interaction(
            request, "config_file_access", 
            payload=f"Path: {request.path_qs}",
            severity="high",
            attack_vector=attacks[0] if attacks else 'sensitive_file_access'
        )
        
        return web_response.Response(status=403, text="Forbidden")
    
    async def handle_backup_file(self, request: web_request.Request) -> web_response.Response:
        """Handle backup file access"""
        attacks, severity = self._detect_attacks(request)
        
        await self._log_interaction(
            request, "backup_file_access", 
            payload=f"Path: {request.path_qs}",
            severity="high",
            attack_vector=attacks[0] if attacks else 'sensitive_file_access'
        )
        
        return web_response.Response(status=404, text="Not Found")
    
    async def handle_catchall(self, request: web_request.Request) -> web_response.Response:
        """Handle all other requests"""
        attacks, severity = self._detect_attacks(request)
        
        await self._log_interaction(
            request, "unknown_path_access", 
            payload=f"Path: {request.path_qs}",
            severity=severity,
            attack_vector=attacks[0] if attacks else None
        )
        
        # Return 404 but log the attempt
        template = self.template_env.get_template('error.html')
        content = template.render(
            title="404 Not Found",
            error_code=404,
            error_message="The requested page could not be found."
        )
        
        return web_response.Response(text=content, content_type='text/html', status=404)

class WebHoneypotManager:
    """Manager for web honeypots"""
    
    def __init__(self, database, intelligence_engine):
        self.database = database
        self.intelligence_engine = intelligence_engine
        self.honeypots = {}
        
    async def deploy_http_honeypot(self, port: int = 8080) -> bool:
        """Deploy HTTP honeypot"""
        try:
            honeypot_id = f"web_http_{port}"
            
            honeypot = WebHoneypotServer(
                honeypot_id, port, self.database, self.intelligence_engine
            )
            
            await honeypot.start()
            self.honeypots[honeypot_id] = honeypot
            
            logger.info(f"HTTP honeypot deployed on port {port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy HTTP honeypot: {e}")
            return False
    
    async def deploy_https_honeypot(self, port: int = 8443) -> bool:
        """Deploy HTTPS honeypot"""
        try:
            # Create self-signed SSL certificate
            ssl_context = await self._create_ssl_context()
            
            honeypot_id = f"web_https_{port}"
            
            honeypot = WebHoneypotServer(
                honeypot_id, port, self.database, self.intelligence_engine,
                ssl_context=ssl_context, enable_https=True
            )
            
            await honeypot.start()
            self.honeypots[honeypot_id] = honeypot
            
            logger.info(f"HTTPS honeypot deployed on port {port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy HTTPS honeypot: {e}")
            return False
    
    async def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context with self-signed certificate"""
        # This is a simplified SSL context creation
        # In production, you'd want proper certificate management
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        
        # For now, use a simple self-signed approach
        # In real deployment, generate proper certificates
        try:
            cert_dir = Path("/opt/sutazaiapp/backend/security/certs")
            cert_dir.mkdir(parents=True, exist_ok=True)
            
            cert_file = cert_dir / "honeypot.crt"
            key_file = cert_dir / "honeypot.key"
            
            # Generate self-signed certificate if it doesn't exist
            if not cert_file.exists() or not key_file.exists():
                await self._generate_self_signed_cert(str(cert_file), str(key_file))
            
            ssl_context.load_cert_chain(str(cert_file), str(key_file))
            
        except Exception as e:
            logger.warning(f"Could not create SSL context: {e}")
            ssl_context = None
        
        return ssl_context
    
    async def _generate_self_signed_cert(self, cert_file: str, key_file: str):
        """Generate self-signed certificate"""
        try:
            # Use openssl to generate certificate
            cmd = [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", key_file, "-out", cert_file,
                "-days", "365", "-nodes",
                "-subj", "/C=US/ST=State/L=City/O=SutazAI/CN=localhost"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            if process.returncode == 0:
                logger.info("Generated self-signed SSL certificate")
            else:
                logger.error("Failed to generate SSL certificate")
                
        except Exception as e:
            logger.error(f"Error generating SSL certificate: {e}")
    
    async def stop_all(self):
        """Stop all web honeypots"""
        for honeypot in self.honeypots.values():
            try:
                await honeypot.stop()
            except Exception as e:
                logger.error(f"Error stopping honeypot: {e}")
        
        self.honeypots.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all web honeypots"""
        return {
            "active_honeypots": len(self.honeypots),
            "honeypots": {
                honeypot_id: {
                    "port": honeypot.port,
                    "ssl_enabled": honeypot.enable_https,
                    "running": honeypot.is_running
                }
                for honeypot_id, honeypot in self.honeypots.items()
            }
        }

# Global web honeypot manager instance
web_honeypot_manager = None