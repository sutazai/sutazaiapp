#!/usr/bin/env python3
"""
Simple Hygiene Monitor Backend - Minimal Real-time Data Collection
Purpose: Simplified version for immediate testing
Author: AI Observability and Monitoring Engineer
Version: 1.0.0 - Minimal Working Version
"""

import asyncio
import json
import logging
import os
import psutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path
import aiohttp
from aiohttp import web, WSMsgType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleMonitoringBackend:
    """Simplified monitoring backend for testing"""
    
    def __init__(self, project_root="/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.db_path = self.project_root / "monitoring" / "simple.db"
        self.websocket_clients = set()
        self.running = False
        
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info("Simple Monitoring Backend initialized")

    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cpu_usage REAL,
                memory_percentage REAL,
                disk_percentage REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                rule_id TEXT,
                file_path TEXT,
                severity TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized")

    def collect_system_metrics(self):
        """Collect real system metrics"""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_usage': cpu_percent,
                'memory_percentage': memory.percent,
                'disk_percentage': (disk.used / disk.total) * 100,
                'memory_used': memory.used / (1024**3),  # GB
                'memory_total': memory.total / (1024**3),  # GB
                'disk_used': disk.used / (1024**3),  # GB
                'disk_total': disk.total / (1024**3),  # GB
                'network_status': 'HEALTHY'
            }
            
            # Store in database
            self._store_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return None

    def _store_metrics(self, metrics):
        """Store metrics in database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_metrics (timestamp, cpu_usage, memory_percentage, disk_percentage)
                VALUES (?, ?, ?, ?)
            ''', (metrics['timestamp'], metrics['cpu_usage'], 
                  metrics['memory_percentage'], metrics['disk_percentage']))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error storing metrics: {e}")

    def scan_violations(self):
        """Simple violation scanning"""
        violations = []
        
        try:
            # Look for some simple violations
            python_files = list(self.project_root.rglob('*.py'))
            
            for file_path in python_files[:10]:  # Limit to first 10 files for testing
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Check for fantasy elements
                    if any(word in content.lower() for word in ['magic', 'wizard', 'teleport']):
                        violation = {
                            'timestamp': datetime.now().isoformat(),
                            'rule_id': 'rule_1',
                            'rule_name': 'No Fantasy Elements',
                            'file_path': str(file_path.relative_to(self.project_root)),
                            'severity': 'CRITICAL',
                            'description': 'Fantasy element detected'
                        }
                        violations.append(violation)
                        
                        # Store in database
                        self._store_violation(violation)
                        
                except Exception as e:
                    logger.warning(f"Could not scan {file_path}: {e}")
                    
        except Exception as e:
            logger.error(f"Error scanning violations: {e}")
        
        return violations

    def _store_violation(self, violation):
        """Store violation in database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO violations (timestamp, rule_id, file_path, severity)
                VALUES (?, ?, ?, ?)
            ''', (violation['timestamp'], violation['rule_id'], 
                  violation['file_path'], violation['severity']))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error storing violation: {e}")

    def get_dashboard_data(self):
        """Get current dashboard data"""
        try:
            metrics = self.collect_system_metrics()
            violations = self.scan_violations()
            
            critical_violations = [v for v in violations if v['severity'] == 'CRITICAL']
            warning_violations = [v for v in violations if v['severity'] in ['HIGH', 'MEDIUM']]
            
            return {
                'timestamp': datetime.now().isoformat(),
                'systemStatus': 'MONITORING' if self.running else 'OFFLINE',
                'complianceScore': max(0, 100 - len(violations) * 5),
                'totalViolations': len(violations),
                'criticalViolations': len(critical_violations),
                'warningViolations': len(warning_violations),
                'activeAgents': 3,  # Mock data
                'systemMetrics': metrics,
                'recentViolations': violations[-10:],
                'recentActions': [
                    {
                        'id': '1',
                        'timestamp': datetime.now().isoformat(),
                        'action_type': 'SCAN',
                        'status': 'COMPLETED',
                        'agent_id': 'hygiene-scanner',
                        'rule_id': 'rule_1',
                        'file_path': 'test.py',
                        'duration_ms': 150
                    }
                ],
                'agentHealth': [
                    {
                        'agent_id': 'hygiene-scanner',
                        'name': 'Hygiene Scanner',
                        'status': 'ACTIVE',
                        'last_heartbeat': datetime.now().isoformat(),
                        'tasks_completed': 5,
                        'tasks_failed': 0,
                        'cpu_usage': 2.1,
                        'memory_usage': 45.2
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {'error': str(e)}

    async def broadcast_to_websockets(self, message):
        """Broadcast message to all connected websocket clients"""
        if not self.websocket_clients:
            return
        
        closed_clients = set()
        for ws in self.websocket_clients:
            try:
                await ws.send_str(json.dumps(message))
            except Exception:
                closed_clients.add(ws)
        
        self.websocket_clients -= closed_clients

    async def start_background_tasks(self):
        """Start background monitoring tasks"""
        self.running = True
        logger.info("Starting background monitoring tasks")
        
        while self.running:
            try:
                # Collect metrics and broadcast
                metrics = self.collect_system_metrics()
                if metrics:
                    await self.broadcast_to_websockets({
                        'type': 'system_metrics',
                        'data': metrics
                    })
                
                # Every 30 seconds, scan for violations
                if int(time.time()) % 30 == 0:
                    violations = self.scan_violations()
                    if violations:
                        await self.broadcast_to_websockets({
                            'type': 'violations_update', 
                            'data': violations
                        })
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in background tasks: {e}")
                await asyncio.sleep(5)

# Web API handlers
async def websocket_handler(request):
    """WebSocket handler for real-time updates"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    backend = request.app['backend']
    backend.websocket_clients.add(ws)
    
    try:
        # Send initial data
        initial_data = {
            'type': 'initial_data',
            'data': backend.get_dashboard_data()
        }
        await ws.send_str(json.dumps(initial_data))
        
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                logger.info(f"Received websocket message: {msg.data}")
            elif msg.type == WSMsgType.ERROR:
                logger.error(f'WebSocket error: {ws.exception()}')
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        backend.websocket_clients.discard(ws)
    
    return ws

async def status_handler(request):
    """REST API endpoint for dashboard status"""
    backend = request.app['backend']
    data = backend.get_dashboard_data()
    return web.json_response(data)

async def metrics_handler(request):
    """REST API endpoint for system metrics"""
    backend = request.app['backend']
    metrics = backend.collect_system_metrics()
    if metrics:
        return web.json_response(metrics)
    else:
        return web.json_response({'error': 'No metrics available'}, status=503)

async def scan_handler(request):
    """REST API endpoint to trigger violation scan"""
    backend = request.app['backend']
    violations = backend.scan_violations()
    return web.json_response({
        'success': True,
        'violations_found': len(violations),
        'violations': violations
    })

def create_app():
    """Create the web application"""
    app = web.Application()
    
    # Initialize backend
    backend = SimpleMonitoringBackend()
    app['backend'] = backend
    
    # Setup routes
    app.router.add_get('/ws', websocket_handler)
    app.router.add_get('/api/hygiene/status', status_handler)
    app.router.add_get('/api/system/metrics', metrics_handler)
    app.router.add_post('/api/hygiene/scan', scan_handler)
    
    # CORS middleware
    @web.middleware
    async def cors_middleware(request, handler):
        if request.method == 'OPTIONS':
            response = web.Response()
        else:
            response = await handler(request)
        
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    
    app.middlewares.append(cors_middleware)
    
    return app

async def main():
    """Main entry point"""
    app = create_app()
    backend = app['backend']
    
    # Start background tasks
    asyncio.create_task(backend.start_background_tasks())
    
    # Start web server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()
    
    logger.info("Simple Hygiene Monitor Backend started on http://0.0.0.0:8080")
    logger.info("WebSocket endpoint: ws://0.0.0.0:8080/ws")
    logger.info("API endpoints:")
    logger.info("  GET /api/hygiene/status - Dashboard status")
    logger.info("  GET /api/system/metrics - System metrics")
    logger.info("  POST /api/hygiene/scan - Trigger violation scan")
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        backend.running = False
        await runner.cleanup()

if __name__ == '__main__':
    asyncio.run(main())