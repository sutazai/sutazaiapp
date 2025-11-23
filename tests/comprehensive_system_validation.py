#!/usr/bin/env python3
"""
Comprehensive System Validation Script
Purpose: Test all SutazAI Platform components systematically
Created: 2025-11-15 00:00:00 UTC
"""

import asyncio
import aiohttp
import json
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s UTC - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

SCRIPT_START = datetime.now(timezone.utc)
logger.info(f"System validation started at {SCRIPT_START.isoformat()}")


class SystemValidator:
    """Comprehensive system validation with detailed reporting"""
    
    def __init__(self):
        self.results = []
        self.failures = []
        
    async def test_backend_health(self, session: aiohttp.ClientSession) -> bool:
        """Test backend API health endpoints"""
        logger.info("Testing Backend API Health...")
        try:
            async with session.get('http://localhost:10200/health') as resp:
                data = await resp.json()
                assert resp.status == 200
                assert data['status'] == 'healthy'
                logger.info("✅ Backend health: PASS")
                return True
        except Exception as e:
            logger.error(f"❌ Backend health: FAIL - {e}")
            self.failures.append(f"Backend health: {e}")
            return False
    
    async def test_backend_services(self, session: aiohttp.ClientSession) -> Dict:
        """Test all backend service connections"""
        logger.info("Testing Backend Service Connections...")
        try:
            async with session.get('http://localhost:10200/health/detailed') as resp:
                data = await resp.json()
                services = data.get('services', {})
                
                service_status = {}
                for service, status in services.items():
                    service_status[service] = status
                    symbol = "✅" if status else "❌"
                    logger.info(f"{symbol} {service}: {'Connected' if status else 'Disconnected'}")
                    if not status:
                        self.failures.append(f"Service {service} is disconnected")
                
                logger.info(f"Service connectivity: {data['healthy_count']}/{data['total_services']}")
                return service_status
        except Exception as e:
            logger.error(f"❌ Backend services test: FAIL - {e}")
            self.failures.append(f"Backend services: {e}")
            return {}
    
    async def test_ai_agent(self, session: aiohttp.ClientSession, name: str, port: int) -> bool:
        """Test individual AI agent endpoint"""
        try:
            # Test health endpoint
            async with session.get(f'http://localhost:{port}/health', timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"✅ {name} (port {port}): Healthy")
                    return True
                else:
                    logger.warning(f"⚠️ {name} (port {port}): Unexpected status {resp.status}")
                    return False
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ {name} (port {port}): Timeout")
            return False
        except Exception as e:
            logger.error(f"❌ {name} (port {port}): FAIL - {e}")
            self.failures.append(f"Agent {name}: {e}")
            return False
    
    async def test_all_ai_agents(self, session: aiohttp.ClientSession) -> Dict[str, bool]:
        """Test all AI agent endpoints"""
        logger.info("Testing AI Agent Endpoints...")
        agents = {
            'Letta': 11401,
            'CrewAI': 11403,
            'Aider': 11404,
            'LangChain': 11405,
            'ShellGPT': 11413,
            'Documind': 11414,
            'FinRobot': 11410,
            'GPT-Engineer': 11416
        }
        
        results = {}
        for name, port in agents.items():
            results[name] = await self.test_ai_agent(session, name, port)
        
        passed = sum(1 for v in results.values() if v)
        logger.info(f"AI Agents: {passed}/{len(agents)} responding")
        return results
    
    async def test_vector_database(self, session: aiohttp.ClientSession, name: str, url: str) -> bool:
        """Test vector database connectivity"""
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    logger.info(f"✅ {name}: Connected")
                    return True
                else:
                    logger.warning(f"⚠️ {name}: Status {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ {name}: FAIL - {e}")
            self.failures.append(f"Vector DB {name}: {e}")
            return False
    
    async def test_all_vector_databases(self, session: aiohttp.ClientSession) -> Dict[str, bool]:
        """Test all vector databases"""
        logger.info("Testing Vector Databases...")
        dbs = {
            'ChromaDB': 'http://localhost:10100/api/v2/heartbeat',
            'Qdrant': 'http://localhost:10101/collections',
            'FAISS': 'http://localhost:10103/health'
        }
        
        results = {}
        for name, url in dbs.items():
            results[name] = await self.test_vector_database(session, name, url)
        
        passed = sum(1 for v in results.values() if v)
        logger.info(f"Vector Databases: {passed}/{len(dbs)} accessible")
        return results
    
    async def test_mcp_bridge(self, session: aiohttp.ClientSession) -> bool:
        """Test MCP Bridge functionality"""
        logger.info("Testing MCP Bridge...")
        try:
            async with session.get('http://localhost:11100/health') as resp:
                if resp.status == 200:
                    logger.info("✅ MCP Bridge: Healthy")
                    
                    # Test agents endpoint
                    async with session.get('http://localhost:11100/agents') as agents_resp:
                        agents_data = await agents_resp.json()
                        logger.info(f"   MCP registered agents: {len(agents_data)}")
                        return True
                else:
                    logger.warning(f"⚠️ MCP Bridge: Status {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ MCP Bridge: FAIL - {e}")
            self.failures.append(f"MCP Bridge: {e}")
            return False
    
    async def test_ollama(self, session: aiohttp.ClientSession) -> bool:
        """Test Ollama LLM service"""
        logger.info("Testing Ollama LLM...")
        try:
            async with session.get('http://localhost:11435/api/version') as resp:
                if resp.status == 200:
                    version = await resp.json()
                    logger.info(f"✅ Ollama: Connected - Version {version.get('version', 'unknown')}")
                    
                    # Test model listing
                    async with session.get('http://localhost:11435/api/tags') as tags_resp:
                        tags_data = await tags_resp.json()
                        models = tags_data.get('models', [])
                        logger.info(f"   Available models: {len(models)}")
                        for model in models:
                            logger.info(f"   - {model.get('name', 'unknown')}")
                        return True
                else:
                    logger.warning(f"⚠️ Ollama: Status {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ Ollama: FAIL - {e}")
            self.failures.append(f"Ollama: {e}")
            return False
    
    async def test_monitoring_stack(self, session: aiohttp.ClientSession) -> Dict[str, bool]:
        """Test monitoring stack components"""
        logger.info("Testing Monitoring Stack...")
        components = {
            'Prometheus': 'http://localhost:10300/-/healthy',
            'Grafana': 'http://localhost:10301/api/health',
            'Loki': 'http://localhost:10310/ready'
        }
        
        results = {}
        for name, url in components.items():
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        logger.info(f"✅ {name}: Operational")
                        results[name] = True
                    else:
                        logger.warning(f"⚠️ {name}: Status {resp.status}")
                        results[name] = False
            except Exception as e:
                logger.error(f"❌ {name}: FAIL - {e}")
                self.failures.append(f"Monitoring {name}: {e}")
                results[name] = False
        
        passed = sum(1 for v in results.values() if v)
        logger.info(f"Monitoring Stack: {passed}/{len(components)} operational")
        return results
    
    async def test_frontend(self, session: aiohttp.ClientSession) -> bool:
        """Test frontend accessibility"""
        logger.info("Testing Frontend...")
        try:
            async with session.get('http://localhost:11000/_stcore/health', timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    logger.info("✅ Frontend (Streamlit): Accessible")
                    return True
                else:
                    logger.warning(f"⚠️ Frontend: Status {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ Frontend: FAIL - {e}")
            self.failures.append(f"Frontend: {e}")
            return False
    
    async def run_all_tests(self) -> Dict:
        """Execute all validation tests"""
        logger.info("="*80)
        logger.info("STARTING COMPREHENSIVE SYSTEM VALIDATION")
        logger.info("="*80)
        
        async with aiohttp.ClientSession() as session:
            # Test backend
            backend_health = await self.test_backend_health(session)
            backend_services = await self.test_backend_services(session)
            
            # Test AI agents
            ai_agents = await self.test_all_ai_agents(session)
            
            # Test vector databases
            vector_dbs = await self.test_all_vector_databases(session)
            
            # Test MCP bridge
            mcp_bridge = await self.test_mcp_bridge(session)
            
            # Test Ollama
            ollama = await self.test_ollama(session)
            
            # Test monitoring stack
            monitoring = await self.test_monitoring_stack(session)
            
            # Test frontend
            frontend = await self.test_frontend(session)
        
        # Compile results
        results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'duration_seconds': (datetime.now(timezone.utc) - SCRIPT_START).total_seconds(),
            'backend': {
                'health': backend_health,
                'services': backend_services
            },
            'ai_agents': ai_agents,
            'vector_databases': vector_dbs,
            'mcp_bridge': mcp_bridge,
            'ollama': ollama,
            'monitoring': monitoring,
            'frontend': frontend,
            'failures': self.failures
        }
        
        return results
    
    def print_summary(self, results: Dict):
        """Print validation summary"""
        logger.info("="*80)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*80)
        
        # Count successes and failures
        total_tests = 0
        passed_tests = 0
        
        # Backend
        if results['backend']['health']:
            passed_tests += 1
        total_tests += 1
        
        backend_services = results['backend']['services']
        total_tests += len(backend_services)
        passed_tests += sum(1 for v in backend_services.values() if v)
        
        # AI Agents
        ai_agents = results['ai_agents']
        total_tests += len(ai_agents)
        passed_tests += sum(1 for v in ai_agents.values() if v)
        
        # Vector DBs
        vector_dbs = results['vector_databases']
        total_tests += len(vector_dbs)
        passed_tests += sum(1 for v in vector_dbs.values() if v)
        
        # MCP Bridge
        total_tests += 1
        if results['mcp_bridge']:
            passed_tests += 1
        
        # Ollama
        total_tests += 1
        if results['ollama']:
            passed_tests += 1
        
        # Monitoring
        monitoring = results['monitoring']
        total_tests += len(monitoring)
        passed_tests += sum(1 for v in monitoring.values() if v)
        
        # Frontend
        total_tests += 1
        if results['frontend']:
            passed_tests += 1
        
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Pass Rate: {pass_rate:.1f}%")
        logger.info(f"Duration: {results['duration_seconds']:.2f}s")
        
        if self.failures:
            logger.warning(f"\nFailures ({len(self.failures)}):")
            for i, failure in enumerate(self.failures, 1):
                logger.warning(f"  {i}. {failure}")
        
        logger.info("="*80)
        
        status = "✅ SYSTEM OPERATIONAL" if pass_rate >= 90 else "⚠️ SYSTEM DEGRADED" if pass_rate >= 75 else "❌ SYSTEM CRITICAL"
        logger.info(status)
        logger.info("="*80)
        
        return pass_rate >= 90


async def main():
    """Main execution function"""
    validator = SystemValidator()
    results = await validator.run_all_tests()
    
    # Save results to file
    output_file = f'/opt/sutazaiapp/system_validation_results_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {output_file}")
    
    # Print summary
    success = validator.print_summary(results)
    
    # Exit code based on success
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
