"""
Ultra System Coordination Service

Handles direct coordination with Ultra System Architect and other lead architects
in the 500-agent deployment. Manages intelligence sharing, decision synchronization,
and pattern discovery communication.
"""

import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import redis.asyncio as redis
import httpx

logger = logging.getLogger(__name__)

class UltraSystemCoordination:
    """Coordination service with Ultra System Architect and lead architects"""
    
    def __init__(self, config):
        self.config = config
        self.redis_client = None
        self.http_client = None
        self.ultra_system_url = config.get('ULTRA_SYSTEM_COORDINATOR', 'http://ultra-system-architect:11200')
        
        # Coordination channels
        self.coordination_channel = 'ultra:frontend:coordination'
        self.health_channel = 'ultra:frontend:health'
        self.patterns_channel = 'ultra:patterns:frontend'
        self.decisions_channel = 'ultra:decisions:frontend'
        self.intelligence_channel = 'ultra:intelligence:frontend'
        
        # Status tracking
        self.registration_status = 'not_registered'
        self.last_health_report = None
        self.coordination_health = 'unknown'
        
        # Background tasks
        self.background_tasks = set()
        self.is_running = False
    
    async def initialize(self):
        """Initialize coordination service"""
        try:
            logger.info("🤝 Initializing Ultra System Coordination...")
            
            # Initialize Redis client
            redis_url = self.config.get('REDIS_URL', 'redis://redis:6379')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Test Redis connection
            await self.redis_client.ping()
            logger.info("✅ Redis connection established")
            
            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(timeout=30.0)
            
            # Test Ultra System Architect connection
            try:
                response = await self.http_client.get(f"{self.ultra_system_url}/health")
                if response.status_code == 200:
                    logger.info("✅ Ultra System Architect connection verified")
                    self.coordination_health = 'healthy'
                else:
                    logger.warning(f"⚠️ Ultra System Architect unhealthy: {response.status_code}")
                    self.coordination_health = 'degraded'
            except Exception as e:
                logger.warning(f"⚠️ Could not connect to Ultra System Architect: {e}")
                self.coordination_health = 'disconnected'
            
            self.is_running = True
            logger.info("🎯 Ultra System Coordination initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Ultra System Coordination: {e}")
            raise
    
    async def register_with_system_architect(self) -> Dict[str, Any]:
        """Register Ultra Frontend UI Architect with Ultra System Architect"""
        logger.info("📝 Registering with Ultra System Architect...")
        
        registration_data = {
            'agent_type': 'ultra-frontend-ui-architect',
            'agent_id': 'ultra-frontend-ui-002',
            'port': 11201,
            'capabilities': ['ULTRAORGANIZE', 'ULTRAPROPERSTRUCTURE'],
            'status': 'initializing',
            'timestamp': datetime.utcnow().isoformat(),
            'lead_architect_rank': 2,
            'specialization': 'frontend_architecture',
            'coordination_channels': [
                self.coordination_channel,
                self.health_channel,
                self.patterns_channel,
                self.decisions_channel,
                self.intelligence_channel
            ],
            'endpoints': {
                'health': '/health',
                'optimize': '/optimize/frontend',
                'metrics': '/metrics',
                'coordinate': '/coordinate/status',
                'patterns': '/patterns'
            }
        }
        
        try:
            # Register via Redis pubsub
            await self.redis_client.publish('ultra:system:registrations', json.dumps(registration_data))
            
            # Also register via HTTP if possible
            if self.coordination_health in ['healthy', 'degraded']:
                try:
                    response = await self.http_client.post(
                        f"{self.ultra_system_url}/register/lead-architect",
                        json=registration_data
                    )
                    if response.status_code == 200:
                        logger.info("✅ HTTP registration successful")
                    else:
                        logger.warning(f"⚠️ HTTP registration failed: {response.status_code}")
                except Exception as e:
                    logger.warning(f"⚠️ HTTP registration error: {e}")
            
            # Wait for acknowledgment (with timeout)
            ack_result = await self._wait_for_registration_ack(timeout=30)
            
            if ack_result.get('status') == 'success':
                self.registration_status = 'registered'
                logger.info("✅ Successfully registered with Ultra System Architect")
                
                # Start background coordination tasks
                await self._start_background_tasks()
                
                return {'status': 'success', 'details': ack_result}
            else:
                self.registration_status = 'failed'
                logger.error("❌ Registration failed or timed out")
                return {'status': 'failed', 'details': ack_result}
                
        except Exception as e:
            logger.error(f"❌ Registration failed: {e}")
            self.registration_status = 'error'
            return {'status': 'error', 'details': str(e)}
    
    async def _wait_for_registration_ack(self, timeout: int = 30) -> Dict[str, Any]:
        """Wait for registration acknowledgment from Ultra System Architect"""
        try:
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe('ultra:frontend:registration_ack')
            
            end_time = datetime.now() + timedelta(seconds=timeout)
            
            async for message in pubsub.listen():
                if datetime.now() > end_time:
                    await pubsub.unsubscribe('ultra:frontend:registration_ack')
                    return {'status': 'timeout', 'message': 'Registration acknowledgment timeout'}
                
                if message['type'] == 'message':
                    try:
                        ack_data = json.loads(message['data'])
                        if ack_data.get('agent_type') == 'ultra-frontend-ui-architect':
                            await pubsub.unsubscribe('ultra:frontend:registration_ack')
                            return {'status': 'success', 'ack_data': ack_data}
                    except json.JSONDecodeError:
                        continue
            
            return {'status': 'timeout', 'message': 'No acknowledgment received'}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    async def report_ultra_status(self, status_data: Dict[str, Any]):
        """Report comprehensive status to Ultra System Architect"""
        try:
            enhanced_status = {
                **status_data,
                'agent_id': 'ultra-frontend-ui-002',
                'agent_type': 'ultra-frontend-ui-architect',
                'port': 11201,
                'coordination_health': self.coordination_health,
                'registration_status': self.registration_status,
                'capabilities': ['ULTRAORGANIZE', 'ULTRAPROPERSTRUCTURE'],
                'timestamp': datetime.utcnow().isoformat(),
                'uptime': self._calculate_uptime(),
                'last_optimization': status_data.get('last_optimization_timestamp'),
                'performance_metrics': status_data.get('metrics', {}),
                'coordination_metrics': {
                    'messages_sent': getattr(self, 'messages_sent', 0),
                    'messages_received': getattr(self, 'messages_received', 0),
                    'patterns_shared': getattr(self, 'patterns_shared', 0),
                    'decisions_processed': getattr(self, 'decisions_processed', 0)
                }
            }
            
            # Send via Redis
            await self.redis_client.publish(self.health_channel, json.dumps(enhanced_status))
            
            # Also send via HTTP if possible
            if self.coordination_health == 'healthy':
                try:
                    await self.http_client.post(
                        f"{self.ultra_system_url}/status/lead-architect",
                        json=enhanced_status,
                        timeout=5.0
                    )
                except Exception as e:
                    logger.debug(f"HTTP status report failed: {e}")
            
            self.last_health_report = datetime.utcnow()
            logger.debug("📊 Status reported to Ultra System Architect")
            
        except Exception as e:
            logger.error(f"❌ Failed to report status: {e}")
    
    async def share_frontend_intelligence(self, intelligence: Dict[str, Any]):
        """Share frontend pattern discoveries with system intelligence"""
        try:
            intelligence_package = {
                'source': 'ultra-frontend-ui-architect',
                'agent_id': 'ultra-frontend-ui-002',
                'type': 'pattern_discovery',
                'intelligence': intelligence,
                'confidence_score': self._calculate_intelligence_confidence(intelligence),
                'timestamp': datetime.utcnow().isoformat(),
                'metadata': {
                    'discovery_context': intelligence.get('context', 'optimization'),
                    'impact_level': intelligence.get('impact', 'medium'),
                    'actionable': intelligence.get('actionable', True)
                }
            }
            
            # Send to patterns channel
            await self.redis_client.publish(self.patterns_channel, json.dumps(intelligence_package))
            
            # Send to general intelligence channel
            await self.redis_client.publish(self.intelligence_channel, json.dumps(intelligence_package))
            
            # Track intelligence sharing
            self.patterns_shared = getattr(self, 'patterns_shared', 0) + 1
            
            logger.info(f"🧠 Shared frontend intelligence: {intelligence.get('type', 'unknown')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to share intelligence: {e}")
    
    async def receive_system_decisions(self) -> List[Dict[str, Any]]:
        """Receive architectural decisions from Ultra System Architect"""
        decisions = []
        
        try:
            # Check for new decisions in Redis
            decision_keys = await self.redis_client.keys('ultra:decisions:frontend:*')
            
            for key in decision_keys:
                decision_data = await self.redis_client.get(key)
                if decision_data:
                    try:
                        decision = json.loads(decision_data)
                        decisions.append(decision)
                        
                        # Mark as processed
                        await self.redis_client.delete(key)
                        
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid decision data in {key}")
            
            if decisions:
                logger.info(f"📋 Received {len(decisions)} system decisions")
            
        except Exception as e:
            logger.error(f"❌ Failed to receive system decisions: {e}")
        
        return decisions
    
    async def process_pending_decisions(self):
        """Process any pending decisions from the system"""
        try:
            decisions = await self.receive_system_decisions()
            
            for decision in decisions:
                await self._process_system_decision(decision)
                self.decisions_processed = getattr(self, 'decisions_processed', 0) + 1
                
        except Exception as e:
            logger.error(f"❌ Failed to process pending decisions: {e}")
    
    async def _process_system_decision(self, decision: Dict[str, Any]):
        """Process a specific system decision"""
        decision_type = decision.get('type')
        
        try:
            if decision_type == 'optimization_directive':
                logger.info(f"🎯 Processing optimization directive: {decision.get('target')}")
                # Implementation would trigger optimization based on directive
                
            elif decision_type == 'pattern_adoption':
                logger.info(f"📋 Processing pattern adoption: {decision.get('pattern')}")
                # Implementation would adopt new patterns
                
            elif decision_type == 'compliance_enforcement':
                logger.info(f"🏗️ Processing compliance enforcement: {decision.get('rule')}")
                # Implementation would enforce new compliance rules
                
            elif decision_type == 'coordination_update':
                logger.info(f"🤝 Processing coordination update: {decision.get('update')}")
                # Implementation would update coordination protocols
                
            else:
                logger.warning(f"⚠️ Unknown decision type: {decision_type}")
                
        except Exception as e:
            logger.error(f"❌ Failed to process decision {decision_type}: {e}")
    
    async def validate_coordination(self) -> Dict[str, Any]:
        """Validate coordination health with Ultra System Architect"""
        validation = {
            'healthy': False,
            'redis_connection': False,
            'ultra_system_connection': False,
            'registration_status': self.registration_status,
            'last_health_report': self.last_health_report.isoformat() if self.last_health_report else None,
            'coordination_health': self.coordination_health
        }
        
        try:
            # Test Redis connection
            await self.redis_client.ping()
            validation['redis_connection'] = True
            
            # Test Ultra System Architect connection
            if self.coordination_health == 'healthy':
                response = await self.http_client.get(f"{self.ultra_system_url}/health", timeout=5.0)
                if response.status_code == 200:
                    validation['ultra_system_connection'] = True
            
            # Overall health
            validation['healthy'] = (
                validation['redis_connection'] and
                self.registration_status == 'registered' and
                self.coordination_health in ['healthy', 'degraded']
            )
            
        except Exception as e:
            logger.error(f"❌ Coordination validation failed: {e}")
            validation['error'] = str(e)
        
        return validation
    
    async def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination status"""
        return {
            'registration_status': self.registration_status,
            'coordination_health': self.coordination_health,
            'last_health_report': self.last_health_report.isoformat() if self.last_health_report else None,
            'ultra_system_url': self.ultra_system_url,
            'is_running': self.is_running,
            'background_tasks_active': len(self.background_tasks),
            'coordination_metrics': {
                'messages_sent': getattr(self, 'messages_sent', 0),
                'messages_received': getattr(self, 'messages_received', 0),
                'patterns_shared': getattr(self, 'patterns_shared', 0),
                'decisions_processed': getattr(self, 'decisions_processed', 0)
            }
        }
    
    async def check_coordination_health(self) -> Dict[str, Any]:
        """Check coordination health"""
        health = {
            'status': 'unknown',
            'details': {}
        }
        
        try:
            validation = await self.validate_coordination()
            
            if validation['healthy']:
                health['status'] = 'healthy'
            elif validation['redis_connection']:
                health['status'] = 'degraded'
            else:
                health['status'] = 'unhealthy'
            
            health['details'] = validation
            
        except Exception as e:
            health['status'] = 'error'
            health['details'] = {'error': str(e)}
        
        return health
    
    async def report_optimization_results(self, results: Dict[str, Any]):
        """Report optimization results to Ultra System Architect"""
        try:
            report = {
                'source': 'ultra-frontend-ui-architect',
                'agent_id': 'ultra-frontend-ui-002',
                'type': 'optimization_results',
                'results': results,
                'timestamp': datetime.utcnow().isoformat(),
                'metrics': {
                    'organization_score': results.get('organization_results', {}).get('organization_score', 0),
                    'compliance_score': results.get('compliance_results', {}).get('compliance_score', 0),
                    'overall_score': results.get('overall_score', 0),
                    'files_optimized': results.get('organization_results', {}).get('files_moved', 0),
                    'violations_fixed': results.get('compliance_results', {}).get('violations_fixed', 0)
                }
            }
            
            # Send via Redis
            await self.redis_client.publish('ultra:system:optimization_results', json.dumps(report))
            
            # Send via HTTP if possible
            if self.coordination_health == 'healthy':
                try:
                    await self.http_client.post(
                        f"{self.ultra_system_url}/results/optimization",
                        json=report,
                        timeout=10.0
                    )
                except Exception as e:
                    logger.debug(f"HTTP optimization report failed: {e}")
            
            logger.info("📊 Optimization results reported to Ultra System Architect")
            
        except Exception as e:
            logger.error(f"❌ Failed to report optimization results: {e}")
    
    async def unregister(self):
        """Unregister from Ultra System Architect"""
        try:
            logger.info("📤 Unregistering from Ultra System Architect...")
            
            unregister_data = {
                'agent_type': 'ultra-frontend-ui-architect',
                'agent_id': 'ultra-frontend-ui-002',
                'timestamp': datetime.utcnow().isoformat(),
                'reason': 'shutdown'
            }
            
            # Unregister via Redis
            await self.redis_client.publish('ultra:system:unregistrations', json.dumps(unregister_data))
            
            # Stop background tasks
            await self._stop_background_tasks()
            
            # Close connections
            if self.http_client:
                await self.http_client.aclose()
            
            if self.redis_client:
                await self.redis_client.close()
            
            self.registration_status = 'unregistered'
            self.is_running = False
            
            logger.info("✅ Successfully unregistered")
            
        except Exception as e:
            logger.error(f"❌ Failed to unregister: {e}")
    
    # Helper methods
    
    def _calculate_intelligence_confidence(self, intelligence: Dict[str, Any]) -> float:
        """Calculate confidence score for intelligence data"""
        # Simple heuristic - in practice this would be more sophisticated
        base_confidence = 0.8
        
        # Adjust based on data quality
        if intelligence.get('data_quality') == 'high':
            base_confidence += 0.15
        elif intelligence.get('data_quality') == 'low':
            base_confidence -= 0.2
        
        # Adjust based on sample size
        sample_size = intelligence.get('sample_size', 0)
        if sample_size > 100:
            base_confidence += 0.05
        elif sample_size < 10:
            base_confidence -= 0.1
        
        return max(0.0, min(1.0, base_confidence))
    
    def _calculate_uptime(self) -> str:
        """Calculate uptime since initialization"""
        # Implementation would track start time
        return "unknown"
    
    async def _start_background_tasks(self):
        """Start background coordination tasks"""
        try:
            # Start periodic health reporting
            health_task = asyncio.create_task(self._periodic_health_reporting())
            self.background_tasks.add(health_task)
            health_task.add_done_callback(self.background_tasks.discard)
            
            # Start decision processing
            decision_task = asyncio.create_task(self._periodic_decision_processing())
            self.background_tasks.add(decision_task)
            decision_task.add_done_callback(self.background_tasks.discard)
            
            logger.info("🔄 Background coordination tasks started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start background tasks: {e}")
    
    async def _stop_background_tasks(self):
        """Stop background coordination tasks"""
        try:
            for task in self.background_tasks:
                task.cancel()
            
            # Wait for tasks to complete cancellation
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            self.background_tasks.clear()
            logger.info("🛑 Background coordination tasks stopped")
            
        except Exception as e:
            logger.error(f"❌ Failed to stop background tasks: {e}")
    
    async def _periodic_health_reporting(self):
        """Periodic health reporting task"""
        while self.is_running:
            try:
                # Report health every 30 seconds
                await asyncio.sleep(30)
                
                if self.registration_status == 'registered':
                    status_data = {
                        'periodic_health_check': True,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    await self.report_ultra_status(status_data)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Periodic health reporting error: {e}")
                await asyncio.sleep(60)  # Longer sleep on error
    
    async def _periodic_decision_processing(self):
        """Periodic decision processing task"""
        while self.is_running:
            try:
                # Check for decisions every 10 seconds
                await asyncio.sleep(10)
                
                if self.registration_status == 'registered':
                    await self.process_pending_decisions()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Periodic decision processing error: {e}")
                await asyncio.sleep(30)  # Longer sleep on error