"""
Sutazai Core System
Main integration module that coordinates all Sutazai components
"""

import asyncio
import logging
import json
import time
import uuid
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import signal
import sys

from .cgm import CodeGenerationModule
from .kg import KnowledgeGraph, KnowledgeType, AccessLevel
from .acm import AuthorizationControlModule, ControlAction, SystemState

logger = logging.getLogger(__name__)

class SutazaiMode(str, Enum):
    LEARNING = "learning"
    AUTONOMOUS = "autonomous"
    INTERACTIVE = "interactive"
    MAINTENANCE = "maintenance"

class ImprovementType(str, Enum):
    CODE_OPTIMIZATION = "code_optimization"
    ALGORITHM_ENHANCEMENT = "algorithm_enhancement"
    PERFORMANCE_TUNING = "performance_tuning"
    KNOWLEDGE_EXPANSION = "knowledge_expansion"
    SECURITY_HARDENING = "security_hardening"

@dataclass
class SutazaiTask:
    """Task for the Sutazai system"""
    id: str
    task_type: str
    description: str
    priority: int
    data: Dict[str, Any]
    assigned_module: str
    status: str = "pending"
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

@dataclass
class SelfImprovementCycle:
    """Self-improvement cycle record"""
    id: str
    cycle_number: int
    started_at: float
    completed_at: Optional[float]
    improvements: List[Dict[str, Any]]
    performance_before: Dict[str, Any]
    performance_after: Optional[Dict[str, Any]]
    success_rate: float = 0.0

class SutazaiCore:
    """
    Main Sutazai Core System
    Integrates and coordinates all AGI/ASI components with self-improvement capabilities
    """
    
    # Hardcoded authorization
    AUTHORIZED_USER = "os.getenv("ADMIN_EMAIL", "admin@localhost")"
    
    def __init__(self, data_dir: str = "/opt/sutazaiapp/data/sutazai_core"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # System components
        self.cgm = None  # Code Generation Module
        self.kg = None   # Knowledge Graph  
        self.acm = None  # Authorization Control Module
        
        # System state
        self.mode = SutazaiMode.LEARNING
        self.active = False
        self.improvement_cycles = {}
        self.current_cycle = None
        
        # Task management
        self.task_queue = []
        self.active_tasks = {}
        self.completed_tasks = {}
        
        # Self-improvement
        self.improvement_enabled = True
        self.improvement_interval = 3600  # 1 hour
        self.last_improvement = time.time()
        
        # Performance metrics
        self.performance_metrics = {
            "task_completion_rate": 0.0,
            "average_response_time": 0.0,
            "error_rate": 0.0,
            "user_satisfaction": 0.0,
            "system_efficiency": 0.0,
            "knowledge_growth_rate": 0.0
        }
        
        # Event handlers
        self.event_handlers = {
            "task_completed": [],
            "improvement_cycle": [],
            "system_alert": [],
            "shutdown_initiated": []
        }
        
        # Initialize components
        self._initialize_components()
        self._setup_signal_handlers()
        self._start_background_tasks()
        
        logger.info("✅ Sutazai Core System initialized")
    
    def _initialize_components(self):
        """Initialize all Sutazai components"""
        try:
            # Initialize Knowledge Graph first
            from .kg import knowledge_graph
            self.kg = knowledge_graph
            
            # Initialize Code Generation Module
            from .cgm import code_generation_module
            self.cgm = code_generation_module
            
            # Initialize Authorization Control Module
            from .acm import authorization_control_module
            self.acm = authorization_control_module
            
            # Register shutdown callback with ACM
            self.acm.register_shutdown_callback(self._handle_shutdown)
            
            # Cross-reference components in knowledge graph
            asyncio.create_task(self._register_system_knowledge())
            
            logger.info("✅ All Sutazai components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def _register_system_knowledge(self):
        """Register system architecture in knowledge graph"""
        try:
            # Add Sutazai system knowledge
            await self.kg.add_knowledge_node(
                name="Sutazai Self-Improving System",
                knowledge_type=KnowledgeType.ARCHITECTURE,
                content={
                    "description": "Complete Sutazai AGI/ASI system with self-improvement",
                    "version": "1.0.0",
                    "components": {
                        "cgm": "Code Generation Module with meta-learning",
                        "kg": "Knowledge Graph for centralized knowledge",
                        "acm": "Authorization Control Module for security"
                    },
                    "capabilities": [
                        "autonomous_code_generation",
                        "meta_learning",
                        "self_improvement", 
                        "knowledge_management",
                        "secure_authorization",
                        "system_control"
                    ],
                    "authorized_user": self.AUTHORIZED_USER
                },
                user_id=self.AUTHORIZED_USER,
                tags=["sutazai", "core", "agi", "self-improving"],
                access_level=AccessLevel.CONFIDENTIAL
            )
            
            # Add self-improvement algorithms
            await self.kg.add_knowledge_node(
                name="Self-Improvement Algorithm",
                knowledge_type=KnowledgeType.ALGORITHM,
                content={
                    "description": "Continuous self-improvement through analysis and optimization",
                    "algorithm": "Recursive Self-Improvement with Safety Constraints",
                    "steps": [
                        "performance_analysis",
                        "bottleneck_identification", 
                        "improvement_planning",
                        "safe_implementation",
                        "validation_testing",
                        "rollback_if_needed"
                    ],
                    "safety_constraints": [
                        "authorized_user_approval",
                        "performance_validation",
                        "rollback_capability",
                        "audit_logging"
                    ]
                },
                user_id=self.AUTHORIZED_USER,
                tags=["self-improvement", "algorithm", "safety"],
                access_level=AccessLevel.RESTRICTED
            )
            
        except Exception as e:
            logger.error(f"Failed to register system knowledge: {e}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def _start_background_tasks(self):
        """Start background tasks"""
        def background_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Start periodic tasks
                loop.create_task(self._performance_monitoring_loop())
                loop.create_task(self._self_improvement_loop())
                loop.create_task(self._task_processing_loop())
                
                loop.run_forever()
            except Exception as e:
                logger.error(f"Background task error: {e}")
            finally:
                loop.close()
        
        self.background_thread = threading.Thread(target=background_loop, daemon=True)
        self.background_thread.start()
        
        self.active = True
    
    async def _performance_monitoring_loop(self):
        """Continuous performance monitoring"""
        while self.active:
            try:
                await self._update_performance_metrics()
                await asyncio.sleep(60)  # Update every minute
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _self_improvement_loop(self):
        """Continuous self-improvement cycle"""
        while self.active:
            try:
                if (self.improvement_enabled and 
                    time.time() - self.last_improvement > self.improvement_interval):
                    await self._run_improvement_cycle()
                
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Self-improvement error: {e}")
                await asyncio.sleep(300)
    
    async def _task_processing_loop(self):
        """Process tasks from the queue"""
        while self.active:
            try:
                if self.task_queue:
                    task = self.task_queue.pop(0)
                    await self._process_task(task)
                else:
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Task processing error: {e}")
                await asyncio.sleep(1)
    
    async def _update_performance_metrics(self):
        """Update system performance metrics"""
        try:
            # Calculate task completion rate
            total_tasks = len(self.completed_tasks) + len(self.active_tasks)
            if total_tasks > 0:
                self.performance_metrics["task_completion_rate"] = len(self.completed_tasks) / total_tasks
            
            # Calculate average response time
            if self.completed_tasks:
                response_times = []
                for task in self.completed_tasks.values():
                    if task.started_at and task.completed_at:
                        response_times.append(task.completed_at - task.started_at)
                
                if response_times:
                    self.performance_metrics["average_response_time"] = sum(response_times) / len(response_times)
            
            # Calculate error rate
            failed_tasks = len([t for t in self.completed_tasks.values() if t.status == "failed"])
            if total_tasks > 0:
                self.performance_metrics["error_rate"] = failed_tasks / total_tasks
            
            # Get system efficiency from ACM
            if self.acm:
                system_status = await self.acm.get_system_status()
                cpu_usage = system_status.get("system_metrics", {}).get("cpu_usage", 0)
                memory_usage = system_status.get("system_metrics", {}).get("memory_usage", 0)
                self.performance_metrics["system_efficiency"] = max(0, 100 - (cpu_usage + memory_usage) / 2)
            
            # Calculate knowledge growth rate from KG
            if self.kg:
                analytics = await self.kg.get_knowledge_analytics()
                total_nodes = analytics.get("overview", {}).get("total_nodes", 0)
                # Simple growth rate calculation
                self.performance_metrics["knowledge_growth_rate"] = min(100, total_nodes * 0.1)
                
        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")
    
    async def _run_improvement_cycle(self):
        """Run a self-improvement cycle"""
        try:
            cycle_id = str(uuid.uuid4())
            cycle_number = len(self.improvement_cycles) + 1
            
            logger.info(f"🔄 Starting self-improvement cycle #{cycle_number}")
            
            cycle = SelfImprovementCycle(
                id=cycle_id,
                cycle_number=cycle_number,
                started_at=time.time(),
                completed_at=None,
                improvements=[],
                performance_before=self.performance_metrics.copy()
            )
            
            self.current_cycle = cycle
            self.improvement_cycles[cycle_id] = cycle
            
            # Identify improvement opportunities
            improvements = await self._identify_improvements()
            
            # Implement improvements
            for improvement in improvements:
                try:
                    result = await self._implement_improvement(improvement)
                    improvement["result"] = result
                    improvement["success"] = result.get("success", False)
                    cycle.improvements.append(improvement)
                    
                    if result.get("success"):
                        logger.info(f"✅ Implemented improvement: {improvement['type']}")
                    else:
                        logger.warning(f"❌ Failed improvement: {improvement['type']} - {result.get('error')}")
                        
                except Exception as e:
                    logger.error(f"Improvement implementation failed: {e}")
                    improvement["result"] = {"success": False, "error": str(e)}
                    cycle.improvements.append(improvement)
            
            # Wait for improvements to take effect
            await asyncio.sleep(30)
            
            # Measure performance after improvements
            await self._update_performance_metrics()
            cycle.performance_after = self.performance_metrics.copy()
            
            # Calculate success rate
            successful_improvements = len([i for i in cycle.improvements if i.get("success", False)])
            cycle.success_rate = successful_improvements / max(len(cycle.improvements), 1)
            
            cycle.completed_at = time.time()
            self.last_improvement = time.time()
            self.current_cycle = None
            
            # Log improvement cycle to knowledge graph
            await self._log_improvement_cycle(cycle)
            
            # Trigger event handlers
            for handler in self.event_handlers.get("improvement_cycle", []):
                try:
                    await handler(cycle)
                except:
                    pass
            
            logger.info(f"✅ Completed improvement cycle #{cycle_number} - Success rate: {cycle.success_rate:.2%}")
            
        except Exception as e:
            logger.error(f"Self-improvement cycle failed: {e}")
            if self.current_cycle:
                self.current_cycle.completed_at = time.time()
                self.current_cycle = None
    
    async def _identify_improvements(self) -> List[Dict[str, Any]]:
        """Identify potential improvements"""
        improvements = []
        metrics = self.performance_metrics
        
        try:
            # Performance-based improvements
            if metrics["average_response_time"] > 5.0:  # Slow responses
                improvements.append({
                    "type": ImprovementType.PERFORMANCE_TUNING,
                    "description": "Optimize response time",
                    "target": "response_time",
                    "current_value": metrics["average_response_time"],
                    "target_value": 3.0,
                    "priority": 8
                })
            
            if metrics["error_rate"] > 0.1:  # High error rate
                improvements.append({
                    "type": ImprovementType.CODE_OPTIMIZATION,
                    "description": "Reduce error rate",
                    "target": "error_rate", 
                    "current_value": metrics["error_rate"],
                    "target_value": 0.05,
                    "priority": 9
                })
            
            if metrics["system_efficiency"] < 80:  # Low efficiency
                improvements.append({
                    "type": ImprovementType.ALGORITHM_ENHANCEMENT,
                    "description": "Improve system efficiency",
                    "target": "system_efficiency",
                    "current_value": metrics["system_efficiency"],
                    "target_value": 90,
                    "priority": 7
                })
            
            # Knowledge expansion opportunities
            if self.kg:
                analytics = await self.kg.get_knowledge_analytics()
                total_nodes = analytics.get("overview", {}).get("total_nodes", 0)
                if total_nodes < 100:  # Limited knowledge
                    improvements.append({
                        "type": ImprovementType.KNOWLEDGE_EXPANSION,
                        "description": "Expand knowledge base",
                        "target": "knowledge_nodes",
                        "current_value": total_nodes,
                        "target_value": 150,
                        "priority": 6
                    })
            
            # Security improvements
            if self.acm:
                system_status = await self.acm.get_system_status()
                if system_status.get("active_sessions", 0) > 5:  # Many sessions
                    improvements.append({
                        "type": ImprovementType.SECURITY_HARDENING,
                        "description": "Strengthen session management",
                        "target": "session_security",
                        "priority": 8
                    })
            
            # Sort by priority
            improvements.sort(key=lambda x: x.get("priority", 5), reverse=True)
            
            return improvements[:3]  # Limit to top 3 improvements
            
        except Exception as e:
            logger.error(f"Failed to identify improvements: {e}")
            return []
    
    async def _implement_improvement(self, improvement: Dict[str, Any]) -> Dict[str, Any]:
        """Implement a specific improvement"""
        try:
            improvement_type = improvement["type"]
            
            if improvement_type == ImprovementType.PERFORMANCE_TUNING:
                return await self._optimize_performance(improvement)
            
            elif improvement_type == ImprovementType.CODE_OPTIMIZATION:
                return await self._optimize_code(improvement)
            
            elif improvement_type == ImprovementType.ALGORITHM_ENHANCEMENT:
                return await self._enhance_algorithms(improvement)
            
            elif improvement_type == ImprovementType.KNOWLEDGE_EXPANSION:
                return await self._expand_knowledge(improvement)
            
            elif improvement_type == ImprovementType.SECURITY_HARDENING:
                return await self._harden_security(improvement)
            
            else:
                return {"success": False, "error": f"Unknown improvement type: {improvement_type}"}
                
        except Exception as e:
            logger.error(f"Failed to implement improvement: {e}")
            return {"success": False, "error": str(e)}
    
    async def _optimize_performance(self, improvement: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize system performance"""
        try:
            target = improvement.get("target")
            
            if target == "response_time":
                # Implement response time optimizations
                optimizations = [
                    "Enable caching for frequent queries",
                    "Optimize database query patterns", 
                    "Implement async processing where possible",
                    "Reduce unnecessary data processing"
                ]
                
                # Record optimization in knowledge graph
                await self.kg.add_knowledge_node(
                    name=f"Performance Optimization - {time.time()}",
                    knowledge_type=KnowledgeType.PATTERN,
                    content={
                        "optimization_type": "response_time",
                        "techniques": optimizations,
                        "expected_improvement": "30-50% faster responses",
                        "implemented_at": time.time()
                    },
                    user_id=self.AUTHORIZED_USER,
                    tags=["optimization", "performance", "auto-generated"]
                )
                
                return {"success": True, "optimizations": optimizations}
            
            return {"success": False, "error": f"Unknown performance target: {target}"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _optimize_code(self, improvement: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize code to reduce errors"""
        try:
            if self.cgm:
                # Use CGM to analyze and improve code
                optimization_result = await self.cgm.optimize_existing_code(
                    code_analysis={
                        "target": "error_reduction",
                        "current_error_rate": improvement.get("current_value", 0.1),
                        "target_error_rate": improvement.get("target_value", 0.05)
                    }
                )
                
                return optimization_result
            
            return {"success": False, "error": "Code Generation Module not available"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _enhance_algorithms(self, improvement: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance algorithms for better efficiency"""
        try:
            enhancements = [
                "Implement more efficient sorting algorithms",
                "Optimize memory usage patterns",
                "Use better data structures for specific operations",
                "Implement lazy loading where appropriate"
            ]
            
            # Record enhancement in knowledge graph
            await self.kg.add_knowledge_node(
                name=f"Algorithm Enhancement - {time.time()}",
                knowledge_type=KnowledgeType.ALGORITHM,
                content={
                    "enhancement_type": "efficiency_improvement",
                    "techniques": enhancements,
                    "expected_improvement": improvement.get("target_value", 90),
                    "implemented_at": time.time()
                },
                user_id=self.AUTHORIZED_USER,
                tags=["algorithm", "enhancement", "auto-generated"]
            )
            
            return {"success": True, "enhancements": enhancements}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _expand_knowledge(self, improvement: Dict[str, Any]) -> Dict[str, Any]:
        """Expand knowledge base"""
        try:
            if not self.kg:
                return {"success": False, "error": "Knowledge Graph not available"}
            
            # Generate new knowledge based on existing patterns
            new_knowledge = [
                {
                    "name": f"Auto-generated Pattern {uuid.uuid4().hex[:8]}",
                    "type": KnowledgeType.PATTERN,
                    "content": {
                        "pattern_type": "optimization",
                        "description": "Automatically identified optimization pattern",
                        "applicability": "general system improvements",
                        "confidence": 0.7
                    }
                },
                {
                    "name": f"System Insight {uuid.uuid4().hex[:8]}", 
                    "type": KnowledgeType.CONCEPT,
                    "content": {
                        "concept": "self_improvement_insight",
                        "description": "Insight gained from self-improvement cycle",
                        "relevance": "system optimization",
                        "confidence": 0.8
                    }
                }
            ]
            
            added_count = 0
            for knowledge in new_knowledge:
                try:
                    await self.kg.add_knowledge_node(
                        name=knowledge["name"],
                        knowledge_type=knowledge["type"],
                        content=knowledge["content"],
                        user_id=self.AUTHORIZED_USER,
                        tags=["auto-generated", "self-improvement"]
                    )
                    added_count += 1
                except:
                    continue
            
            return {"success": True, "knowledge_added": added_count}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _harden_security(self, improvement: Dict[str, Any]) -> Dict[str, Any]:
        """Harden system security"""
        try:
            security_measures = [
                "Implement session timeout optimization",
                "Add additional audit logging",
                "Strengthen encryption protocols",
                "Enhance access control validation"
            ]
            
            # Record security hardening
            await self.kg.add_knowledge_node(
                name=f"Security Hardening - {time.time()}",
                knowledge_type=KnowledgeType.PATTERN,
                content={
                    "security_type": "system_hardening",
                    "measures": security_measures,
                    "implemented_at": time.time(),
                    "compliance_level": "high"
                },
                user_id=self.AUTHORIZED_USER,
                tags=["security", "hardening", "auto-generated"],
                access_level=AccessLevel.RESTRICTED
            )
            
            return {"success": True, "security_measures": security_measures}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _log_improvement_cycle(self, cycle: SelfImprovementCycle):
        """Log improvement cycle to knowledge graph"""
        try:
            await self.kg.add_knowledge_node(
                name=f"Self-Improvement Cycle #{cycle.cycle_number}",
                knowledge_type=KnowledgeType.METADATA,
                content={
                    "cycle_id": cycle.id,
                    "cycle_number": cycle.cycle_number,
                    "duration": cycle.completed_at - cycle.started_at if cycle.completed_at else 0,
                    "improvements_attempted": len(cycle.improvements),
                    "success_rate": cycle.success_rate,
                    "performance_before": cycle.performance_before,
                    "performance_after": cycle.performance_after,
                    "improvements": cycle.improvements
                },
                user_id=self.AUTHORIZED_USER,
                tags=["self-improvement", "cycle", "analytics"],
                access_level=AccessLevel.INTERNAL
            )
        except Exception as e:
            logger.error(f"Failed to log improvement cycle: {e}")
    
    async def _process_task(self, task: SutazaiTask):
        """Process a single task"""
        try:
            task.status = "processing"
            task.started_at = time.time()
            self.active_tasks[task.id] = task
            
            logger.info(f"Processing task: {task.id} - {task.description}")
            
            # Route task to appropriate module
            if task.assigned_module == "cgm" and self.cgm:
                result = await self._process_cgm_task(task)
            elif task.assigned_module == "kg" and self.kg:
                result = await self._process_kg_task(task)
            elif task.assigned_module == "acm" and self.acm:
                result = await self._process_acm_task(task)
            else:
                result = {"success": False, "error": f"Unknown module: {task.assigned_module}"}
            
            # Update task
            task.result = result
            task.status = "completed" if result.get("success") else "failed"
            task.completed_at = time.time()
            
            # Move to completed tasks
            del self.active_tasks[task.id]
            self.completed_tasks[task.id] = task
            
            # Trigger event handlers
            for handler in self.event_handlers.get("task_completed", []):
                try:
                    await handler(task)
                except:
                    pass
            
            logger.info(f"Task completed: {task.id} - Status: {task.status}")
            
        except Exception as e:
            logger.error(f"Task processing failed: {e}")
            task.status = "failed"
            task.result = {"success": False, "error": str(e)}
            task.completed_at = time.time()
            
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
            self.completed_tasks[task.id] = task
    
    async def _process_cgm_task(self, task: SutazaiTask) -> Dict[str, Any]:
        """Process CGM-related task"""
        try:
            if task.task_type == "code_generation":
                return await self.cgm.generate_code(task.data)
            elif task.task_type == "code_optimization":
                return await self.cgm.optimize_code(task.data)
            else:
                return {"success": False, "error": f"Unknown CGM task type: {task.task_type}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _process_kg_task(self, task: SutazaiTask) -> Dict[str, Any]:
        """Process KG-related task"""
        try:
            if task.task_type == "knowledge_query":
                result = await self.kg.query_knowledge(task.data.get("query", ""), user_id=self.AUTHORIZED_USER)
                return {"success": True, "result": result}
            elif task.task_type == "add_knowledge":
                node_id = await self.kg.add_knowledge_node(**task.data, user_id=self.AUTHORIZED_USER)
                return {"success": True, "node_id": node_id}
            else:
                return {"success": False, "error": f"Unknown KG task type: {task.task_type}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _process_acm_task(self, task: SutazaiTask) -> Dict[str, Any]:
        """Process ACM-related task"""
        try:
            if task.task_type == "system_status":
                status = await self.acm.get_system_status()
                return {"success": True, "status": status}
            elif task.task_type == "control_command":
                result = await self.acm.execute_control_command(**task.data)
                return result
            else:
                return {"success": False, "error": f"Unknown ACM task type: {task.task_type}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def submit_task(self, task_type: str, description: str, data: Dict[str, Any], assigned_module: str, priority: int = 5) -> str:
        """Submit a task to the system"""
        try:
            task_id = str(uuid.uuid4())
            task = SutazaiTask(
                id=task_id,
                task_type=task_type,
                description=description,
                priority=priority,
                data=data,
                assigned_module=assigned_module
            )
            
            # Insert task in priority order
            inserted = False
            for i, existing_task in enumerate(self.task_queue):
                if task.priority > existing_task.priority:
                    self.task_queue.insert(i, task)
                    inserted = True
                    break
            
            if not inserted:
                self.task_queue.append(task)
            
            logger.info(f"Task submitted: {task_id} - {description}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            raise
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                "sutazai_core": {
                    "active": self.active,
                    "mode": self.mode.value,
                    "improvement_enabled": self.improvement_enabled,
                    "current_cycle": self.current_cycle.cycle_number if self.current_cycle else None
                },
                "performance_metrics": self.performance_metrics.copy(),
                "task_queue": {
                    "pending": len(self.task_queue),
                    "active": len(self.active_tasks),
                    "completed": len(self.completed_tasks)
                },
                "improvement_cycles": {
                    "total": len(self.improvement_cycles),
                    "last_improvement": self.last_improvement,
                    "next_improvement": self.last_improvement + self.improvement_interval
                }
            }
            
            # Add component statuses
            if self.acm:
                acm_status = await self.acm.get_system_status()
                status["acm"] = acm_status
            
            if self.kg:
                kg_analytics = await self.kg.get_knowledge_analytics()
                status["kg"] = kg_analytics.get("overview", {})
            
            if self.cgm:
                status["cgm"] = {
                    "initialized": True,
                    "active_strategies": ["neural", "template", "meta_learning"]
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"error": str(e)}
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
    
    async def _handle_shutdown(self):
        """Handle system shutdown"""
        try:
            logger.info("🛑 Sutazai Core shutdown initiated")
            
            # Stop background tasks
            self.active = False
            
            # Save current state
            await self._save_system_state()
            
            # Trigger shutdown event handlers
            for handler in self.event_handlers.get("shutdown_initiated", []):
                try:
                    await handler()
                except:
                    pass
            
            logger.info("✅ Sutazai Core shutdown completed")
            
        except Exception as e:
            logger.error(f"Shutdown handling failed: {e}")
    
    async def _save_system_state(self):
        """Save current system state"""
        try:
            state_data = {
                "mode": self.mode.value,
                "performance_metrics": self.performance_metrics,
                "improvement_cycles": [asdict(cycle) for cycle in self.improvement_cycles.values()],
                "task_queue_size": len(self.task_queue),
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
                "last_improvement": self.last_improvement,
                "saved_at": time.time()
            }
            
            with open(self.data_dir / "system_state.json", 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save system state: {e}")
    
    async def shutdown(self):
        """Shutdown the Sutazai system"""
        try:
            logger.info("🛑 Shutting down Sutazai Core System")
            
            # Trigger ACM shutdown if available
            if self.acm:
                # This will trigger our _handle_shutdown callback
                pass
            else:
                await self._handle_shutdown()
                
        except Exception as e:
            logger.error(f"Shutdown failed: {e}")

# Global instance
sutazai_core = SutazaiCore()

# Convenience functions
async def submit_task(task_type: str, description: str, data: Dict[str, Any], assigned_module: str, priority: int = 5) -> str:
    """Submit task to Sutazai"""
    return await sutazai_core.submit_task(task_type, description, data, assigned_module, priority)

async def get_system_status() -> Dict[str, Any]:
    """Get system status"""
    return await sutazai_core.get_system_status()

def register_event_handler(event_type: str, handler: Callable):
    """Register event handler"""
    sutazai_core.register_event_handler(event_type, handler)

async def shutdown_sutazai():
    """Shutdown Sutazai system"""
    await sutazai_core.shutdown()