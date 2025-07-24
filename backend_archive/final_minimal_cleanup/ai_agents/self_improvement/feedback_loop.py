"""
AI-Driven Self-Improvement Feedback Loop
Monitors system performance, identifies issues, and proposes improvements
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import os
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class ImprovementType(Enum):
    """Types of improvements the system can make"""
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    SECURITY = "security"
    USABILITY = "usability"
    CODE_QUALITY = "code_quality"

class ImprovementPriority(Enum):
    """Priority levels for improvements"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class PerformanceMetric:
    """Performance metric data"""
    name: str
    value: float
    timestamp: datetime
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class Issue:
    """Identified system issue"""
    id: str
    type: ImprovementType
    priority: ImprovementPriority
    description: str
    metrics: List[PerformanceMetric]
    detected_at: datetime
    root_cause: Optional[str] = None
    affected_components: List[str] = field(default_factory=list)

@dataclass
class Improvement:
    """Proposed improvement"""
    id: str
    issue_id: str
    type: ImprovementType
    description: str
    implementation_plan: List[str]
    expected_impact: Dict[str, Any]
    code_changes: Dict[str, str]  # file_path -> new_content
    requires_approval: bool = True
    approved: bool = False
    implemented: bool = False

class MetricsCollector:
    """Collects system metrics for analysis"""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetric] = []
        self.collection_interval = 60  # seconds
        
    async def collect_metrics(self) -> List[PerformanceMetric]:
        """Collect current system metrics"""
        metrics = []
        
        # API response time
        metrics.append(PerformanceMetric(
            name="api_response_time",
            value=await self._get_api_response_time(),
            timestamp=datetime.utcnow(),
            unit="ms",
            tags={"component": "api"}
        ))
        
        # Model inference time
        metrics.append(PerformanceMetric(
            name="model_inference_time",
            value=await self._get_model_inference_time(),
            timestamp=datetime.utcnow(),
            unit="ms",
            tags={"component": "models"}
        ))
        
        # Memory usage
        metrics.append(PerformanceMetric(
            name="memory_usage",
            value=await self._get_memory_usage(),
            timestamp=datetime.utcnow(),
            unit="MB",
            tags={"component": "system"}
        ))
        
        # Error rate
        metrics.append(PerformanceMetric(
            name="error_rate",
            value=await self._get_error_rate(),
            timestamp=datetime.utcnow(),
            unit="%",
            tags={"component": "api"}
        ))
        
        # Agent utilization
        metrics.append(PerformanceMetric(
            name="agent_utilization",
            value=await self._get_agent_utilization(),
            timestamp=datetime.utcnow(),
            unit="%",
            tags={"component": "agents"}
        ))
        
        self.metrics_history.extend(metrics)
        return metrics
    
    async def _get_api_response_time(self) -> float:
        """Get average API response time"""
        # In production, this would query actual metrics
        import random
        return 100 + random.randint(-20, 50)
    
    async def _get_model_inference_time(self) -> float:
        """Get average model inference time"""
        import random
        return 500 + random.randint(-100, 200)
    
    async def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 1024.0  # Default value
    
    async def _get_error_rate(self) -> float:
        """Get current error rate"""
        import random
        return random.uniform(0, 5)
    
    async def _get_agent_utilization(self) -> float:
        """Get agent utilization percentage"""
        import random
        return random.uniform(40, 80)
    
    def get_metric_trends(self, metric_name: str, hours: int = 24) -> List[PerformanceMetric]:
        """Get metric trends over time"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            m for m in self.metrics_history
            if m.name == metric_name and m.timestamp > cutoff_time
        ]

class IssueDetector:
    """Detects issues from metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.detected_issues: List[Issue] = []
        
    async def analyze_metrics(self) -> List[Issue]:
        """Analyze metrics and detect issues"""
        new_issues = []
        
        # Check API response time
        response_times = self.metrics_collector.get_metric_trends("api_response_time", hours=1)
        if response_times:
            avg_response_time = sum(m.value for m in response_times) / len(response_times)
            if avg_response_time > 200:  # ms threshold
                issue = Issue(
                    id=f"perf_{datetime.utcnow().timestamp()}",
                    type=ImprovementType.PERFORMANCE,
                    priority=ImprovementPriority.HIGH if avg_response_time > 500 else ImprovementPriority.MEDIUM,
                    description=f"High API response time: {avg_response_time:.0f}ms average",
                    metrics=response_times[-10:],  # Last 10 metrics
                    detected_at=datetime.utcnow(),
                    affected_components=["api", "backend"]
                )
                new_issues.append(issue)
        
        # Check error rate
        error_rates = self.metrics_collector.get_metric_trends("error_rate", hours=1)
        if error_rates:
            avg_error_rate = sum(m.value for m in error_rates) / len(error_rates)
            if avg_error_rate > 2:  # % threshold
                issue = Issue(
                    id=f"err_{datetime.utcnow().timestamp()}",
                    type=ImprovementType.ACCURACY,
                    priority=ImprovementPriority.CRITICAL if avg_error_rate > 5 else ImprovementPriority.HIGH,
                    description=f"High error rate: {avg_error_rate:.1f}% average",
                    metrics=error_rates[-10:],
                    detected_at=datetime.utcnow(),
                    affected_components=["api", "models"]
                )
                new_issues.append(issue)
        
        # Check memory usage
        memory_usage = self.metrics_collector.get_metric_trends("memory_usage", hours=1)
        if memory_usage:
            current_memory = memory_usage[-1].value if memory_usage else 0
            if current_memory > 4096:  # MB threshold
                issue = Issue(
                    id=f"mem_{datetime.utcnow().timestamp()}",
                    type=ImprovementType.EFFICIENCY,
                    priority=ImprovementPriority.HIGH if current_memory > 8192 else ImprovementPriority.MEDIUM,
                    description=f"High memory usage: {current_memory:.0f}MB",
                    metrics=memory_usage[-10:],
                    detected_at=datetime.utcnow(),
                    affected_components=["system", "models"]
                )
                new_issues.append(issue)
        
        self.detected_issues.extend(new_issues)
        return new_issues

class ImprovementGenerator:
    """Generates improvement proposals based on detected issues"""
    
    def __init__(self):
        self.generated_improvements: List[Improvement] = []
        
    async def generate_improvements(self, issues: List[Issue]) -> List[Improvement]:
        """Generate improvement proposals for issues"""
        improvements = []
        
        for issue in issues:
            if issue.type == ImprovementType.PERFORMANCE:
                improvement = await self._generate_performance_improvement(issue)
            elif issue.type == ImprovementType.ACCURACY:
                improvement = await self._generate_accuracy_improvement(issue)
            elif issue.type == ImprovementType.EFFICIENCY:
                improvement = await self._generate_efficiency_improvement(issue)
            else:
                improvement = await self._generate_generic_improvement(issue)
            
            if improvement:
                improvements.append(improvement)
                self.generated_improvements.append(improvement)
        
        return improvements
    
    async def _generate_performance_improvement(self, issue: Issue) -> Improvement:
        """Generate performance improvement"""
        # Analyze the issue and generate specific improvements
        if "api" in issue.affected_components:
            return Improvement(
                id=f"imp_perf_{datetime.utcnow().timestamp()}",
                issue_id=issue.id,
                type=ImprovementType.PERFORMANCE,
                description="Optimize API response time with caching and async improvements",
                implementation_plan=[
                    "Add Redis caching for frequent queries",
                    "Implement connection pooling",
                    "Optimize database queries with indexing",
                    "Add request batching for multiple operations"
                ],
                expected_impact={
                    "response_time_reduction": "40-60%",
                    "throughput_increase": "2x",
                    "resource_usage": "reduced"
                },
                code_changes={
                    "/opt/sutazaiapp/backend/app/core/cache.py": self._generate_cache_code(),
                    "/opt/sutazaiapp/backend/app/db/optimizations.py": self._generate_db_optimization_code()
                }
            )
        return None
    
    async def _generate_accuracy_improvement(self, issue: Issue) -> Improvement:
        """Generate accuracy improvement"""
        return Improvement(
            id=f"imp_acc_{datetime.utcnow().timestamp()}",
            issue_id=issue.id,
            type=ImprovementType.ACCURACY,
            description="Implement enhanced error handling and validation",
            implementation_plan=[
                "Add comprehensive input validation",
                "Implement retry logic with exponential backoff",
                "Add detailed error logging and monitoring",
                "Improve model error recovery"
            ],
            expected_impact={
                "error_rate_reduction": "70-80%",
                "reliability_increase": "significant",
                "user_experience": "improved"
            },
            code_changes={
                "/opt/sutazaiapp/backend/app/core/error_handler.py": self._generate_error_handler_code(),
                "/opt/sutazaiapp/backend/app/core/validation.py": self._generate_validation_code()
            }
        )
    
    async def _generate_efficiency_improvement(self, issue: Issue) -> Improvement:
        """Generate efficiency improvement"""
        return Improvement(
            id=f"imp_eff_{datetime.utcnow().timestamp()}",
            issue_id=issue.id,
            type=ImprovementType.EFFICIENCY,
            description="Optimize memory usage and resource allocation",
            implementation_plan=[
                "Implement memory pooling for models",
                "Add garbage collection optimization",
                "Use memory-mapped files for large data",
                "Implement lazy loading for resources"
            ],
            expected_impact={
                "memory_reduction": "30-50%",
                "startup_time": "faster",
                "scalability": "improved"
            },
            code_changes={
                "/opt/sutazaiapp/backend/app/core/memory_manager.py": self._generate_memory_manager_code()
            }
        )
    
    async def _generate_generic_improvement(self, issue: Issue) -> Improvement:
        """Generate generic improvement"""
        return Improvement(
            id=f"imp_gen_{datetime.utcnow().timestamp()}",
            issue_id=issue.id,
            type=issue.type,
            description=f"Address {issue.type.value} issue",
            implementation_plan=[
                f"Analyze {issue.description}",
                "Implement targeted fixes",
                "Monitor improvements"
            ],
            expected_impact={
                "issue_resolution": "expected",
                "system_stability": "improved"
            },
            code_changes={}
        )
    
    def _generate_cache_code(self) -> str:
        """Generate caching implementation code"""
        return '''"""
Redis caching implementation for API optimization
"""
import redis.asyncio as redis
from typing import Any, Optional
import json
import hashlib

class CacheManager:
    """Manages Redis caching for API responses"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.default_ttl = 300  # 5 minutes
        
    async def connect(self):
        """Connect to Redis"""
        self.redis_client = await redis.from_url(self.redis_url)
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.redis_client:
            return None
            
        value = await self.redis_client.get(key)
        if value:
            return json.loads(value)
        return None
        
    async def set(self, key: str, value: Any, ttl: int = None):
        """Set value in cache"""
        if not self.redis_client:
            return
            
        ttl = ttl or self.default_ttl
        await self.redis_client.set(
            key,
            json.dumps(value),
            ex=ttl
        )
        
    def generate_key(self, prefix: str, params: dict) -> str:
        """Generate cache key from parameters"""
        param_str = json.dumps(params, sort_keys=True)
        hash_str = hashlib.md5(param_str.encode()).hexdigest()
        return f"{prefix}:{hash_str}"

# Global cache instance
cache_manager = CacheManager()
'''
    
    def _generate_db_optimization_code(self) -> str:
        """Generate database optimization code"""
        return '''"""
Database optimization utilities
"""
from sqlalchemy import create_index, text
from typing import List, Dict
import asyncio

class DatabaseOptimizer:
    """Optimizes database performance"""
    
    async def create_indexes(self, db_session):
        """Create performance indexes"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status)",
            "CREATE INDEX IF NOT EXISTS idx_thoughts_timestamp ON thoughts(timestamp DESC)"
        ]
        
        for index_sql in indexes:
            await db_session.execute(text(index_sql))
            
    async def analyze_slow_queries(self) -> List[Dict[str, Any]]:
        """Analyze and identify slow queries"""
        # Implementation would analyze query logs
        return []
        
    async def optimize_connection_pool(self, engine):
        """Optimize database connection pooling"""
        engine.pool._recycle = 3600  # Recycle connections after 1 hour
        engine.pool._pre_ping = True  # Enable connection health checks
'''
    
    def _generate_error_handler_code(self) -> str:
        """Generate error handler code"""
        return '''"""
Enhanced error handling system
"""
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import logging
import traceback
from typing import Any
import asyncio

logger = logging.getLogger(__name__)

class ErrorHandler:
    """Comprehensive error handling"""
    
    def __init__(self):
        self.error_count = {}
        self.retry_config = {
            "max_retries": 3,
            "base_delay": 1,
            "max_delay": 60
        }
        
    async def handle_error(self, error: Exception, context: Dict[str, Any] = None):
        """Handle errors with logging and recovery"""
        error_type = type(error).__name__
        self.error_count[error_type] = self.error_count.get(error_type, 0) + 1
        
        logger.error(f"Error: {error_type} - {str(error)}")
        if context:
            logger.error(f"Context: {context}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Attempt recovery based on error type
        if isinstance(error, ConnectionError):
            return await self.handle_connection_error(error)
        elif isinstance(error, TimeoutError):
            return await self.handle_timeout_error(error)
        else:
            return self.create_error_response(error)
            
    async def retry_with_backoff(self, func, *args, **kwargs):
        """Retry function with exponential backoff"""
        for attempt in range(self.retry_config["max_retries"]):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.retry_config["max_retries"] - 1:
                    raise
                    
                delay = min(
                    self.retry_config["base_delay"] * (2 ** attempt),
                    self.retry_config["max_delay"]
                )
                logger.warning(f"Retry attempt {attempt + 1}, waiting {delay}s")
                await asyncio.sleep(delay)
                
    def create_error_response(self, error: Exception) -> JSONResponse:
        """Create standardized error response"""
        return JSONResponse(
            status_code=500,
            content={
                "error": type(error).__name__,
                "message": str(error),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

# Global error handler
error_handler = ErrorHandler()
'''
    
    def _generate_validation_code(self) -> str:
        """Generate validation code"""
        return '''"""
Input validation system
"""
from pydantic import BaseModel, validator, Field
from typing import Any, List, Optional
import re

class ValidationError(Exception):
    """Custom validation error"""
    pass

class InputValidator:
    """Validates API inputs"""
    
    @staticmethod
    def validate_text_input(text: str, max_length: int = 10000) -> str:
        """Validate text input"""
        if not text or not text.strip():
            raise ValidationError("Text input cannot be empty")
            
        if len(text) > max_length:
            raise ValidationError(f"Text exceeds maximum length of {max_length}")
            
        # Check for malicious patterns
        if re.search(r'<script|javascript:|onerror=', text, re.IGNORECASE):
            raise ValidationError("Potentially malicious content detected")
            
        return text.strip()
        
    @staticmethod
    def validate_model_name(model: str) -> str:
        """Validate model name"""
        valid_models = [
            "deepseek-r1:8b",
            "qwen2.5:3b", 
            "llama3.2:3b",
            "codellama:7b"
        ]
        
        if model not in valid_models:
            raise ValidationError(f"Invalid model: {model}")
            
        return model
        
    @staticmethod
    def validate_reasoning_type(reasoning_type: str) -> str:
        """Validate reasoning type"""
        valid_types = [
            "deductive", "inductive", "abductive",
            "analogical", "causal", "creative", "strategic"
        ]
        
        if reasoning_type not in valid_types:
            raise ValidationError(f"Invalid reasoning type: {reasoning_type}")
            
        return reasoning_type

# Global validator
input_validator = InputValidator()
'''
    
    def _generate_memory_manager_code(self) -> str:
        """Generate memory manager code"""
        return '''"""
Memory optimization manager
"""
import gc
import psutil
import asyncio
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages memory usage and optimization"""
    
    def __init__(self):
        self.memory_threshold = 0.8  # 80% threshold
        self.check_interval = 60  # seconds
        
    async def start_monitoring(self):
        """Start memory monitoring loop"""
        while True:
            await self.check_memory_usage()
            await asyncio.sleep(self.check_interval)
            
    async def check_memory_usage(self):
        """Check current memory usage"""
        memory = psutil.virtual_memory()
        usage_percent = memory.percent / 100
        
        if usage_percent > self.memory_threshold:
            logger.warning(f"High memory usage: {usage_percent:.1%}")
            await self.optimize_memory()
            
    async def optimize_memory(self):
        """Optimize memory usage"""
        # Force garbage collection
        gc.collect()
        
        # Clear caches
        await self.clear_caches()
        
        # Unload unused models
        await self.unload_unused_models()
        
        logger.info("Memory optimization completed")
        
    async def clear_caches(self):
        """Clear various caches"""
        # Implementation would clear specific caches
        pass
        
    async def unload_unused_models(self):
        """Unload models not used recently"""
        # Implementation would track and unload models
        pass

# Global memory manager
memory_manager = MemoryManager()
'''

class FeedbackLoop:
    """Main feedback loop coordinator"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.issue_detector = IssueDetector(self.metrics_collector)
        self.improvement_generator = ImprovementGenerator()
        self.is_running = False
        self.check_interval = 300  # 5 minutes
        self.approval_webhook = os.getenv("APPROVAL_WEBHOOK_URL")
        
    async def start(self):
        """Start the feedback loop"""
        self.is_running = True
        logger.info("Starting AI self-improvement feedback loop")
        
        # Start background tasks
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._issue_detection_loop())
        asyncio.create_task(self._improvement_loop())
        
    async def stop(self):
        """Stop the feedback loop"""
        self.is_running = False
        logger.info("Stopping AI self-improvement feedback loop")
        
    async def _metrics_collection_loop(self):
        """Continuously collect metrics"""
        while self.is_running:
            try:
                metrics = await self.metrics_collector.collect_metrics()
                logger.info(f"Collected {len(metrics)} metrics")
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                
            await asyncio.sleep(self.metrics_collector.collection_interval)
            
    async def _issue_detection_loop(self):
        """Continuously detect issues"""
        while self.is_running:
            try:
                issues = await self.issue_detector.analyze_metrics()
                if issues:
                    logger.warning(f"Detected {len(issues)} new issues")
                    for issue in issues:
                        logger.warning(f"Issue: {issue.description} (Priority: {issue.priority.value})")
            except Exception as e:
                logger.error(f"Error detecting issues: {e}")
                
            await asyncio.sleep(self.check_interval)
            
    async def _improvement_loop(self):
        """Generate and apply improvements"""
        while self.is_running:
            try:
                # Get recent unresolved issues
                recent_issues = [
                    issue for issue in self.issue_detector.detected_issues
                    if not any(imp.issue_id == issue.id and imp.implemented 
                             for imp in self.improvement_generator.generated_improvements)
                ]
                
                if recent_issues:
                    # Generate improvements
                    improvements = await self.improvement_generator.generate_improvements(recent_issues)
                    
                    if improvements:
                        logger.info(f"Generated {len(improvements)} improvements")
                        
                        # Process improvements
                        for improvement in improvements:
                            await self._process_improvement(improvement)
                            
            except Exception as e:
                logger.error(f"Error in improvement loop: {e}")
                
            await asyncio.sleep(self.check_interval * 2)  # Run less frequently
            
    async def _process_improvement(self, improvement: Improvement):
        """Process a single improvement"""
        logger.info(f"Processing improvement: {improvement.description}")
        
        if improvement.requires_approval:
            # Send for approval
            approved = await self._request_approval(improvement)
            improvement.approved = approved
            
            if not approved:
                logger.info(f"Improvement {improvement.id} not approved")
                return
                
        # Apply improvement
        success = await self._apply_improvement(improvement)
        improvement.implemented = success
        
        if success:
            logger.info(f"Successfully applied improvement {improvement.id}")
        else:
            logger.error(f"Failed to apply improvement {improvement.id}")
            
    async def _request_approval(self, improvement: Improvement) -> bool:
        """Request human approval for improvement"""
        if self.approval_webhook:
            # Send webhook notification
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "improvement_id": improvement.id,
                        "type": improvement.type.value,
                        "description": improvement.description,
                        "expected_impact": improvement.expected_impact,
                        "implementation_plan": improvement.implementation_plan
                    }
                    
                    async with session.post(self.approval_webhook, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result.get("approved", False)
            except Exception as e:
                logger.error(f"Error requesting approval: {e}")
                
        # Auto-approve low-risk improvements
        if improvement.type in [ImprovementType.PERFORMANCE, ImprovementType.EFFICIENCY]:
            if len(improvement.code_changes) <= 2:  # Small changes
                logger.info(f"Auto-approving low-risk improvement {improvement.id}")
                return True
                
        return False
        
    async def _apply_improvement(self, improvement: Improvement) -> bool:
        """Apply improvement to the system"""
        try:
            # Apply code changes
            for file_path, new_content in improvement.code_changes.items():
                # Create backup
                backup_path = f"{file_path}.backup_{datetime.utcnow().timestamp()}"
                
                try:
                    # Read original file
                    if os.path.exists(file_path):
                        with open(file_path, 'r') as f:
                            original_content = f.read()
                        
                        # Write backup
                        with open(backup_path, 'w') as f:
                            f.write(original_content)
                    
                    # Write new content
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'w') as f:
                        f.write(new_content)
                        
                    logger.info(f"Applied changes to {file_path}")
                    
                except Exception as e:
                    logger.error(f"Error applying changes to {file_path}: {e}")
                    # Restore backup if exists
                    if os.path.exists(backup_path):
                        import shutil
                        shutil.move(backup_path, file_path)
                    return False
                    
            # Restart affected services if needed
            if "api" in improvement.expected_impact:
                logger.info("Restarting API services...")
                # Would trigger service restart here
                
            return True
            
        except Exception as e:
            logger.error(f"Error applying improvement: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current feedback loop status"""
        return {
            "is_running": self.is_running,
            "metrics_collected": len(self.metrics_collector.metrics_history),
            "issues_detected": len(self.issue_detector.detected_issues),
            "improvements_generated": len(self.improvement_generator.generated_improvements),
            "improvements_implemented": sum(
                1 for imp in self.improvement_generator.generated_improvements
                if imp.implemented
            ),
            "recent_issues": [
                {
                    "id": issue.id,
                    "type": issue.type.value,
                    "priority": issue.priority.value,
                    "description": issue.description
                }
                for issue in self.issue_detector.detected_issues[-5:]
            ]
        }

# Global feedback loop instance
feedback_loop = FeedbackLoop()

# API endpoints for feedback loop control
from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.post("/start")
async def start_feedback_loop():
    """Start the self-improvement feedback loop"""
    if feedback_loop.is_running:
        raise HTTPException(status_code=400, detail="Feedback loop already running")
    
    await feedback_loop.start()
    return {"status": "started"}

@router.post("/stop")
async def stop_feedback_loop():
    """Stop the self-improvement feedback loop"""
    if not feedback_loop.is_running:
        raise HTTPException(status_code=400, detail="Feedback loop not running")
    
    await feedback_loop.stop()
    return {"status": "stopped"}

@router.get("/status")
async def get_feedback_loop_status():
    """Get feedback loop status"""
    return feedback_loop.get_status()

@router.get("/metrics/recent")
async def get_recent_metrics():
    """Get recent metrics"""
    metrics = feedback_loop.metrics_collector.metrics_history[-100:]
    return {
        "count": len(metrics),
        "metrics": [
            {
                "name": m.name,
                "value": m.value,
                "timestamp": m.timestamp.isoformat(),
                "unit": m.unit,
                "tags": m.tags
            }
            for m in metrics
        ]
    }

@router.get("/improvements")
async def get_improvements():
    """Get all generated improvements"""
    return {
        "total": len(feedback_loop.improvement_generator.generated_improvements),
        "implemented": sum(
            1 for imp in feedback_loop.improvement_generator.generated_improvements
            if imp.implemented
        ),
        "improvements": [
            {
                "id": imp.id,
                "type": imp.type.value,
                "description": imp.description,
                "approved": imp.approved,
                "implemented": imp.implemented,
                "expected_impact": imp.expected_impact
            }
            for imp in feedback_loop.improvement_generator.generated_improvements
        ]
    }

@router.post("/improvements/{improvement_id}/approve")
async def approve_improvement(improvement_id: str):
    """Manually approve an improvement"""
    improvement = next(
        (imp for imp in feedback_loop.improvement_generator.generated_improvements
         if imp.id == improvement_id),
        None
    )
    
    if not improvement:
        raise HTTPException(status_code=404, detail="Improvement not found")
    
    improvement.approved = True
    
    # Apply the improvement
    success = await feedback_loop._apply_improvement(improvement)
    improvement.implemented = success
    
    return {
        "id": improvement_id,
        "approved": True,
        "implemented": success
    }