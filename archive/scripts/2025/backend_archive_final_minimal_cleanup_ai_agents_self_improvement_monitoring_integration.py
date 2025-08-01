#!/usr/bin/env python3
"""
SutazAI v9 Monitoring Integration for Autonomous Coder
Integrates self-improvement metrics with Prometheus and health monitoring
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"

@dataclass
class CodeQualityMetrics:
    """Code quality metrics"""
    complexity_score: float
    maintainability_score: float
    security_score: float
    performance_score: float
    test_coverage: float
    documentation_score: float
    overall_score: float
    timestamp: float

@dataclass
class SelfImprovementMetrics:
    """Self-improvement process metrics"""
    improvements_attempted: int
    improvements_successful: int
    improvements_failed: int
    improvements_pending_approval: int
    code_lines_analyzed: int
    issues_detected: int
    issues_fixed: int
    processing_time_seconds: float
    timestamp: float

class PrometheusMetricsCollector:
    """Collects and exports metrics to Prometheus"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or prometheus_client.REGISTRY
        
        # Initialize Prometheus metrics
        self._init_prometheus_metrics()
        
        # Metrics cache
        self.metrics_cache = {}
        self.cache_lock = threading.Lock()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metric objects"""
        
        # Code quality metrics
        self.code_quality_gauge = Gauge(
            'sutazai_code_quality_score',
            'Overall code quality score (0-100)',
            ['metric_type'],
            registry=self.registry
        )
        
        self.code_complexity_histogram = Histogram(
            'sutazai_code_complexity',
            'Code complexity distribution',
            ['file_type'],
            registry=self.registry
        )
        
        # Self-improvement metrics
        self.improvement_attempts_counter = Counter(
            'sutazai_improvement_attempts_total',
            'Total number of improvement attempts',
            ['status', 'improvement_type'],
            registry=self.registry
        )
        
        self.improvement_processing_time = Histogram(
            'sutazai_improvement_processing_seconds',
            'Time spent processing improvements',
            ['improvement_type'],
            registry=self.registry
        )
        
        self.code_analysis_duration = Histogram(
            'sutazai_code_analysis_duration_seconds',
            'Time spent analyzing code',
            ['analysis_type'],
            registry=self.registry
        )
        
        self.issues_detected_counter = Counter(
            'sutazai_issues_detected_total',
            'Total number of issues detected',
            ['issue_type', 'severity'],
            registry=self.registry
        )
        
        self.issues_fixed_counter = Counter(
            'sutazai_issues_fixed_total',
            'Total number of issues fixed',
            ['issue_type', 'fix_type'],
            registry=self.registry
        )
        
        # System health metrics
        self.autonomous_coder_health = Gauge(
            'sutazai_autonomous_coder_health',
            'Health status of autonomous coder (1=healthy, 0.5=degraded, 0=unhealthy)',
            registry=self.registry
        )
        
        self.pending_improvements_gauge = Gauge(
            'sutazai_pending_improvements',
            'Number of improvements pending approval',
            registry=self.registry
        )
        
        # Performance metrics
        self.lines_analyzed_counter = Counter(
            'sutazai_lines_analyzed_total',
            'Total lines of code analyzed',
            ['language'],
            registry=self.registry
        )
        
        self.test_coverage_gauge = Gauge(
            'sutazai_test_coverage_percent',
            'Test coverage percentage',
            ['module'],
            registry=self.registry
        )
        
        # Resource usage metrics
        self.ai_model_usage_histogram = Histogram(
            'sutazai_ai_model_usage_seconds',
            'AI model usage time for code analysis',
            ['model_name', 'task_type'],
            registry=self.registry
        )
        
        self.memory_usage_gauge = Gauge(
            'sutazai_autonomous_coder_memory_bytes',
            'Memory usage by autonomous coder',
            registry=self.registry
        )
    
    def record_code_quality_metrics(self, metrics: CodeQualityMetrics):
        """Record code quality metrics"""
        with self.cache_lock:
            self.metrics_cache['code_quality'] = metrics
        
        # Update Prometheus metrics
        self.code_quality_gauge.labels(metric_type='overall').set(metrics.overall_score)
        self.code_quality_gauge.labels(metric_type='complexity').set(metrics.complexity_score)
        self.code_quality_gauge.labels(metric_type='maintainability').set(metrics.maintainability_score)
        self.code_quality_gauge.labels(metric_type='security').set(metrics.security_score)
        self.code_quality_gauge.labels(metric_type='performance').set(metrics.performance_score)
        self.code_quality_gauge.labels(metric_type='documentation').set(metrics.documentation_score)
        
        self.test_coverage_gauge.labels(module='overall').set(metrics.test_coverage)
        
        logger.info(f"Recorded code quality metrics: overall_score={metrics.overall_score:.2f}")
    
    def record_self_improvement_metrics(self, metrics: SelfImprovementMetrics):
        """Record self-improvement metrics"""
        with self.cache_lock:
            self.metrics_cache['self_improvement'] = metrics
        
        # Update gauges
        self.pending_improvements_gauge.set(metrics.improvements_pending_approval)
        
        logger.info(f"Recorded self-improvement metrics: successful={metrics.improvements_successful}, "
                   f"failed={metrics.improvements_failed}")
    
    def record_improvement_attempt(self, improvement_type: str, status: str, processing_time: float):
        """Record an improvement attempt"""
        self.improvement_attempts_counter.labels(
            status=status,
            improvement_type=improvement_type
        ).inc()
        
        self.improvement_processing_time.labels(
            improvement_type=improvement_type
        ).observe(processing_time)
    
    def record_code_analysis(self, analysis_type: str, duration: float, lines_analyzed: int, language: str):
        """Record code analysis metrics"""
        self.code_analysis_duration.labels(analysis_type=analysis_type).observe(duration)
        self.lines_analyzed_counter.labels(language=language).inc(lines_analyzed)
    
    def record_issue_detection(self, issue_type: str, severity: str, count: int = 1):
        """Record detected issues"""
        self.issues_detected_counter.labels(
            issue_type=issue_type,
            severity=severity
        ).inc(count)
    
    def record_issue_fix(self, issue_type: str, fix_type: str, count: int = 1):
        """Record fixed issues"""
        self.issues_fixed_counter.labels(
            issue_type=issue_type,
            fix_type=fix_type
        ).inc(count)
    
    def record_ai_model_usage(self, model_name: str, task_type: str, duration: float):
        """Record AI model usage"""
        self.ai_model_usage_histogram.labels(
            model_name=model_name,
            task_type=task_type
        ).observe(duration)
    
    def update_health_status(self, health_score: float):
        """Update autonomous coder health status"""
        self.autonomous_coder_health.set(health_score)
    
    def update_memory_usage(self, memory_bytes: int):
        """Update memory usage"""
        self.memory_usage_gauge.set(memory_bytes)
    
    def get_cached_metrics(self) -> Dict[str, Any]:
        """Get cached metrics for health checks"""
        with self.cache_lock:
            return self.metrics_cache.copy()

class HealthIntegration:
    """Integrates autonomous coder with health monitoring system"""
    
    def __init__(self, prometheus_collector: PrometheusMetricsCollector):
        self.prometheus_collector = prometheus_collector
        self.health_checks = {}
        self.alert_thresholds = self._init_alert_thresholds()
    
    def _init_alert_thresholds(self) -> Dict[str, Any]:
        """Initialize alert thresholds"""
        return {
            'code_quality_critical': 50.0,
            'code_quality_warning': 70.0,
            'max_pending_improvements': 10,
            'max_failure_rate': 0.3,
            'max_processing_time': 600,  # 10 minutes
            'min_test_coverage': 60.0
        }
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check for autonomous coder"""
        checks = {}
        overall_status = "healthy"
        
        # Check code quality
        quality_check = await self._check_code_quality()
        checks['code_quality'] = quality_check
        if quality_check['status'] != 'healthy':
            overall_status = 'degraded' if overall_status == 'healthy' else 'unhealthy'
        
        # Check self-improvement system
        improvement_check = await self._check_improvement_system()
        checks['improvement_system'] = improvement_check
        if improvement_check['status'] != 'healthy':
            overall_status = 'degraded' if overall_status == 'healthy' else 'unhealthy'
        
        # Check resource usage
        resource_check = await self._check_resource_usage()
        checks['resources'] = resource_check
        if resource_check['status'] != 'healthy':
            overall_status = 'degraded' if overall_status == 'healthy' else 'unhealthy'
        
        # Check AI model availability
        ai_check = await self._check_ai_models()
        checks['ai_models'] = ai_check
        if ai_check['status'] != 'healthy':
            overall_status = 'unhealthy'
        
        # Update health metric
        health_score = 1.0 if overall_status == 'healthy' else (0.5 if overall_status == 'degraded' else 0.0)
        self.prometheus_collector.update_health_status(health_score)
        
        return {
            'status': overall_status,
            'timestamp': time.time(),
            'checks': checks,
            'thresholds': self.alert_thresholds
        }
    
    async def _check_code_quality(self) -> Dict[str, Any]:
        """Check code quality metrics"""
        try:
            cached_metrics = self.prometheus_collector.get_cached_metrics()
            quality_metrics = cached_metrics.get('code_quality')
            
            if not quality_metrics:
                return {
                    'status': 'unknown',
                    'message': 'No code quality metrics available'
                }
            
            overall_score = quality_metrics.overall_score
            
            if overall_score < self.alert_thresholds['code_quality_critical']:
                status = 'unhealthy'
                message = f'Code quality critically low: {overall_score:.2f}'
            elif overall_score < self.alert_thresholds['code_quality_warning']:
                status = 'degraded'
                message = f'Code quality below threshold: {overall_score:.2f}'
            else:
                status = 'healthy'
                message = f'Code quality good: {overall_score:.2f}'
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'overall_score': overall_score,
                    'complexity_score': quality_metrics.complexity_score,
                    'security_score': quality_metrics.security_score,
                    'test_coverage': quality_metrics.test_coverage
                }
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Failed to check code quality: {str(e)}'
            }
    
    async def _check_improvement_system(self) -> Dict[str, Any]:
        """Check self-improvement system health"""
        try:
            cached_metrics = self.prometheus_collector.get_cached_metrics()
            improvement_metrics = cached_metrics.get('self_improvement')
            
            if not improvement_metrics:
                return {
                    'status': 'unknown',
                    'message': 'No improvement metrics available'
                }
            
            # Check pending improvements
            pending = improvement_metrics.improvements_pending_approval
            if pending > self.alert_thresholds['max_pending_improvements']:
                return {
                    'status': 'degraded',
                    'message': f'Too many pending improvements: {pending}',
                    'details': {'pending_improvements': pending}
                }
            
            # Check failure rate
            total_attempts = improvement_metrics.improvements_attempted
            if total_attempts > 0:
                failure_rate = improvement_metrics.improvements_failed / total_attempts
                if failure_rate > self.alert_thresholds['max_failure_rate']:
                    return {
                        'status': 'degraded',
                        'message': f'High improvement failure rate: {failure_rate:.2%}',
                        'details': {'failure_rate': failure_rate}
                    }
            
            # Check processing time
            if improvement_metrics.processing_time_seconds > self.alert_thresholds['max_processing_time']:
                return {
                    'status': 'degraded',
                    'message': 'Improvement processing taking too long',
                    'details': {'processing_time': improvement_metrics.processing_time_seconds}
                }
            
            return {
                'status': 'healthy',
                'message': 'Self-improvement system operating normally',
                'details': {
                    'successful_improvements': improvement_metrics.improvements_successful,
                    'pending_improvements': pending,
                    'processing_time': improvement_metrics.processing_time_seconds
                }
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Failed to check improvement system: {str(e)}'
            }
    
    async def _check_resource_usage(self) -> Dict[str, Any]:
        """Check resource usage"""
        try:
            import psutil
            
            # Check memory usage
            memory_info = psutil.virtual_memory()
            if memory_info.percent > 90:
                status = 'unhealthy'
                message = f'High memory usage: {memory_info.percent:.1f}%'
            elif memory_info.percent > 80:
                status = 'degraded'
                message = f'Elevated memory usage: {memory_info.percent:.1f}%'
            else:
                status = 'healthy'
                message = f'Memory usage normal: {memory_info.percent:.1f}%'
            
            # Update memory metric
            process = psutil.Process()
            memory_bytes = process.memory_info().rss
            self.prometheus_collector.update_memory_usage(memory_bytes)
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'memory_percent': memory_info.percent,
                    'process_memory_mb': memory_bytes / (1024 * 1024)
                }
            }
        except Exception as e:
            return {
                'status': 'unknown',
                'message': f'Failed to check resource usage: {str(e)}'
            }
    
    async def _check_ai_models(self) -> Dict[str, Any]:
        """Check AI model availability"""
        try:
            # This would check if AI models are loaded and responsive
            # Placeholder implementation
            models_status = {
                'tinyllama': 'healthy',
                'qwen3:8b': 'healthy'
            }
            
            unhealthy_models = [name for name, status in models_status.items() if status != 'healthy']
            
            if unhealthy_models:
                return {
                    'status': 'unhealthy',
                    'message': f'AI models not available: {", ".join(unhealthy_models)}',
                    'details': models_status
                }
            else:
                return {
                    'status': 'healthy',
                    'message': 'All AI models available',
                    'details': models_status
                }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Failed to check AI models: {str(e)}'
            }

class AlertManager:
    """Manages alerts based on metrics and health checks"""
    
    def __init__(self, prometheus_collector: PrometheusMetricsCollector):
        self.prometheus_collector = prometheus_collector
        self.alert_rules = self._init_alert_rules()
        self.active_alerts = {}
    
    def _init_alert_rules(self) -> List[Dict[str, Any]]:
        """Initialize alert rules"""
        return [
            {
                'name': 'CodeQualityDegradation',
                'condition': lambda metrics: metrics.get('code_quality', {}).get('overall_score', 100) < 70,
                'severity': 'warning',
                'message': 'Code quality has degraded below acceptable threshold'
            },
            {
                'name': 'HighImprovementFailureRate',
                'condition': lambda metrics: self._check_failure_rate(metrics) > 0.3,
                'severity': 'warning',
                'message': 'High rate of improvement failures detected'
            },
            {
                'name': 'TooManyPendingImprovements',
                'condition': lambda metrics: metrics.get('self_improvement', {}).get('improvements_pending_approval', 0) > 10,
                'severity': 'warning',
                'message': 'Too many improvements pending approval'
            },
            {
                'name': 'SecurityIssuesDetected',
                'condition': lambda metrics: metrics.get('code_quality', {}).get('security_score', 100) < 60,
                'severity': 'critical',
                'message': 'Security issues detected in codebase'
            }
        ]
    
    def _check_failure_rate(self, metrics: Dict[str, Any]) -> float:
        """Calculate improvement failure rate"""
        improvement_metrics = metrics.get('self_improvement', {})
        total = improvement_metrics.get('improvements_attempted', 0)
        failed = improvement_metrics.get('improvements_failed', 0)
        
        return failed / total if total > 0 else 0.0
    
    async def check_alerts(self) -> List[Dict[str, Any]]:
        """Check all alert conditions"""
        cached_metrics = self.prometheus_collector.get_cached_metrics()
        triggered_alerts = []
        
        for rule in self.alert_rules:
            try:
                if rule['condition'](cached_metrics):
                    alert = {
                        'name': rule['name'],
                        'severity': rule['severity'],
                        'message': rule['message'],
                        'timestamp': time.time(),
                        'status': 'firing'
                    }
                    triggered_alerts.append(alert)
                    
                    # Track active alerts
                    self.active_alerts[rule['name']] = alert
                    
                elif rule['name'] in self.active_alerts:
                    # Alert resolved
                    resolved_alert = self.active_alerts.pop(rule['name'])
                    resolved_alert['status'] = 'resolved'
                    resolved_alert['resolved_timestamp'] = time.time()
                    triggered_alerts.append(resolved_alert)
            
            except Exception as e:
                logger.error(f"Error checking alert rule {rule['name']}: {e}")
        
        return triggered_alerts

class MonitoringDashboard:
    """Provides dashboard data for monitoring interfaces"""
    
    def __init__(self, prometheus_collector: PrometheusMetricsCollector, health_integration: HealthIntegration):
        self.prometheus_collector = prometheus_collector
        self.health_integration = health_integration
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        cached_metrics = self.prometheus_collector.get_cached_metrics()
        health_status = await self.health_integration.perform_health_check()
        
        # Calculate summary statistics
        code_quality = cached_metrics.get('code_quality', {})
        improvement_metrics = cached_metrics.get('self_improvement', {})
        
        return {
            'summary': {
                'overall_health': health_status['status'],
                'code_quality_score': getattr(code_quality, 'overall_score', 0) if code_quality else 0,
                'improvements_successful': getattr(improvement_metrics, 'improvements_successful', 0) if improvement_metrics else 0,
                'improvements_pending': getattr(improvement_metrics, 'improvements_pending_approval', 0) if improvement_metrics else 0
            },
            'code_quality': asdict(code_quality) if code_quality else {},
            'self_improvement': asdict(improvement_metrics) if improvement_metrics else {},
            'health_checks': health_status['checks'],
            'timestamp': time.time()
        }

# Example usage and integration
async def main():
    """Example usage of monitoring integration"""
    
    # Initialize components
    prometheus_collector = PrometheusMetricsCollector()
    health_integration = HealthIntegration(prometheus_collector)
    alert_manager = AlertManager(prometheus_collector)
    dashboard = MonitoringDashboard(prometheus_collector, health_integration)
    
    # Example: Record some metrics
    quality_metrics = CodeQualityMetrics(
        complexity_score=85.5,
        maintainability_score=90.2,
        security_score=88.7,
        performance_score=92.1,
        test_coverage=78.3,
        documentation_score=85.0,
        overall_score=86.6,
        timestamp=time.time()
    )
    
    improvement_metrics = SelfImprovementMetrics(
        improvements_attempted=25,
        improvements_successful=22,
        improvements_failed=2,
        improvements_pending_approval=1,
        code_lines_analyzed=15000,
        issues_detected=45,
        issues_fixed=40,
        processing_time_seconds=120.5,
        timestamp=time.time()
    )
    
    # Record metrics
    prometheus_collector.record_code_quality_metrics(quality_metrics)
    prometheus_collector.record_self_improvement_metrics(improvement_metrics)
    
    # Perform health check
    health_report = await health_integration.perform_health_check()
    print(f"Health Status: {health_report['status']}")
    
    # Check alerts
    alerts = await alert_manager.check_alerts()
    print(f"Active Alerts: {len(alerts)}")
    
    # Get dashboard data
    dashboard_data = await dashboard.get_dashboard_data()
    print(f"Dashboard Summary: {dashboard_data['summary']}")

if __name__ == "__main__":
    asyncio.run(main())