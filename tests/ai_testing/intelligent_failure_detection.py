#!/usr/bin/env python3
"""
INTELLIGENT AI FAILURE DETECTION SYSTEM
🧠 Uses AI-powered pattern recognition to detect system failures

This advanced AI testing module uses machine learning and intelligent analysis
to detect failures that traditional testing might miss.
"""

import asyncio
import json
import numpy as np
import pandas as pd
import logging
import requests
import time
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class FailurePattern:
    """AI-detected failure pattern"""
    pattern_id: str
    pattern_type: str
    severity: float
    confidence: float
    affected_services: List[str]
    symptoms: List[str]
    prediction: str
    recommendation: str
    timestamp: str

@dataclass
class AITestResult:
    """AI-enhanced test result"""
    test_name: str
    ai_confidence: float
    anomaly_score: float
    predicted_failure_probability: float
    intelligent_insights: List[str]
    failure_patterns: List[FailurePattern]
    timestamp: str

class IntelligentFailureDetector:
    """AI-powered failure detection system"""
    
    def __init__(self):
        self.base_url = "http://localhost:10010"
        self.historical_data = []
        self.trained_models = {}
        self.failure_patterns = []
        
    async def collect_intelligent_metrics(self) -> Dict[str, Any]:
        """Collect metrics with AI-enhanced analysis"""
        logger.info("🧠 COLLECTING INTELLIGENT METRICS")
        
        metrics = {
            "response_time_patterns": await self._analyze_response_time_patterns(),
            "error_rate_anomalies": await self._detect_error_rate_anomalies(),
            "resource_usage_intelligence": await self._analyze_resource_usage_intelligence(),
            "service_interaction_patterns": await self._analyze_service_interaction_patterns(),
            "behavioral_anomalies": await self._detect_behavioral_anomalies()
        }
        
        return metrics
    
    async def _analyze_response_time_patterns(self) -> Dict[str, Any]:
        """AI analysis of response time patterns"""
        response_times = []
        services_tested = []
        
        # Collect response time data
        for i in range(20):
            start_time = time.time()
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                end_time = time.time()
                
                response_times.append(end_time - start_time)
                services_tested.append("health_endpoint")
                
            except Exception as e:
                response_times.append(5.0)  # Timeout value
                services_tested.append("health_endpoint")
            
            await asyncio.sleep(0.1)  # Small delay between requests
        
        # AI analysis
        if len(response_times) >= 10:
            # Detect anomalies using Isolation Forest
            X = np.array(response_times).reshape(-1, 1)
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(X)
            
            # Calculate statistics
            mean_response_time = np.mean(response_times)
            std_response_time = np.std(response_times)
            anomaly_count = sum(1 for a in anomalies if a == -1)
            
            # AI insights
            insights = []
            if mean_response_time > 2.0:
                insights.append("Response times indicate potential performance degradation")
            if std_response_time > 1.0:
                insights.append("High variance in response times suggests instability")
            if anomaly_count > 2:
                insights.append(f"Detected {anomaly_count} response time anomalies")
            
            return {
                "mean_response_time": mean_response_time,
                "std_response_time": std_response_time,
                "anomaly_count": anomaly_count,
                "anomaly_percentage": (anomaly_count / len(response_times)) * 100,
                "ai_insights": insights,
                "health_score": max(0, 100 - (mean_response_time * 20) - (anomaly_count * 10))
            }
        
        return {"error": "Insufficient data for analysis"}
    
    async def _detect_error_rate_anomalies(self) -> Dict[str, Any]:
        """AI detection of error rate anomalies"""
        error_data = []
        
        # Test multiple endpoints to gather error data
        test_endpoints = [
            "/api/v1/mcp/services",
            "/api/v1/mcp/claude-flow/tools",
            "/api/v1/mcp/ruv-swarm/tools",
            "/api/v1/mcp/memory-bank-mcp/tools"
        ]
        
        for endpoint in test_endpoints:
            endpoint_errors = 0
            total_requests = 10
            
            for _ in range(total_requests):
                try:
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=3)
                    if response.status_code >= 400:
                        endpoint_errors += 1
                except Exception:
                    endpoint_errors += 1
                
                await asyncio.sleep(0.05)
            
            error_rate = endpoint_errors / total_requests
            error_data.append({
                "endpoint": endpoint,
                "error_rate": error_rate,
                "total_requests": total_requests,
                "errors": endpoint_errors
            })
        
        # AI analysis of error patterns
        error_rates = [data["error_rate"] for data in error_data]
        
        if error_rates:
            mean_error_rate = np.mean(error_rates)
            max_error_rate = max(error_rates)
            
            # Detect problematic endpoints
            problematic_endpoints = [
                data["endpoint"] for data in error_data 
                if data["error_rate"] > 0.3  # > 30% error rate
            ]
            
            # AI insights
            insights = []
            if mean_error_rate > 0.2:
                insights.append("System-wide error rate indicates significant issues")
            if max_error_rate > 0.5:
                insights.append("At least one endpoint has critical error rate")
            if len(problematic_endpoints) > 1:
                insights.append(f"Multiple endpoints failing: {problematic_endpoints}")
            
            return {
                "mean_error_rate": mean_error_rate,
                "max_error_rate": max_error_rate,
                "problematic_endpoints": problematic_endpoints,
                "endpoint_analysis": error_data,
                "ai_insights": insights,
                "system_stability_score": max(0, 100 - (mean_error_rate * 100))
            }
        
        return {"error": "No error data collected"}
    
    async def _analyze_resource_usage_intelligence(self) -> Dict[str, Any]:
        """Intelligent analysis of resource usage patterns"""
        import psutil
        
        # Collect resource data over time
        cpu_samples = []
        memory_samples = []
        
        for _ in range(10):
            cpu_samples.append(psutil.cpu_percent(interval=0.1))
            memory_samples.append(psutil.virtual_memory().percent)
            await asyncio.sleep(0.2)
        
        # AI analysis
        cpu_mean = np.mean(cpu_samples)
        cpu_std = np.std(cpu_samples)
        memory_mean = np.mean(memory_samples)
        memory_std = np.std(memory_samples)
        
        # Detect resource anomalies
        insights = []
        
        if cpu_mean > 80:
            insights.append("High CPU utilization detected - potential performance bottleneck")
        if memory_mean > 80:
            insights.append("High memory utilization detected - potential memory leak")
        if cpu_std > 20:
            insights.append("High CPU variance indicates unstable workload")
        if memory_std > 10:
            insights.append("Memory usage variance suggests memory management issues")
        
        # Calculate resource health score
        resource_health = 100
        if cpu_mean > 70:
            resource_health -= (cpu_mean - 70) * 2
        if memory_mean > 70:
            resource_health -= (memory_mean - 70) * 2
        
        return {
            "cpu_utilization": {
                "mean": cpu_mean,
                "std": cpu_std,
                "samples": cpu_samples
            },
            "memory_utilization": {
                "mean": memory_mean,
                "std": memory_std,
                "samples": memory_samples
            },
            "ai_insights": insights,
            "resource_health_score": max(0, resource_health)
        }
    
    async def _analyze_service_interaction_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in service interactions"""
        interaction_data = []
        
        # Test service-to-service interactions
        service_pairs = [
            ("claude-flow", "ruv-swarm"),
            ("memory-bank-mcp", "extended-memory"),
            ("files", "context7")
        ]
        
        for service1, service2 in service_pairs:
            interaction_success = 0
            total_interactions = 5
            
            for _ in range(total_interactions):
                try:
                    # Test if services can be accessed sequentially (simulating interaction)
                    response1 = requests.post(
                        f"{self.base_url}/api/v1/mcp/{service1}/tools",
                        json={
                            "jsonrpc": "2.0",
                            "id": "interaction_test",
                            "method": "tools/list",
                            "params": {}
                        },
                        timeout=3
                    )
                    
                    response2 = requests.post(
                        f"{self.base_url}/api/v1/mcp/{service2}/tools",
                        json={
                            "jsonrpc": "2.0",
                            "id": "interaction_test",
                            "method": "tools/list",
                            "params": {}
                        },
                        timeout=3
                    )
                    
                    if response1.status_code == 200 and response2.status_code == 200:
                        interaction_success += 1
                        
                except Exception:
                    pass
                
                await asyncio.sleep(0.1)
            
            interaction_rate = interaction_success / total_interactions
            interaction_data.append({
                "service_pair": f"{service1} <-> {service2}",
                "success_rate": interaction_rate,
                "total_tests": total_interactions,
                "successful_interactions": interaction_success
            })
        
        # AI analysis
        success_rates = [data["success_rate"] for data in interaction_data]
        mean_success_rate = np.mean(success_rates) if success_rates else 0
        
        insights = []
        if mean_success_rate < 0.7:
            insights.append("Poor service interaction success rate indicates coordination issues")
        
        failed_pairs = [
            data["service_pair"] for data in interaction_data 
            if data["success_rate"] < 0.5
        ]
        
        if failed_pairs:
            insights.append(f"Service pairs with poor interaction: {failed_pairs}")
        
        return {
            "interaction_analysis": interaction_data,
            "mean_success_rate": mean_success_rate,
            "failed_interaction_pairs": failed_pairs,
            "ai_insights": insights,
            "coordination_health_score": mean_success_rate * 100
        }
    
    async def _detect_behavioral_anomalies(self) -> Dict[str, Any]:
        """Detect behavioral anomalies using AI"""
        behavioral_data = []
        
        # Collect behavioral data
        for i in range(15):
            behavior_sample = {
                "timestamp": time.time(),
                "request_id": i,
                "response_pattern": await self._get_response_pattern(),
                "service_availability": await self._check_service_availability(),
                "performance_metrics": await self._get_performance_snapshot()
            }
            behavioral_data.append(behavior_sample)
            await asyncio.sleep(0.2)
        
        # AI anomaly detection
        features = []
        for sample in behavioral_data:
            feature_vector = [
                sample["response_pattern"].get("response_time", 0),
                sample["service_availability"].get("available_services", 0),
                sample["performance_metrics"].get("cpu_usage", 0),
                sample["performance_metrics"].get("memory_usage", 0)
            ]
            features.append(feature_vector)
        
        if len(features) >= 10:
            # Normalize features
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(features)
            
            # Detect anomalies using DBSCAN clustering
            dbscan = DBSCAN(eps=0.5, min_samples=3)
            cluster_labels = dbscan.fit_predict(normalized_features)
            
            # Count anomalies (points labeled as -1)
            anomaly_count = sum(1 for label in cluster_labels if label == -1)
            anomaly_percentage = (anomaly_count / len(features)) * 100
            
            insights = []
            if anomaly_percentage > 20:
                insights.append(f"High behavioral anomaly rate: {anomaly_percentage:.1f}%")
            if anomaly_count > 3:
                insights.append("Multiple behavioral anomalies detected - system instability")
            
            return {
                "total_samples": len(features),
                "anomaly_count": anomaly_count,
                "anomaly_percentage": anomaly_percentage,
                "cluster_analysis": {
                    "unique_clusters": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                    "noise_points": anomaly_count
                },
                "ai_insights": insights,
                "behavioral_health_score": max(0, 100 - anomaly_percentage)
            }
        
        return {"error": "Insufficient behavioral data"}
    
    async def _get_response_pattern(self) -> Dict[str, Any]:
        """Get current response pattern"""
        start_time = time.time()
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            end_time = time.time()
            return {
                "response_time": end_time - start_time,
                "status_code": response.status_code,
                "success": response.status_code == 200
            }
        except Exception:
            return {
                "response_time": 2.0,
                "status_code": 0,
                "success": False
            }
    
    async def _check_service_availability(self) -> Dict[str, Any]:
        """Check current service availability"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/mcp/services", timeout=3)
            if response.status_code == 200:
                services = response.json()
                return {
                    "available_services": len(services) if isinstance(services, list) else 0,
                    "check_successful": True
                }
        except Exception:
            pass
        
        return {
            "available_services": 0,
            "check_successful": False
        }
    
    async def _get_performance_snapshot(self) -> Dict[str, Any]:
        """Get current performance snapshot"""
        import psutil
        
        return {
            "cpu_usage": psutil.cpu_percent(interval=0.1),
            "memory_usage": psutil.virtual_memory().percent,
            "timestamp": time.time()
        }
    
    async def generate_failure_predictions(self, metrics: Dict[str, Any]) -> List[FailurePattern]:
        """Generate AI-powered failure predictions"""
        logger.info("🔮 GENERATING FAILURE PREDICTIONS")
        
        patterns = []
        
        # Analyze response time patterns
        if "response_time_patterns" in metrics:
            rtp = metrics["response_time_patterns"]
            if rtp.get("health_score", 100) < 70:
                pattern = FailurePattern(
                    pattern_id="rtp_001",
                    pattern_type="Performance Degradation",
                    severity=0.8,
                    confidence=0.9,
                    affected_services=["health_endpoint"],
                    symptoms=[
                        f"Mean response time: {rtp.get('mean_response_time', 0):.2f}s",
                        f"Anomaly count: {rtp.get('anomaly_count', 0)}"
                    ],
                    prediction="System performance will continue to degrade",
                    recommendation="Investigate system resources and optimize performance",
                    timestamp=datetime.now().isoformat()
                )
                patterns.append(pattern)
        
        # Analyze error rate anomalies
        if "error_rate_anomalies" in metrics:
            era = metrics["error_rate_anomalies"]
            if era.get("system_stability_score", 100) < 60:
                pattern = FailurePattern(
                    pattern_id="era_001",
                    pattern_type="Service Instability",
                    severity=0.9,
                    confidence=0.85,
                    affected_services=era.get("problematic_endpoints", []),
                    symptoms=[
                        f"Mean error rate: {era.get('mean_error_rate', 0):.1%}",
                        f"Max error rate: {era.get('max_error_rate', 0):.1%}"
                    ],
                    prediction="Service failures will increase without intervention",
                    recommendation="Immediate investigation of failing endpoints required",
                    timestamp=datetime.now().isoformat()
                )
                patterns.append(pattern)
        
        # Analyze resource usage
        if "resource_usage_intelligence" in metrics:
            rui = metrics["resource_usage_intelligence"]
            if rui.get("resource_health_score", 100) < 50:
                pattern = FailurePattern(
                    pattern_id="rui_001",
                    pattern_type="Resource Exhaustion",
                    severity=0.7,
                    confidence=0.8,
                    affected_services=["system_resources"],
                    symptoms=[
                        f"CPU utilization: {rui['cpu_utilization']['mean']:.1f}%",
                        f"Memory utilization: {rui['memory_utilization']['mean']:.1f}%"
                    ],
                    prediction="Resource exhaustion may cause system failures",
                    recommendation="Scale resources or optimize resource usage",
                    timestamp=datetime.now().isoformat()
                )
                patterns.append(pattern)
        
        # Analyze coordination issues
        if "service_interaction_patterns" in metrics:
            sip = metrics["service_interaction_patterns"]
            if sip.get("coordination_health_score", 100) < 70:
                pattern = FailurePattern(
                    pattern_id="sip_001",
                    pattern_type="Coordination Failure",
                    severity=0.6,
                    confidence=0.75,
                    affected_services=sip.get("failed_interaction_pairs", []),
                    symptoms=[
                        f"Mean success rate: {sip.get('mean_success_rate', 0):.1%}",
                        f"Failed pairs: {len(sip.get('failed_interaction_pairs', []))}"
                    ],
                    prediction="Service coordination will continue to deteriorate",
                    recommendation="Fix service communication and coordination mechanisms",
                    timestamp=datetime.now().isoformat()
                )
                patterns.append(pattern)
        
        # Analyze behavioral anomalies
        if "behavioral_anomalies" in metrics:
            ba = metrics["behavioral_anomalies"]
            if ba.get("behavioral_health_score", 100) < 60:
                pattern = FailurePattern(
                    pattern_id="ba_001",
                    pattern_type="Behavioral Instability",
                    severity=0.8,
                    confidence=0.7,
                    affected_services=["system_behavior"],
                    symptoms=[
                        f"Anomaly percentage: {ba.get('anomaly_percentage', 0):.1f}%",
                        f"Anomaly count: {ba.get('anomaly_count', 0)}"
                    ],
                    prediction="System behavior will become increasingly unpredictable",
                    recommendation="Comprehensive system stability analysis required",
                    timestamp=datetime.now().isoformat()
                )
                patterns.append(pattern)
        
        return patterns
    
    async def execute_intelligent_analysis(self) -> AITestResult:
        """Execute comprehensive intelligent failure analysis"""
        logger.info("🧠 EXECUTING INTELLIGENT FAILURE ANALYSIS")
        
        start_time = time.time()
        
        # Collect intelligent metrics
        metrics = await self.collect_intelligent_metrics()
        
        # Generate failure predictions
        failure_patterns = await self.generate_failure_predictions(metrics)
        
        # Calculate AI confidence scores
        health_scores = []
        for metric_category in metrics.values():
            if isinstance(metric_category, dict):
                for key, value in metric_category.items():
                    if "health_score" in key and isinstance(value, (int, float)):
                        health_scores.append(value)
        
        overall_health = np.mean(health_scores) if health_scores else 50
        ai_confidence = min(0.95, overall_health / 100)
        
        # Calculate anomaly score
        anomaly_indicators = []
        for pattern in failure_patterns:
            anomaly_indicators.append(pattern.severity * pattern.confidence)
        
        anomaly_score = np.mean(anomaly_indicators) if anomaly_indicators else 0.0
        
        # Calculate failure probability
        failure_probability = 1.0 - (overall_health / 100)
        
        # Generate intelligent insights
        insights = []
        if overall_health < 60:
            insights.append("Multiple system health indicators below acceptable thresholds")
        if len(failure_patterns) > 2:
            insights.append("Multiple failure patterns detected - high-risk system state")
        if anomaly_score > 0.7:
            insights.append("High anomaly score indicates significant system issues")
        if failure_probability > 0.4:
            insights.append("High probability of system failures without intervention")
        
        end_time = time.time()
        
        result = AITestResult(
            test_name="Intelligent Failure Detection Analysis",
            ai_confidence=ai_confidence,
            anomaly_score=anomaly_score,
            predicted_failure_probability=failure_probability,
            intelligent_insights=insights,
            failure_patterns=failure_patterns,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"🎯 INTELLIGENT ANALYSIS COMPLETE - Confidence: {ai_confidence:.1%}, Anomaly Score: {anomaly_score:.2f}")
        
        return result

async def main():
    """Main execution function for intelligent failure detection"""
    print("🧠 INTELLIGENT AI FAILURE DETECTION SYSTEM")
    print("=" * 60)
    
    detector = IntelligentFailureDetector()
    result = await detector.execute_intelligent_analysis()
    
    # Save results
    results_file = "/opt/sutazaiapp/tests/intelligent_failure_analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump(asdict(result), f, indent=2, default=str)
    
    print(f"\n📊 INTELLIGENT ANALYSIS RESULTS SAVED TO: {results_file}")
    
    # Print summary
    print(f"\n🎯 AI ANALYSIS SUMMARY:")
    print(f"AI Confidence: {result.ai_confidence:.1%}")
    print(f"Anomaly Score: {result.anomaly_score:.2f}")
    print(f"Failure Probability: {result.predicted_failure_probability:.1%}")
    print(f"Patterns Detected: {len(result.failure_patterns)}")
    
    if result.intelligent_insights:
        print(f"\n💡 KEY INSIGHTS:")
        for insight in result.intelligent_insights:
            print(f"  • {insight}")
    
    if result.failure_patterns:
        print(f"\n⚠️  FAILURE PATTERNS:")
        for pattern in result.failure_patterns:
            print(f"  • {pattern.pattern_type}: {pattern.prediction}")
    
    return result

if __name__ == "__main__":
    asyncio.run(main())