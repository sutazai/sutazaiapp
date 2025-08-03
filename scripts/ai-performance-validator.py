#!/usr/bin/env python3
"""
Purpose: Validate AI system performance across all 131 agents
Usage: python scripts/ai-performance-validator.py [--full-test]
Requirements: asyncio, httpx, json, statistics
"""

import os
import sys
import json
import time
import asyncio
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import httpx
import argparse
import logging
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.core.ollama_integration import OllamaConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Single test result"""
    agent_name: str
    model: str
    test_type: str
    success: bool
    response_time: float
    tokens_used: int
    error: Optional[str] = None
    response_quality: float = 0.0  # 0-1 score


@dataclass
class PerformanceMetrics:
    """Performance metrics for an agent"""
    agent_name: str
    model: str
    total_tests: int = 0
    successful_tests: int = 0
    failed_tests: int = 0
    avg_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    avg_tokens: float = 0.0
    avg_quality: float = 0.0
    error_types: Dict[str, int] = field(default_factory=dict)


class AIPerformanceValidator:
    """Validate AI system performance across all agents"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.test_results: List[TestResult] = []
        self.agent_metrics: Dict[str, PerformanceMetrics] = {}
        self.http_client = None
        
        # Test prompts for different complexity levels
        self.test_prompts = {
            'simple': {
                'prompt': 'What is 2+2?',
                'expected_keywords': ['4', 'four'],
                'max_tokens': 50
            },
            'medium': {
                'prompt': 'Write a Python function to reverse a string.',
                'expected_keywords': ['def', 'return', '[::-1]', 'reverse'],
                'max_tokens': 200
            },
            'complex': {
                'prompt': 'Design a distributed caching system with fault tolerance.',
                'expected_keywords': ['distributed', 'cache', 'fault', 'replication', 'consistency'],
                'max_tokens': 500
            }
        }
    
    async def setup(self):
        """Setup HTTP client"""
        self.http_client = httpx.AsyncClient(timeout=60.0)
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.http_client:
            await self.http_client.aclose()
    
    async def validate_ollama_connection(self) -> bool:
        """Validate Ollama is running and accessible"""
        try:
            response = await self.http_client.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                logger.info(f"Ollama connected. Available models: {len(models)}")
                return True
            else:
                logger.error(f"Ollama connection failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            return False
    
    async def test_model_availability(self) -> Dict[str, bool]:
        """Test if required models are available"""
        required_models = [
            OllamaConfig.DEFAULT_MODEL,
            OllamaConfig.SONNET_MODEL,
            OllamaConfig.OPUS_MODEL
        ]
        
        availability = {}
        
        try:
            response = await self.http_client.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                available_models = {
                    m.get('name', '').split(':')[0] 
                    for m in response.json().get('models', [])
                }
                
                for model in required_models:
                    model_name = model.split(':')[0]
                    availability[model] = model_name in available_models
                    
                    if not availability[model]:
                        logger.warning(f"Model {model} not available")
            
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            for model in required_models:
                availability[model] = False
        
        return availability
    
    async def test_agent_performance(self, 
                                   agent_name: str, 
                                   model: str,
                                   test_type: str = 'medium') -> TestResult:
        """Test a single agent's performance"""
        test_prompt = self.test_prompts[test_type]
        
        start_time = time.time()
        
        try:
            # Query the model
            response = await self.http_client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    'model': model,
                    'prompt': test_prompt['prompt'],
                    'stream': False,
                    'options': {
                        'temperature': 0.7,
                        'num_predict': test_prompt['max_tokens']
                    }
                }
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                
                # Estimate tokens (simple)
                tokens_used = len(response_text.split())
                
                # Evaluate response quality
                quality_score = self._evaluate_response_quality(
                    response_text, 
                    test_prompt['expected_keywords']
                )
                
                return TestResult(
                    agent_name=agent_name,
                    model=model,
                    test_type=test_type,
                    success=True,
                    response_time=response_time,
                    tokens_used=tokens_used,
                    response_quality=quality_score
                )
            else:
                return TestResult(
                    agent_name=agent_name,
                    model=model,
                    test_type=test_type,
                    success=False,
                    response_time=response_time,
                    tokens_used=0,
                    error=f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            return TestResult(
                agent_name=agent_name,
                model=model,
                test_type=test_type,
                success=False,
                response_time=time.time() - start_time,
                tokens_used=0,
                error=str(e)
            )
    
    def _evaluate_response_quality(self, response: str, expected_keywords: List[str]) -> float:
        """Evaluate response quality based on expected keywords"""
        if not response:
            return 0.0
        
        response_lower = response.lower()
        keywords_found = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)
        
        # Basic quality score
        keyword_score = keywords_found / len(expected_keywords) if expected_keywords else 0.5
        
        # Length appropriateness (not too short, not too long)
        word_count = len(response.split())
        if word_count < 5:
            length_score = 0.3
        elif word_count > 1000:
            length_score = 0.7
        else:
            length_score = 1.0
        
        # Combine scores
        return (keyword_score * 0.7 + length_score * 0.3)
    
    async def run_performance_tests(self, full_test: bool = False):
        """Run performance tests on all agents"""
        # Get all agents and their models
        agent_models = OllamaConfig.AGENT_MODELS
        
        # Filter for testing
        if not full_test:
            # Test a representative sample
            test_agents = {
                # One from each tier
                'garbage-collector': OllamaConfig.DEFAULT_MODEL,  # Simple
                'ai-agent-orchestrator': OllamaConfig.SONNET_MODEL,  # Medium
                'ai-system-architect': OllamaConfig.OPUS_MODEL  # Complex
            }
        else:
            test_agents = agent_models
        
        logger.info(f"Testing {len(test_agents)} agents...")
        
        # Run tests
        for agent_name, model in test_agents.items():
            logger.info(f"Testing {agent_name} with {model}")
            
            # Initialize metrics
            if agent_name not in self.agent_metrics:
                self.agent_metrics[agent_name] = PerformanceMetrics(
                    agent_name=agent_name,
                    model=model
                )
            
            # Determine test complexity based on model
            if model == OllamaConfig.DEFAULT_MODEL:
                test_types = ['simple']
            elif model == OllamaConfig.SONNET_MODEL:
                test_types = ['simple', 'medium']
            else:  # OPUS
                test_types = ['simple', 'medium', 'complex']
            
            # Run tests
            for test_type in test_types:
                result = await self.test_agent_performance(agent_name, model, test_type)
                self.test_results.append(result)
                self._update_metrics(agent_name, result)
                
                # Small delay between tests
                await asyncio.sleep(0.5)
    
    def _update_metrics(self, agent_name: str, result: TestResult):
        """Update agent metrics with test result"""
        metrics = self.agent_metrics[agent_name]
        
        metrics.total_tests += 1
        
        if result.success:
            metrics.successful_tests += 1
            
            # Update response time stats
            if metrics.avg_response_time == 0:
                metrics.avg_response_time = result.response_time
            else:
                metrics.avg_response_time = (
                    (metrics.avg_response_time * (metrics.successful_tests - 1) + result.response_time) /
                    metrics.successful_tests
                )
            
            metrics.min_response_time = min(metrics.min_response_time, result.response_time)
            metrics.max_response_time = max(metrics.max_response_time, result.response_time)
            
            # Update token stats
            if metrics.avg_tokens == 0:
                metrics.avg_tokens = result.tokens_used
            else:
                metrics.avg_tokens = (
                    (metrics.avg_tokens * (metrics.successful_tests - 1) + result.tokens_used) /
                    metrics.successful_tests
                )
            
            # Update quality stats
            if metrics.avg_quality == 0:
                metrics.avg_quality = result.response_quality
            else:
                metrics.avg_quality = (
                    (metrics.avg_quality * (metrics.successful_tests - 1) + result.response_quality) /
                    metrics.successful_tests
                )
        else:
            metrics.failed_tests += 1
            if result.error:
                error_type = result.error.split(':')[0]
                metrics.error_types[error_type] = metrics.error_types.get(error_type, 0) + 1
    
    async def test_concurrent_performance(self, concurrent_requests: int = 5):
        """Test system performance under concurrent load"""
        logger.info(f"Testing concurrent performance with {concurrent_requests} requests...")
        
        # Create concurrent tasks
        tasks = []
        agents = list(OllamaConfig.AGENT_MODELS.items())[:concurrent_requests]
        
        start_time = time.time()
        
        for agent_name, model in agents:
            task = self.test_agent_performance(agent_name, model, 'simple')
            tasks.append(task)
        
        # Run concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful = sum(1 for r in results if isinstance(r, TestResult) and r.success)
        
        logger.info(f"Concurrent test completed in {total_time:.2f}s")
        logger.info(f"Successful: {successful}/{concurrent_requests}")
        
        return {
            'total_time': total_time,
            'concurrent_requests': concurrent_requests,
            'successful': successful,
            'throughput': successful / total_time if total_time > 0 else 0
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate performance validation report"""
        # Overall statistics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        
        # Model-level statistics
        model_stats = defaultdict(lambda: {
            'total': 0, 'successful': 0, 'avg_response_time': 0, 
            'avg_quality': 0, 'response_times': []
        })
        
        for result in self.test_results:
            stats = model_stats[result.model]
            stats['total'] += 1
            if result.success:
                stats['successful'] += 1
                stats['response_times'].append(result.response_time)
        
        # Calculate averages
        for model, stats in model_stats.items():
            if stats['response_times']:
                stats['avg_response_time'] = statistics.mean(stats['response_times'])
                stats['p95_response_time'] = statistics.quantiles(stats['response_times'], n=20)[18]  # 95th percentile
                del stats['response_times']  # Remove raw data from report
        
        # Agent performance summary
        agent_summary = []
        for agent_name, metrics in self.agent_metrics.items():
            success_rate = metrics.successful_tests / metrics.total_tests if metrics.total_tests > 0 else 0
            
            agent_summary.append({
                'agent': agent_name,
                'model': metrics.model,
                'success_rate': round(success_rate * 100, 1),
                'avg_response_time': round(metrics.avg_response_time, 2),
                'avg_quality': round(metrics.avg_quality, 2),
                'tests_run': metrics.total_tests
            })
        
        # Sort by success rate
        agent_summary.sort(key=lambda x: x['success_rate'], reverse=True)
        
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'overall_statistics': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': round((successful_tests / total_tests * 100) if total_tests > 0 else 0, 1),
                'unique_agents_tested': len(self.agent_metrics)
            },
            'model_performance': dict(model_stats),
            'agent_performance': agent_summary[:10],  # Top 10 agents
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check overall success rate
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        if success_rate < 0.9:
            recommendations.append(f"Success rate is {success_rate*100:.1f}%. Investigate failing agents.")
        
        # Check response times by model
        model_response_times = defaultdict(list)
        for result in self.test_results:
            if result.success:
                model_response_times[result.model].append(result.response_time)
        
        for model, times in model_response_times.items():
            avg_time = statistics.mean(times)
            if model == OllamaConfig.DEFAULT_MODEL and avg_time > 2:
                recommendations.append(f"TinyLlama average response time ({avg_time:.1f}s) is high. Check system load.")
            elif model == OllamaConfig.SONNET_MODEL and avg_time > 5:
                recommendations.append(f"Qwen2.5 average response time ({avg_time:.1f}s) is high. Consider optimization.")
            elif model == OllamaConfig.OPUS_MODEL and avg_time > 10:
                recommendations.append(f"DeepSeek average response time ({avg_time:.1f}s) is high. Review complex prompts.")
        
        # Check quality scores
        low_quality_agents = [
            (name, metrics.avg_quality) 
            for name, metrics in self.agent_metrics.items() 
            if metrics.avg_quality < 0.5 and metrics.successful_tests > 0
        ]
        
        if low_quality_agents:
            recommendations.append(f"{len(low_quality_agents)} agents have low quality scores. Review prompt engineering.")
        
        # Memory recommendations
        if any(m == OllamaConfig.OPUS_MODEL for m in model_response_times.keys()):
            recommendations.append("Consider quantization for DeepSeek model to reduce memory usage.")
        
        if not recommendations:
            recommendations.append("System is performing well within expected parameters.")
        
        return recommendations
    
    async def run_validation(self, full_test: bool = False):
        """Run complete validation suite"""
        await self.setup()
        
        try:
            # Check Ollama connection
            if not await self.validate_ollama_connection():
                logger.error("Cannot connect to Ollama. Exiting.")
                return None
            
            # Check model availability
            model_availability = await self.test_model_availability()
            logger.info(f"Model availability: {model_availability}")
            
            # Run performance tests
            await self.run_performance_tests(full_test)
            
            # Test concurrent performance
            if full_test:
                concurrent_results = await self.test_concurrent_performance()
                logger.info(f"Concurrent performance: {concurrent_results}")
            
            # Generate report
            report = self.generate_report()
            
            # Save report
            report_path = f"reports/ai_performance_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Validation report saved to: {report_path}")
            
            # Print summary
            print("\n=== AI System Performance Validation Summary ===")
            print(f"Total Tests: {report['overall_statistics']['total_tests']}")
            print(f"Success Rate: {report['overall_statistics']['success_rate']}%")
            print(f"Agents Tested: {report['overall_statistics']['unique_agents_tested']}")
            
            print("\nModel Performance:")
            for model, stats in report['model_performance'].items():
                print(f"  {model}:")
                print(f"    Success Rate: {(stats['successful']/stats['total']*100):.1f}%")
                print(f"    Avg Response Time: {stats['avg_response_time']:.2f}s")
            
            print("\nTop Performing Agents:")
            for agent in report['agent_performance'][:5]:
                print(f"  {agent['agent']} ({agent['model']}): {agent['success_rate']}% success, {agent['avg_response_time']}s avg")
            
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
            
            return report
            
        finally:
            await self.cleanup()


async def main():
    parser = argparse.ArgumentParser(description='Validate AI system performance')
    parser.add_argument('--full-test', action='store_true', help='Run full test on all agents')
    parser.add_argument('--ollama-url', default='http://localhost:11434', help='Ollama API URL')
    args = parser.parse_args()
    
    validator = AIPerformanceValidator(ollama_url=args.ollama_url)
    await validator.run_validation(full_test=args.full_test)


if __name__ == "__main__":
    asyncio.run(main())