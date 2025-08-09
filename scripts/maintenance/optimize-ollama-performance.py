#!/usr/bin/env python3
"""
Purpose: Master Ollama Performance Optimization Orchestrator for SutazAI
Usage: Comprehensive optimization of AI model performance across 69 agents
Requirements: All optimization modules, asyncio, logging
"""

import os
import sys
import json
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

# Add the agents/core directory to the Python path
sys.path.append('/opt/sutazaiapp/agents/core')

try:
    from ollama_performance_optimizer import OllamaPerformanceOptimizer
    from ollama_batch_processor import OllamaBatchProcessor
    from ollama_context_optimizer import OllamaContextOptimizer
    from ollama_model_manager import OllamaModelManager
except ImportError as e:
    logging.error(f"Failed to import optimization modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/ollama_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ollama-master-optimizer')

class SutazAIPerformanceOrchestrator:
    """Master orchestrator for all Ollama performance optimizations"""
    
    def __init__(self):
        self.performance_optimizer = None
        self.batch_processor = None
        self.context_optimizer = None
        self.model_manager = None
        
        # Optimization results
        self.optimization_results = {
            'timestamp': datetime.now().isoformat(),
            'performance_optimization': None,
            'batch_processing': None,
            'context_optimization': None,
            'model_management': None,
            'overall_score': 0.0,
            'recommendations': []
        }
    
    async def initialize_components(self):
        """Initialize all optimization components"""
        logger.info("Initializing SutazAI Performance Orchestrator")
        
        try:
            # Initialize all components
            self.performance_optimizer = OllamaPerformanceOptimizer()
            self.batch_processor = OllamaBatchProcessor()
            self.context_optimizer = OllamaContextOptimizer()
            self.model_manager = OllamaModelManager()
            
            logger.info("All optimization components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False
    
    async def run_comprehensive_optimization(self) -> Dict[str, Any]:
        """Run comprehensive optimization across all systems"""
        logger.info("Starting comprehensive Ollama performance optimization")
        start_time = time.time()
        
        # Phase 1: Performance Analysis and Optimization
        logger.info("Phase 1: Performance Analysis and Optimization")
        try:
            await self.performance_optimizer.start_optimization()
            
            # Run benchmarks on all models
            benchmark_results = await self.performance_optimizer.run_comprehensive_benchmark()
            
            # Generate performance report
            performance_report = self.performance_optimizer.generate_performance_report()
            
            self.optimization_results['performance_optimization'] = {
                'benchmarks': benchmark_results,
                'report': performance_report,
                'status': 'completed'
            }
            
            logger.info("Phase 1 completed successfully")
            
        except Exception as e:
            logger.error(f"Phase 1 failed: {e}")
            self.optimization_results['performance_optimization'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # Phase 2: Batch Processing Optimization
        logger.info("Phase 2: Batch Processing Optimization")
        try:
            # Start batch processor
            await self.batch_processor.start()
            
            # Warm up cache with common prompts
            common_prompts = [
                ('tinyllama', 'Hello, how can I help you?'),
                ('tinyllama', 'What is artificial intelligence?'),
                ('tinyllama', 'Explain machine learning briefly.'),
                ('tinyllama', 'What are the benefits of AI?'),
                ('tinyllama', 'How does deep learning work?')
            ]
            
            await self.batch_processor.warm_cache(common_prompts)
            
            # Get batch processing statistics
            batch_stats = self.batch_processor.get_stats()
            
            self.optimization_results['batch_processing'] = {
                'stats': batch_stats,
                'cache_warmed': len(common_prompts),
                'status': 'completed'
            }
            
            logger.info("Phase 2 completed successfully")
            
        except Exception as e:
            logger.error(f"Phase 2 failed: {e}")
            self.optimization_results['batch_processing'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # Phase 3: Context Window Optimization
        logger.info("Phase 3: Context Window Optimization")
        try:
            # Run full context optimization
            context_report = await self.context_optimizer.run_full_optimization()
            
            self.optimization_results['context_optimization'] = {
                'report': context_report,
                'status': 'completed'
            }
            
            logger.info("Phase 3 completed successfully")
            
        except Exception as e:
            logger.error(f"Phase 3 failed: {e}")
            self.optimization_results['context_optimization'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # Phase 4: Model Management and Benchmarking
        logger.info("Phase 4: Model Management and Benchmarking")
        try:
            # Discover all models
            discovered_models = await self.model_manager.discover_models()
            
            # Run comprehensive benchmarks
            benchmark_results = await self.model_manager.run_all_model_benchmarks('standard')
            
            # Generate performance report
            model_report = self.model_manager.generate_performance_report()
            
            self.optimization_results['model_management'] = {
                'discovered_models': len(discovered_models),
                'benchmark_results': len(benchmark_results),
                'report': model_report,
                'status': 'completed'
            }
            
            logger.info("Phase 4 completed successfully")
            
        except Exception as e:
            logger.error(f"Phase 4 failed: {e}")
            self.optimization_results['model_management'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # Phase 5: Generate Final Report and Recommendations
        logger.info("Phase 5: Generating Final Report and Recommendations")
        await self._generate_final_recommendations()
        
        # Calculate total optimization time
        total_time = time.time() - start_time
        self.optimization_results['optimization_time_seconds'] = total_time
        
        logger.info(f"Comprehensive optimization completed in {total_time:.2f} seconds")
        
        # Save final results
        await self._save_optimization_results()
        
        return self.optimization_results
    
    async def _generate_final_recommendations(self):
        """Generate final optimization recommendations"""
        recommendations = []
        overall_score = 0.0
        score_components = 0
        
        # Analyze performance optimization results
        if self.optimization_results['performance_optimization']['status'] == 'completed':
            score_components += 1
            overall_score += 0.8  # Base score for completion
            recommendations.append("✓ Performance optimization completed successfully")
        else:
            recommendations.append("✗ Performance optimization failed - investigate Ollama connectivity")
        
        # Analyze batch processing results
        if self.optimization_results['batch_processing']['status'] == 'completed':
            score_components += 1
            batch_stats = self.optimization_results['batch_processing'].get('stats', {})
            cache_hit_rate = batch_stats.get('cache_hit_rate', 0)
            
            if cache_hit_rate > 0.3:
                overall_score += 0.9
                recommendations.append(f"✓ Batch processing optimized with {cache_hit_rate*100:.1f}% cache hit rate")
            else:
                overall_score += 0.6
                recommendations.append(f"⚠ Batch processing active but low cache hit rate: {cache_hit_rate*100:.1f}%")
        else:
            recommendations.append("✗ Batch processing optimization failed - check Redis connectivity")
        
        # Analyze context optimization results
        if self.optimization_results['context_optimization']['status'] == 'completed':
            score_components += 1
            context_report = self.optimization_results['context_optimization'].get('report', {})
            memory_savings = context_report.get('potential_savings', {}).get('memory_mb', 0)
            
            if memory_savings > 100:
                overall_score += 0.9
                recommendations.append(f"✓ Context optimization completed with {memory_savings:.0f}MB potential savings")
            else:
                overall_score += 0.7
                recommendations.append("✓ Context optimization completed with minimal memory impact")
        else:
            recommendations.append("✗ Context optimization failed - check model accessibility")
        
        # Analyze model management results
        if self.optimization_results['model_management']['status'] == 'completed':
            score_components += 1
            model_count = self.optimization_results['model_management'].get('discovered_models', 0)
            benchmark_count = self.optimization_results['model_management'].get('benchmark_results', 0)
            
            if benchmark_count == model_count and model_count > 0:
                overall_score += 1.0
                recommendations.append(f"✓ Model management completed - {model_count} models benchmarked")
            elif benchmark_count > 0:
                overall_score += 0.8
                recommendations.append(f"⚠ Partial model benchmarking - {benchmark_count}/{model_count} models")
            else:
                overall_score += 0.4
                recommendations.append("⚠ Model discovery completed but benchmarking failed")
        else:
            recommendations.append("✗ Model management failed - check Ollama service status")
        
        # Calculate final score
        if score_components > 0:
            self.optimization_results['overall_score'] = overall_score / score_components
        else:
            self.optimization_results['overall_score'] = 0.0
        
        # Add system-level recommendations
        if self.optimization_results['overall_score'] > 0.8:
            recommendations.append("🎉 Excellent optimization results - system is well-tuned")
        elif self.optimization_results['overall_score'] > 0.6:
            recommendations.append("👍 Good optimization results - minor improvements possible")
        elif self.optimization_results['overall_score'] > 0.4:
            recommendations.append("⚠ Moderate optimization success - several issues need attention")
        else:
            recommendations.append("❌ Poor optimization results - major issues require investigation")
        
        # Add specific technical recommendations
        recommendations.extend([
            "💡 Consider running optimization during low-traffic periods",
            "💡 Monitor system resources during peak agent usage",
            "💡 Review agent-specific model assignments for optimal performance",
            "💡 Schedule regular benchmarking to track performance degradation"
        ])
        
        self.optimization_results['recommendations'] = recommendations
    
    async def _save_optimization_results(self):
        """Save comprehensive optimization results"""
        # Save detailed results
        results_path = f"/opt/sutazaiapp/logs/sutazai_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(results_path, 'w') as f:
                json.dump(self.optimization_results, f, indent=2, default=str)
            
            logger.info(f"Optimization results saved to: {results_path}")
            
            # Also save a summary report
            summary_path = f"/opt/sutazaiapp/logs/optimization_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(summary_path, 'w') as f:
                f.write("SutazAI Ollama Performance Optimization Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Optimization Date: {self.optimization_results['timestamp']}\n")
                f.write(f"Overall Score: {self.optimization_results['overall_score']:.2f}/1.0\n")
                f.write(f"Total Time: {self.optimization_results.get('optimization_time_seconds', 0):.2f} seconds\n\n")
                
                f.write("Recommendations:\n")
                for i, rec in enumerate(self.optimization_results['recommendations'], 1):
                    f.write(f"{i:2d}. {rec}\n")
                
                f.write("\nPhase Results:\n")
                for phase, result in self.optimization_results.items():
                    if isinstance(result, dict) and 'status' in result:
                        f.write(f"  {phase}: {result['status']}\n")
            
            logger.info(f"Optimization summary saved to: {summary_path}")
            
        except Exception as e:
            logger.error(f"Failed to save optimization results: {e}")
    
    async def run_quick_health_check(self) -> Dict[str, Any]:
        """Run quick health check of optimization systems"""
        logger.info("Running quick health check")
        
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'ollama_service': 'unknown',
            'redis_service': 'unknown',
            'models_available': 0,
            'optimization_systems': {},
            'overall_health': 'unknown'
        }
        
        try:
            # Check Ollama service
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get("http://localhost:10104/api/tags")
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    health_status['ollama_service'] = 'healthy'
                    health_status['models_available'] = len(models)
                else:
                    health_status['ollama_service'] = 'unhealthy'
        except Exception as e:
            health_status['ollama_service'] = f'error: {str(e)}'
        
        try:
            # Check Redis service
            import redis
            redis_client = redis.Redis(host='localhost', port=10001, db=0)
            redis_client.ping()
            health_status['redis_service'] = 'healthy'
        except Exception as e:
            health_status['redis_service'] = f'error: {str(e)}'
        
        # Check optimization system files
        optimization_files = [
            '/opt/sutazaiapp/agents/core/ollama_performance_optimizer.py',
            '/opt/sutazaiapp/agents/core/ollama_batch_processor.py',
            '/opt/sutazaiapp/agents/core/ollama_context_optimizer.py',
            '/opt/sutazaiapp/agents/core/ollama_model_manager.py',
            '/opt/sutazaiapp/config/ollama_performance_optimization.yaml'
        ]
        
        for file_path in optimization_files:
            file_name = os.path.basename(file_path)
            health_status['optimization_systems'][file_name] = 'present' if os.path.exists(file_path) else 'missing'
        
        # Determine overall health
        if (health_status['ollama_service'] == 'healthy' and 
            health_status['redis_service'] == 'healthy' and
            health_status['models_available'] > 0):
            health_status['overall_health'] = 'healthy'
        elif health_status['ollama_service'] == 'healthy':
            health_status['overall_health'] = 'degraded'
        else:
            health_status['overall_health'] = 'unhealthy'
        
        return health_status
    
    async def cleanup(self):
        """Cleanup all optimization components"""
        logger.info("Cleaning up optimization components")
        
        try:
            if self.performance_optimizer:
                await self.performance_optimizer.stop_optimization()
            
            if self.batch_processor:
                await self.batch_processor.stop()
            
            if self.context_optimizer:
                await self.context_optimizer.client.aclose()
            
            if self.model_manager:
                await self.model_manager.client.aclose()
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SutazAI Ollama Performance Master Optimizer')
    parser.add_argument('--full-optimization', action='store_true', 
                       help='Run comprehensive optimization across all systems')
    parser.add_argument('--health-check', action='store_true', 
                       help='Run quick health check')
    parser.add_argument('--performance-only', action='store_true', 
                       help='Run only performance optimization')
    parser.add_argument('--batch-only', action='store_true', 
                       help='Run only batch processing optimization')
    parser.add_argument('--context-only', action='store_true', 
                       help='Run only context optimization')
    parser.add_argument('--benchmark-only', action='store_true', 
                       help='Run only model benchmarking')
    
    args = parser.parse_args()
    
    orchestrator = SutazAIPerformanceOrchestrator()
    
    try:
        if not await orchestrator.initialize_components():
            logger.error("Failed to initialize optimization components")
            sys.exit(1)
        
        if args.full_optimization:
            logger.info("Starting full optimization suite...")
            results = await orchestrator.run_comprehensive_optimization()
            
            print("\n" + "=" * 60)
            print("SutazAI Ollama Optimization Results")
            print("=" * 60)
            print(f"Overall Score: {results['overall_score']:.2f}/1.0")
            print(f"Optimization Time: {results.get('optimization_time_seconds', 0):.2f} seconds")
            print("\nRecommendations:")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"{i:2d}. {rec}")
        
        elif args.health_check:
            health = await orchestrator.run_quick_health_check()
            print("\nSystem Health Check:")
            print(f"Overall Health: {health['overall_health']}")
            print(f"Ollama Service: {health['ollama_service']}")
            print(f"Redis Service: {health['redis_service']}")
            print(f"Available Models: {health['models_available']}")
        
        elif args.performance_only:
            logger.info("Running performance optimization only...")
            await orchestrator.performance_optimizer.start_optimization()
            results = await orchestrator.performance_optimizer.run_comprehensive_benchmark()
            print(f"Performance optimization completed: {len(results)} benchmarks")
        
        elif args.batch_only:
            logger.info("Running batch processing optimization only...")
            await orchestrator.batch_processor.start()
            # Add test requests or cache warming here
            stats = orchestrator.batch_processor.get_stats()
            print(f"Batch processing stats: {json.dumps(stats, indent=2)}")
        
        elif args.context_only:
            logger.info("Running context optimization only...")
            report = await orchestrator.context_optimizer.run_full_optimization()
            print(f"Context optimization completed with {len(report.get('recommendations', []))} recommendations")
        
        elif args.benchmark_only:
            logger.info("Running model benchmarking only...")
            results = await orchestrator.model_manager.run_all_model_benchmarks()
            print(f"Benchmarking completed for {len(results)} models")
        
        else:
            parser.print_help()
            print("\nFor full optimization of SutazAI's 69 agents, use: --full-optimization")
    
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(1)
    
    finally:
        await orchestrator.cleanup()

if __name__ == '__main__':
    # Ensure log directory exists
    os.makedirs('/opt/sutazaiapp/logs', exist_ok=True)
    
    # Run the main function
    asyncio.run(main())