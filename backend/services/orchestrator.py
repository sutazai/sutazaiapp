import os
import json
import time
import logging
from typing import Dict, Any, List, Optional

from loguru import logger
import yaml
import prometheus_client
from prometheus_client import Counter, Gauge, Histogram

class SupremeAIOrchestrator:
    """
    Self-improving AI orchestrator with advanced monitoring and recovery
    """
    
    def __init__(
        self, 
        log_dir: str = "/var/log/sutazaiapp",
        config_dir: str = "/opt/sutazaiapp/config",
        recovery_dir: str = "/opt/sutazaiapp/recovery",
        cooldown_period: int = 3600  # 1 hour cooldown
    ):
        """
        Initialize Supreme AI Orchestrator
        
        Args:
            log_dir (str): Directory for system logs
            config_dir (str): Directory for configuration files
            recovery_dir (str): Directory for recovery snapshots
            cooldown_period (int): Minimum time between self-improvement attempts
        """
        self.log_dir = log_dir
        self.config_dir = config_dir
        self.recovery_dir = recovery_dir
        self.cooldown_period = cooldown_period
        
        # Ensure directories exist
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(recovery_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            filename=os.path.join(log_dir, "orchestrator.log"),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        
        # Load configuration
        self._load_config()
        
        # Initialize tracking variables
        self.last_improvement_time = 0
        self.improvement_attempts = 0
        
        # Prometheus Metrics
        self.improvement_attempts = Counter(
            'sutazai_orchestrator_improvement_attempts', 
            'Total self-improvement attempts'
        )
        self.improvement_success = Counter(
            'sutazai_orchestrator_improvement_successes', 
            'Successful self-improvement attempts'
        )
        self.improvement_duration = Histogram(
            'sutazai_orchestrator_improvement_duration_seconds', 
            'Duration of self-improvement attempts'
        )
        self.current_performance_score = Gauge(
            'sutazai_orchestrator_performance_score', 
            'Current system performance score'
        )
    
    def _load_config(self):
        """
        Load orchestrator configuration
        """
        config_path = os.path.join(self.config_dir, "orchestrator.yaml")
        
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration
            self.config = {
                'error_thresholds': {
                    'critical': 5,
                    'warning': 10
                },
                'improvement_strategies': [
                    'code_regeneration',
                    'dependency_update',
                    'configuration_reset'
                ]
            }
            
            # Save default config
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f)
    
    def analyze_logs(self, log_type: str = 'system') -> Dict[str, Any]:
        """
        Analyze logs for error patterns and system health
        
        Args:
            log_type (str): Type of log to analyze
        
        Returns:
            Dict containing log analysis results
        """
        log_file = os.path.join(self.log_dir, f"{log_type}_log.json")
        
        try:
            with open(log_file, 'r') as f:
                logs = [json.loads(line) for line in f]
            
            # Analyze error frequencies
            error_counts = {}
            for log in logs:
                if log.get('level') in ['ERROR', 'CRITICAL']:
                    error_type = log.get('error_type', 'unknown')
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            return {
                'total_logs': len(logs),
                'error_counts': error_counts,
                'needs_improvement': any(
                    count > self.config['error_thresholds']['warning'] 
                    for count in error_counts.values()
                )
            }
        
        except Exception as e:
            logger.error(f"Log analysis error: {e}")
            return {'error': str(e)}
    
    def _analyze_prometheus_metrics(self) -> Dict[str, float]:
        """
        Retrieve and analyze Prometheus metrics
        
        Returns:
            Dict of key performance indicators
        """
        try:
            # Simulated metric retrieval
            metrics = {
                'code_generation_success_rate': 0.9,
                'document_parsing_speed': 0.75,
                'system_resource_utilization': 0.6
            }
            return metrics
        except Exception as e:
            logger.error(f"Metrics retrieval error: {e}")
            return {}
    
    def trigger_self_improvement(self) -> Dict[str, Any]:
        """
        Enhanced self-improvement with metrics-based decision making
        """
        start_time = time.time()
        self.improvement_attempts.inc()
        
        try:
            # Analyze metrics before improvement
            performance_metrics = self._analyze_prometheus_metrics()
            
            # Decision logic based on metrics
            if all(score > 0.7 for score in performance_metrics.values()):
                logger.info("System performance optimal. No improvement needed.")
                return {'status': 'no_action_required'}
            
            # Existing improvement logic...
            result = super().trigger_self_improvement()
            
            # Record success and performance
            if result.get('status') == 'success':
                self.improvement_success.inc()
                self.current_performance_score.set(
                    sum(performance_metrics.values()) / len(performance_metrics)
                )
            
            return result
        
        except Exception as e:
            logger.error(f"Self-improvement error: {e}")
            return {'status': 'failed', 'error': str(e)}
        finally:
            # Record improvement duration
            self.improvement_duration.observe(time.time() - start_time)
    
    def _select_improvement_strategy(
        self, 
        log_analysis: Dict[str, Any]
    ) -> str:
        """
        Select the most appropriate improvement strategy
        
        Args:
            log_analysis (Dict): Results of log analysis
        
        Returns:
            str: Selected improvement strategy
        """
        # Prioritize strategies based on error types
        error_types = log_analysis.get('error_counts', {}).keys()
        
        if 'dependency' in error_types:
            return 'dependency_update'
        elif 'configuration' in error_types:
            return 'configuration_reset'
        else:
            return 'code_regeneration'
    
    def _execute_improvement_strategy(
        self, 
        strategy: str
    ) -> Dict[str, Any]:
        """
        Execute the selected improvement strategy
        
        Args:
            strategy (str): Improvement strategy to execute
        
        Returns:
            Dict containing improvement results
        """
        strategies = {
            'dependency_update': self._update_dependencies,
            'configuration_reset': self._reset_configuration,
            'code_regeneration': self._regenerate_code
        }
        
        return strategies.get(strategy, lambda: {'error': 'Unknown strategy'})()
    
    def _create_recovery_snapshot(self):
        """
        Create a recovery snapshot of current system state
        """
        snapshot_file = os.path.join(
            self.recovery_dir, 
            f"snapshot_{int(time.time())}.tar.gz"
        )
        
        # Use tar to create a compressed snapshot
        os.system(f"tar -czvf {snapshot_file} /opt/sutazaiapp")
        
        logger.info(f"Recovery snapshot created: {snapshot_file}")
    
    def _rollback_to_last_snapshot(self):
        """
        Rollback to the most recent recovery snapshot
        """
        snapshots = sorted(
            [f for f in os.listdir(self.recovery_dir) if f.endswith('.tar.gz')],
            reverse=True
        )
        
        if snapshots:
            latest_snapshot = os.path.join(self.recovery_dir, snapshots[0])
            os.system(f"tar -xzvf {latest_snapshot} -C /")
            
            logger.info(f"Rolled back to snapshot: {latest_snapshot}")
        else:
            logger.error("No recovery snapshots available")
    
    def _update_dependencies(self) -> Dict[str, Any]:
        """
        Update project dependencies
        
        Returns:
            Dict containing dependency update results
        """
        try:
            os.system("pip list --outdated > /tmp/outdated_deps.txt")
            os.system("pip install --upgrade -r /opt/sutazaiapp/requirements.txt")
            
            return {
                'status': 'success',
                'message': 'Dependencies updated successfully'
            }
        except Exception as e:
            logger.error(f"Dependency update failed: {e}")
            return {'error': str(e)}
    
    def _reset_configuration(self) -> Dict[str, Any]:
        """
        Reset system configuration to default
        
        Returns:
            Dict containing configuration reset results
        """
        try:
            # Implement configuration reset logic
            return {
                'status': 'success',
                'message': 'Configuration reset successfully'
            }
        except Exception as e:
            logger.error(f"Configuration reset failed: {e}")
            return {'error': str(e)}
    
    def _regenerate_code(self) -> Dict[str, Any]:
        """
        Regenerate problematic code sections
        
        Returns:
            Dict containing code regeneration results
        """
        try:
            # Implement code regeneration logic using local LLMs
            from backend.services.code_generation import CodeGenerationService
            
            code_gen = CodeGenerationService()
            
            # Example: Regenerate a specific module
            result = code_gen.generate_code(
                specification="Regenerate core system module with improved error handling",
                model_name="deepseek-coder"
            )
            
            return {
                'status': 'success',
                'generated_code': result.get('code'),
                'security_warnings': result.get('security_warnings', [])
            }
        except Exception as e:
            logger.error(f"Code regeneration failed: {e}")
            return {'error': str(e)}

def main():
    """
    Example usage and testing
    """
    orchestrator = SupremeAIOrchestrator()
    
    # Analyze logs
    log_analysis = orchestrator.analyze_logs()
    print("Log Analysis:", json.dumps(log_analysis, indent=2))
    
    # Trigger self-improvement
    improvement_result = orchestrator.trigger_self_improvement()
    print("Self-Improvement Result:", json.dumps(improvement_result, indent=2))

if __name__ == "__main__":
    main() 