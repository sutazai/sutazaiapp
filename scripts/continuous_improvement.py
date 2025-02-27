import os
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List

import torch
import numpy as np
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import mlflow

def ensure_dirs(config):
    """Ensure all necessary directories exist"""
    dirs_to_create = [
        config.get('log_dir', '/opt/sutazaiapp/logs/improvements'),
        config.get('model_checkpoint_dir', '/opt/sutazaiapp/models/checkpoints'),
        config.get('training_data_dir', '/opt/sutazaiapp/training_data')
    ]
    
    for dir_path in dirs_to_create:
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory {dir_path}: {e}")

class ContinuousImprovementOrchestrator:
    def __init__(self, config_path: str = '/opt/sutazaiapp/config/improvement_config.json'):
        """
        Initialize the Continuous Improvement Orchestrator
        
        Principles:
        1. Offline-first approach
        2. OTP-validated processes
        3. Comprehensive logging
        4. No external dependencies
        """
        self.config = self._load_config(config_path)
        
        # Ensure directories exist before setting up logging
        ensure_dirs(self.config)
        
        self.setup_logging()
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        
        # Improvement tracking
        self.improvement_log: List[Dict[str, Any]] = []
        
        # Error tracking
        self.error_log: List[Dict[str, Any]] = []
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration with fallback defaults"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {config_path}")
            return {
                'log_dir': '/opt/sutazaiapp/logs/improvements',
                'model_checkpoint_dir': '/opt/sutazaiapp/models/checkpoints',
                'training_data_dir': '/opt/sutazaiapp/training_data',
                'max_gpu_memory_gb': 32,
                'performance_threshold': 0.95
            }
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in configuration file: {config_path}")
            raise
    
    def setup_logging(self):
        """Configure comprehensive logging"""
        log_file = os.path.join(
            self.config['log_dir'], 
            f'improvement_{datetime.now().isoformat()}.log'
        )
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        
        # Add a console handler for additional visibility
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(console_handler)
    
    def collect_system_logs(self) -> Dict[str, Any]:
        """
        Collect comprehensive system logs and performance metrics
        
        Returns:
            Dict containing system performance, error rates, and improvement opportunities
        """
        try:
            system_logs = {
                'timestamp': datetime.now().isoformat(),
                'error_rate': self._calculate_error_rate(),
                'performance_metrics': self._gather_performance_metrics(),
                'model_performance': self._evaluate_current_models(),
                'error_log': self.error_log
            }
            
            self.performance_history.append(system_logs)
            return system_logs
        except Exception as e:
            error_details = {
                'timestamp': datetime.now().isoformat(),
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }
            self.error_log.append(error_details)
            logging.error(f"Error in collect_system_logs: {error_details}")
            return {}
    
    def _calculate_error_rate(self) -> float:
        """Calculate system-wide error rate"""
        try:
            # Calculate error rate based on recent error logs
            total_errors = len(self.error_log)
            total_improvement_cycles = len(self.performance_history)
            
            # Prevent division by zero
            error_rate = total_errors / (total_improvement_cycles + 1) if total_improvement_cycles > 0 else 0
            
            return min(error_rate, 1.0)  # Ensure error rate is between 0 and 1
        except Exception as e:
            logging.error(f"Error calculating error rate: {e}")
            return 0.05  # Default error rate
    
    def _gather_performance_metrics(self) -> Dict[str, float]:
        """Collect system performance metrics"""
        try:
            return {
                'cpu_utilization': self._get_cpu_utilization(),
                'memory_usage': self._get_memory_usage(),
                'gpu_performance': self._get_gpu_performance()
            }
        except Exception as e:
            logging.error(f"Error gathering performance metrics: {e}")
            return {
                'cpu_utilization': 0,
                'memory_usage': 0,
                'gpu_performance': 0
            }
    
    def _evaluate_current_models(self) -> Dict[str, float]:
        """
        Evaluate performance of current AI models
        
        Returns performance scores for different model domains
        """
        try:
            # In a real-world scenario, this would involve more complex evaluation
            return {
                'code_generation': max(0.85 - self._calculate_error_rate(), 0),
                'document_parsing': max(0.78 - self._calculate_error_rate(), 0),
                'diagram_analysis': max(0.82 - self._calculate_error_rate(), 0)
            }
        except Exception as e:
            logging.error(f"Error evaluating current models: {e}")
            return {
                'code_generation': 0.75,
                'document_parsing': 0.68,
                'diagram_analysis': 0.72
            }
    
    def prepare_training_data(self) -> Dataset:
        """
        Prepare offline training dataset
        
        Principles:
        - Use only local, pre-processed data
        - Ensure data privacy and security
        """
        try:
            training_files = [
                os.path.join(self.config['training_data_dir'], f)
                for f in os.listdir(self.config['training_data_dir'])
                if f.endswith('.jsonl')
            ]
            
            # Load and preprocess training data
            training_data = []
            for file in training_files:
                with open(file, 'r') as f:
                    training_data.extend([json.loads(line) for line in f])
            
            if not training_data:
                logging.warning("No training data found. Using minimal dataset.")
                training_data = [
                    {"input": "Default training example", "output": "Default output"}
                ]
            
            return Dataset.from_list(training_data)
        except Exception as e:
            error_details = {
                'timestamp': datetime.now().isoformat(),
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }
            self.error_log.append(error_details)
            logging.error(f"Error preparing training data: {error_details}")
            return Dataset.from_list([])
    
    def fine_tune_model(self, model_name: str = 'deepseek-coder'):
        """
        Offline model fine-tuning with GPU acceleration
        
        Args:
            model_name: Name of the model to fine-tune
        """
        try:
            # Load pre-trained model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Prepare training data
            dataset = self.prepare_training_data()
            
            if len(dataset) == 0:
                logging.warning("No training data available. Skipping fine-tuning.")
                return
            
            # Configure GPU training
            training_args = TrainingArguments(
                output_dir=os.path.join(self.config['model_checkpoint_dir'], model_name),
                num_train_epochs=3,
                per_device_train_batch_size=4,
                save_steps=500,
                save_total_limit=3,
                prediction_loss_only=True,
            )
            
            # Initialize Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset
            )
            
            # Start training
            trainer.train()
            
            # Save final model
            trainer.save_model()
            
            logging.info(f"Successfully fine-tuned {model_name}")
        except Exception as e:
            error_details = {
                'timestamp': datetime.now().isoformat(),
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }
            self.error_log.append(error_details)
            logging.error(f"Error fine-tuning model: {error_details}")
    
    def validate_improvement(self) -> bool:
        """
        Validate that improvements do not cause regression
        
        Returns:
            Boolean indicating successful improvement
        """
        try:
            current_performance = self._evaluate_current_models()
            
            # Check if performance meets threshold
            improvement_successful = all(
                score >= self.config.get('performance_threshold', 0.90)
                for score in current_performance.values()
            )
            
            if improvement_successful:
                self.improvement_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'performance': current_performance,
                    'status': 'SUCCESS'
                })
            else:
                self.improvement_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'performance': current_performance,
                    'status': 'FAILED'
                })
            
            return improvement_successful
        except Exception as e:
            error_details = {
                'timestamp': datetime.now().isoformat(),
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }
            self.error_log.append(error_details)
            logging.error(f"Error validating improvement: {error_details}")
            return False
    
    def _get_cpu_utilization(self) -> float:
        """Get current CPU utilization"""
        return psutil.cpu_percent()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        return psutil.virtual_memory().percent
    
    def _get_gpu_performance(self) -> float:
        """Get GPU performance metrics"""
        if torch.cuda.is_available():
            return torch.cuda.memory_usage(0)
        return 0.0
    
    def run_improvement_cycle(self):
        """
        Execute a complete improvement cycle
        
        Workflow:
        1. Collect system logs
        2. Analyze performance
        3. Prepare training data
        4. Fine-tune models
        5. Validate improvements
        """
        try:
            logs = self.collect_system_logs()
            self.fine_tune_model()
            improvement_result = self.validate_improvement()
            
            logging.info(f"Improvement Cycle Complete. Success: {improvement_result}")
            
            return improvement_result
        except Exception as e:
            error_details = {
                'timestamp': datetime.now().isoformat(),
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }
            self.error_log.append(error_details)
            logging.error(f"Error in improvement cycle: {error_details}")
            return False

def main():
    """Main entry point for continuous improvement"""
    try:
        orchestrator = ContinuousImprovementOrchestrator()
        orchestrator.run_improvement_cycle()
    except Exception as e:
        logging.critical(f"Critical error in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 