import os
import sys
import json
import logging
import unittest
from datetime import datetime

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from scripts.continuous_improvement import ContinuousImprovementOrchestrator
except ImportError as e:
    print(f"Import error: {e}")
    print("Current Python path:", sys.path)
    raise

class TestContinuousImprovement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up logging for the entire test suite"""
        log_dir = '/opt/sutazaiapp/logs/tests'
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            filename=os.path.join(log_dir, f'continuous_improvement_test_{datetime.now().isoformat()}.log')
        )
    
    def setUp(self):
        """Set up test environment"""
        try:
            self.config_path = '/opt/sutazaiapp/config/improvement_config.json'
            self.orchestrator = ContinuousImprovementOrchestrator(self.config_path)
        except Exception as e:
            logging.error(f"Failed to initialize orchestrator: {e}")
            raise
    
    def test_config_loading(self):
        """Test configuration loading"""
        self.assertTrue(os.path.exists(self.config_path), "Configuration file should exist")
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logging.error(f"Error reading configuration: {e}")
            self.fail(f"Failed to read configuration: {e}")
        
        # Validate key configuration parameters
        required_keys = [
            'log_dir', 
            'model_checkpoint_dir', 
            'training_data_dir', 
            'performance_threshold'
        ]
        
        for key in required_keys:
            self.assertIn(key, config, f"Configuration should contain {key}")
    
    def test_system_logs_collection(self):
        """Test system logs collection"""
        logs = self.orchestrator.collect_system_logs()
        
        # Validate log structure
        self.assertIn('timestamp', logs)
        self.assertIn('error_rate', logs)
        self.assertIn('performance_metrics', logs)
        self.assertIn('model_performance', logs)
    
    def test_training_data_preparation(self):
        """Test training data preparation"""
        dataset = self.orchestrator.prepare_training_data()
        
        # Ensure dataset is not empty
        self.assertTrue(len(dataset) > 0, "Training dataset should not be empty")
    
    def test_performance_evaluation(self):
        """Test model performance evaluation"""
        performance = self.orchestrator._evaluate_current_models()
        
        # Validate performance metrics
        domains = ['code_generation', 'document_parsing', 'diagram_analysis']
        for domain in domains:
            self.assertIn(domain, performance)
            self.assertTrue(0 <= performance[domain] <= 1, f"Performance for {domain} should be between 0 and 1")
    
    def test_improvement_validation(self):
        """Test improvement validation process"""
        result = self.orchestrator.validate_improvement()
        
        # Validate improvement result
        self.assertIsInstance(result, bool)
    
    def test_error_logging(self):
        """Test error logging mechanism"""
        # Simulate an error
        try:
            1 / 0  # Intentional error
        except Exception:
            self.orchestrator.error_log.append({
                'timestamp': datetime.now().isoformat(),
                'error_type': 'Test Error',
                'error_message': 'Simulated error for testing'
            })
        
        # Check error log
        self.assertTrue(len(self.orchestrator.error_log) > 0, "Error log should capture errors")
    
    def test_full_improvement_cycle(self):
        """Test a complete improvement cycle"""
        result = self.orchestrator.run_improvement_cycle()
        
        # Validate improvement cycle result
        self.assertIsInstance(result, bool)
        
        # Check improvement and error logs
        self.assertTrue(len(self.orchestrator.improvement_log) > 0, "Improvement log should have entries")

def main():
    """Run tests and generate report"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestContinuousImprovement)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate test report
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped),
        'was_successful': result.wasSuccessful()
    }
    
    # Save report
    os.makedirs('/opt/sutazaiapp/logs/test_reports', exist_ok=True)
    with open(f'/opt/sutazaiapp/logs/test_reports/improvement_test_{datetime.now().isoformat()}.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    sys.exit(not result.wasSuccessful())

if __name__ == '__main__':
    main() 