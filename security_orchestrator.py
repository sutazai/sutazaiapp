#!/usr/bin/env python3
"""
Security Orchestrator for SutazAI System
Master orchestrator for all security components and monitoring
"""

import subprocess
import threading
import time
import json
import signal
import sys
from datetime import datetime
from pathlib import Path
import logging

class SecurityOrchestrator:
    def __init__(self):
        self.components = {
            'intrusion_detection': {
                'script': 'intrusion_detection_system.py',
                'process': None,
                'enabled': True,
                'description': 'Real-time intrusion detection and threat monitoring'
            },
            'security_logging': {
                'script': 'security_event_logger.py',
                'process': None,
                'enabled': True,
                'description': 'Centralized security event logging and correlation'
            },
            'threat_response': {
                'script': 'automated_threat_response.py',
                'process': None,
                'enabled': True,
                'description': 'Automated threat response and incident handling'
            },
            'dashboard': {
                'script': 'start_security_dashboard.sh',
                'process': None,
                'enabled': True,
                'description': 'Security monitoring dashboard (Streamlit)'
            }
        }
        
        self.running = False
        self.setup_logging()
        self.setup_signal_handlers()
    
    def setup_logging(self):
        """Setup logging for the orchestrator"""
        log_dir = Path('/opt/sutazaiapp/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler('/opt/sutazaiapp/logs/security_orchestrator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop_all_components()
        sys.exit(0)
    
    def start_component(self, component_name, component_config):
        """Start a security component"""
        if not component_config['enabled']:
            self.logger.info(f"Component {component_name} is disabled, skipping...")
            return False
        
        script_path = Path('/opt/sutazaiapp') / component_config['script']
        
        if not script_path.exists():
            self.logger.error(f"Script not found: {script_path}")
            return False
        
        try:
            self.logger.info(f"Starting {component_name}: {component_config['description']}")
            
            if script_path.suffix == '.sh':
                # Shell script
                process = subprocess.Popen(
                    ['bash', str(script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd='/opt/sutazaiapp'
                )
            else:
                # Python script
                process = subprocess.Popen(
                    ['python3', str(script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd='/opt/sutazaiapp'
                )
            
            component_config['process'] = process
            self.logger.info(f"Successfully started {component_name} (PID: {process.pid})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start {component_name}: {e}")
            return False
    
    def stop_component(self, component_name, component_config):
        """Stop a security component"""
        process = component_config.get('process')
        if process and process.poll() is None:
            self.logger.info(f"Stopping {component_name}...")
            try:
                process.terminate()
                process.wait(timeout=10)
                self.logger.info(f"Successfully stopped {component_name}")
            except subprocess.TimeoutExpired:
                self.logger.warning(f"Force killing {component_name}...")
                process.kill()
                process.wait()
            except Exception as e:
                self.logger.error(f"Error stopping {component_name}: {e}")
        
        component_config['process'] = None
    
    def check_component_health(self, component_name, component_config):
        """Check if a component is running and healthy"""
        process = component_config.get('process')
        if process and process.poll() is None:
            return True
        return False
    
    def restart_component(self, component_name):
        """Restart a specific component"""
        self.logger.info(f"Restarting {component_name}...")
        component_config = self.components[component_name]
        self.stop_component(component_name, component_config)
        time.sleep(2)  # Wait a bit before restarting
        return self.start_component(component_name, component_config)
    
    def monitor_components(self):
        """Monitor all components and restart if necessary"""
        self.logger.info("Starting component health monitoring...")
        
        while self.running:
            try:
                for component_name, component_config in self.components.items():
                    if component_config['enabled']:
                        if not self.check_component_health(component_name, component_config):
                            self.logger.warning(f"Component {component_name} is not healthy, restarting...")
                            self.restart_component(component_name)
                
                # Wait before next health check
                time.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Error in component monitoring: {e}")
                time.sleep(60)
    
    def generate_status_report(self):
        """Generate status report of all components"""
        status_report = {
            'timestamp': datetime.now().isoformat(),
            'orchestrator_status': 'running' if self.running else 'stopped',
            'components': {}
        }
        
        for component_name, component_config in self.components.items():
            status_report['components'][component_name] = {
                'enabled': component_config['enabled'],
                'running': self.check_component_health(component_name, component_config),
                'description': component_config['description'],
                'pid': component_config.get('process', {}).pid if component_config.get('process') else None
            }
        
        return status_report
    
    def save_status_report(self):
        """Save status report to file"""
        report = self.generate_status_report()
        status_file = Path('/opt/sutazaiapp/data/security_orchestrator_status.json')
        status_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(status_file, 'w') as f:
            json.dump(report, f, indent=2)
    
    def run_security_assessments(self):
        """Run periodic security assessments"""
        self.logger.info("Running initial security assessments...")
        
        assessments = [
            ('security_pentest_scanner.py', 'Penetration Testing'),
            ('container_security_auditor.py', 'Container Security Audit'),
            ('network_security_analyzer.py', 'Network Security Analysis'),
            ('auth_security_tester.py', 'Authentication Security Testing')
        ]
        
        for script, description in assessments:
            script_path = Path('/opt/sutazaiapp') / script
            if script_path.exists():
                self.logger.info(f"Running {description}...")
                try:
                    result = subprocess.run(
                        ['python3', str(script_path)],
                        capture_output=True,
                        text=True,
                        timeout=300,  # 5 minute timeout
                        cwd='/opt/sutazaiapp'
                    )
                    if result.returncode == 0:
                        self.logger.info(f"Successfully completed {description}")
                    else:
                        self.logger.warning(f"{description} completed with warnings")
                
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"{description} timed out")
                except Exception as e:
                    self.logger.error(f"Error running {description}: {e}")
            else:
                self.logger.warning(f"Assessment script not found: {script}")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive security report"""
        self.logger.info("Generating comprehensive security report...")
        
        report_script = Path('/opt/sutazaiapp/comprehensive_security_report_generator.py')
        if report_script.exists():
            try:
                result = subprocess.run(
                    ['python3', str(report_script)],
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout
                    cwd='/opt/sutazaiapp'
                )
                if result.returncode == 0:
                    self.logger.info("Successfully generated comprehensive security report")
                else:
                    self.logger.warning("Security report generation completed with warnings")
                    
            except subprocess.TimeoutExpired:
                self.logger.warning("Security report generation timed out")
            except Exception as e:
                self.logger.error(f"Error generating security report: {e}")
        else:
            self.logger.error("Security report generator not found")
    
    def start_all_components(self):
        """Start all enabled security components"""
        self.logger.info("Starting all security components...")
        
        started_components = 0
        for component_name, component_config in self.components.items():
            if self.start_component(component_name, component_config):
                started_components += 1
                time.sleep(2)  # Stagger startup
        
        self.logger.info(f"Successfully started {started_components}/{len(self.components)} components")
        return started_components
    
    def stop_all_components(self):
        """Stop all running security components"""
        self.logger.info("Stopping all security components...")
        
        for component_name, component_config in self.components.items():
            self.stop_component(component_name, component_config)
        
        self.logger.info("All components stopped")
    
    def start_orchestrator(self):
        """Start the security orchestrator"""
        print("=" * 60)
        print("SutazAI Security Orchestrator")
        print("=" * 60)
        
        self.running = True
        
        # Run initial security assessments
        self.run_security_assessments()
        
        # Generate initial comprehensive report
        self.generate_comprehensive_report()
        
        # Start all security components
        started = self.start_all_components()
        
        if started == 0:
            self.logger.error("No components started successfully. Exiting...")
            return
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_components, daemon=True)
        monitor_thread.start()
        
        self.logger.info("Security orchestrator is now running")
        self.logger.info("Press Ctrl+C to stop the orchestrator")
        
        # Main loop
        try:
            while self.running:
                # Save status report
                self.save_status_report()
                
                # Generate periodic reports (every 6 hours)
                current_hour = datetime.now().hour
                if current_hour in [0, 6, 12, 18] and datetime.now().minute < 5:
                    self.generate_comprehensive_report()
                
                time.sleep(300)  # Sleep for 5 minutes
                
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down...")
        finally:
            self.stop_all_components()
            self.running = False
    
    def show_status(self):
        """Show current status of all components"""
        report = self.generate_status_report()
        
        print("\n" + "=" * 60)
        print("SECURITY ORCHESTRATOR STATUS")
        print("=" * 60)
        print(f"Orchestrator Status: {report['orchestrator_status'].upper()}")
        print(f"Last Updated: {report['timestamp']}")
        print("\nComponent Status:")
        print("-" * 60)
        
        for component_name, status in report['components'].items():
            status_symbol = "🟢" if status['running'] else "🔴"
            enabled_symbol = "✅" if status['enabled'] else "❌"
            pid_info = f"(PID: {status['pid']})" if status['pid'] else ""
            
            print(f"{status_symbol} {enabled_symbol} {component_name.upper():<20} {pid_info}")
            print(f"   {status['description']}")
            print()

def main():
    orchestrator = SecurityOrchestrator()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'status':
            orchestrator.show_status()
        elif command == 'start':
            orchestrator.start_orchestrator()
        elif command == 'stop':
            print("Stopping security orchestrator...")
            orchestrator.stop_all_components()
        elif command == 'restart':
            print("Restarting security orchestrator...")
            orchestrator.stop_all_components()
            time.sleep(2)
            orchestrator.start_orchestrator()
        elif command == 'report':
            orchestrator.generate_comprehensive_report()
        elif command == 'assess':
            orchestrator.run_security_assessments()
        else:
            print("Usage: python3 security_orchestrator.py [start|stop|restart|status|report|assess]")
    else:
        # Default action is to start
        orchestrator.start_orchestrator()

if __name__ == "__main__":
    main()