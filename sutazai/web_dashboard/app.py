import os
import json
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

from sutazai.auto_remediation import AutoRemediationManager
from sutazai.config_manager import ConfigurationManager
from sutazai.security_scanner import SecurityScanner

class RemediationDashboard:
    """
    Web dashboard for monitoring and managing SutazAI remediation activities.
    """

    def __init__(
        self, 
        project_root: str = '.', 
        log_dir: str = 'logs/remediation'
    ):
        """
        Initialize the remediation dashboard.

        Args:
            project_root (str): Root directory of the project
            log_dir (str): Directory to store remediation logs
        """
        self.project_root = os.path.abspath(project_root)
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger('RemediationDashboard')

        # Initialize Flask app and SocketIO
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)

        # Configure routes
        self._configure_routes()

    def _configure_routes(self):
        """
        Configure Flask routes for the dashboard.
        """
        @self.app.route('/')
        def index():
            """
            Render the main dashboard page.
            """
            return render_template('index.html')

        @self.app.route('/api/remediation_reports')
        def get_remediation_reports():
            """
            Get a list of recent remediation reports.
            """
            reports = self._get_recent_reports()
            return jsonify(reports)

        @self.app.route('/api/security_scan')
        def trigger_security_scan():
            """
            Trigger a manual security scan and remediation.
            """
            try:
                security_scanner = SecurityScanner(project_root=self.project_root)
                security_report = security_scanner.generate_security_report()

                remediation_manager = AutoRemediationManager(project_root=self.project_root)
                results = remediation_manager.auto_remediate()

                return jsonify({
                    'status': 'success',
                    'vulnerability_count': security_report.vulnerability_count,
                    'remediation_results': {
                        k: len(v) for k, v in results.items()
                    }
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500

        @self.socketio.on('connect')
        def handle_connect():
            """
            Handle WebSocket connection for real-time updates.
            """
            self.logger.info("Client connected to dashboard")
            emit('dashboard_update', {'message': 'Connected to SutazAI Remediation Dashboard'})

    def _get_recent_reports(
        self, 
        days: int = 7, 
        max_reports: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent remediation reports.

        Args:
            days (int): Number of days to look back
            max_reports (int): Maximum number of reports to return

        Returns:
            List[Dict[str, Any]]: List of recent remediation reports
        """
        reports = []
        cutoff_date = datetime.now() - timedelta(days=days)

        for filename in sorted(os.listdir(self.project_root), reverse=True):
            if filename.startswith('remediation_report_') and filename.endswith('.json'):
                file_path = os.path.join(self.project_root, filename)
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_mtime >= cutoff_date:
                    try:
                        with open(file_path, 'r') as f:
                            report_data = json.load(f)
                            report_data['filename'] = filename
                            report_data['timestamp'] = file_mtime.isoformat()
                            reports.append(report_data)
                    except Exception as e:
                        self.logger.warning(f"Could not read report {filename}: {e}")

                if len(reports) >= max_reports:
                    break

        return reports

    def run(
        self, 
        host: str = '0.0.0.0', 
        port: int = 5000, 
        debug: bool = False
    ):
        """
        Run the dashboard web server.

        Args:
            host (str): Host to bind the server
            port (int): Port to run the server
            debug (bool): Enable debug mode
        """
        self.logger.info(f"Starting SutazAI Remediation Dashboard on {host}:{port}")
        self.socketio.run(
            self.app, 
            host=host, 
            port=port, 
            debug=debug
        )

def main():
    dashboard = RemediationDashboard()
    dashboard.run(debug=True)

if __name__ == '__main__':
    main() 