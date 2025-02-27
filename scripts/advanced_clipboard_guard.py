#!/usr/bin/env python3
"""
Advanced Clipboard Guard
Prevents specific text insertions and manages clipboard content
"""
import logging
import os
import re
import threading
import time

import pyperclip


    class AdvancedClipboardGuard:
        def __init__(self, log_file="/var/log/advanced_clipboard_guard.log"):
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        filename=log_file,
        filemode="a",
    )
    self.logger = logging.getLogger("AdvancedClipboardGuard")
    
    # Blocked patterns with more comprehensive matching
    self.blocked_patterns = [
    r"Now, I'll create a system health monitor to complement these components:",
    r"<function_calls>",
    r"<invoke name=\"edit_file\">",
    r"Create an ultra-comprehensive system health monitoring framework",
]

# Last known good clipboard content
self.last_good_content = ""

# Monitoring control
self.stop_monitoring = threading.Event()
self.monitoring_thread = None

    def is_suspicious_content(self, content):
    """
    Advanced content matching with regex and multiple strategies
    """
        if not content:
        return False
        
        # Check for exact pattern matches
            for pattern in self.blocked_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                return True
                
                return False
                
                    def sanitize_clipboard(self):
                    """
                    Sanitize clipboard content, preventing suspicious insertions
                    """
                        try:
                        current_content = pyperclip.paste()
                        
                            if self.is_suspicious_content(current_content):
                            self.logger.warning(
                            f"Blocked suspicious clipboard content: {current_content}",
                        )
                        
                        # Restore to last known good content or clear clipboard
                            if self.last_good_content:
                            pyperclip.copy(self.last_good_content)
                                else:
                                pyperclip.copy("")
                                    else:
                                    # Update last good content
                                    self.last_good_content = current_content
                                    
                                    except Exception:
                                    self.logger.error(f"Clipboard sanitization error: {e}")
                                    
                                        def start_monitoring(self):
                                        """
                                        Start continuous clipboard monitoring
                                        """
                                        
                                            def monitor_worker():
                                                while not self.stop_monitoring.is_set():
                                                    try:
                                                    self.sanitize_clipboard()
                                                    time.sleep(0.5)  # Check every half second
                                                    except Exception:
                                                    self.logger.error(f"Clipboard monitoring error: {e}")
                                                    time.sleep(1)
                                                    
                                                    self.monitoring_thread = threading.Thread(target=monitor_worker, daemon=True)
                                                    self.monitoring_thread.start()
                                                    self.logger.info(f"Advanced Clipboard Guard started")
                                                    
                                                        def stop_monitoring(self):
                                                        """
                                                        Stop clipboard monitoring
                                                        """
                                                        self.stop_monitoring.set()
                                                            if self.monitoring_thread:
                                                            self.monitoring_thread.join()
                                                            self.logger.info(f"Advanced Clipboard Guard stopped")
                                                            
                                                            
                                                                def main():
                                                                guard = AdvancedClipboardGuard()
                                                                guard.start_monitoring()
                                                                
                                                                    try:
                                                                    # Keep main thread alive
                                                                        while True:
                                                                        time.sleep(3600)
                                                                        except KeyboardInterrupt:
                                                                        guard.stop_monitoring()
                                                                        
                                                                        
                                                                            if __name__ == "__main__":
                                                                            main()
                                                                            