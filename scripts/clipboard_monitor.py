#!/usr/bin/env python3
import logging
import threading
import time

import pyperclip


    class ClipboardGuard:
        def __init__(self, log_file="/var/log/clipboard_guard.log"):
        logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        filename=log_file,
    )
    self.logger = logging.getLogger("ClipboardGuard")
    self.last_known_content = ""
    self.blocked_patterns = [
    "Now, I'll create a system health monitor to complement these components:",
    "<function_calls>",
    '<invoke name="edit_file">',
]
self.monitoring_thread = None
self.stop_monitoring = threading.Event()

    def is_suspicious_content(self, content):
    """Check if content matches any blocked patterns"""
    return any(pattern in content for pattern in self.blocked_patterns)
    
        def monitor_clipboard(self):
        """Continuously monitor clipboard and prevent suspicious insertions"""
            while not self.stop_monitoring.is_set():
                try:
                current_content = pyperclip.paste()
                
                    if current_content and self.is_suspicious_content(current_content):
                    self.logger.warning(
                    f"Blocked suspicious clipboard content: {current_content}",
                )
                # Reset clipboard to last known good content
                    if self.last_known_content:
                    pyperclip.copy(self.last_known_content)
                        else:
                        # Update last known good content
                        self.last_known_content = current_content
                        
                        time.sleep(0.5)  # Check every half second
                        except Exception:
                        self.logger.error(f"Clipboard monitoring error: {e}")
                        time.sleep(1)
                        
                            def start(self):
                            """Start clipboard monitoring"""
                            self.monitoring_thread = threading.Thread(
                            target=self.monitor_clipboard,
                            daemon=True,
                        )
                        self.monitoring_thread.start()
                        self.logger.info(f"Clipboard Guard started")
                        
                            def stop(self):
                            """Stop clipboard monitoring"""
                            self.stop_monitoring.set()
                                if self.monitoring_thread:
                                self.monitoring_thread.join()
                                self.logger.info(f"Clipboard Guard stopped")
                                
                                
                                    def main():
                                    guard = ClipboardGuard()
                                    guard.start()
                                    
                                        try:
                                        # Keep main thread alive
                                            while True:
                                            time.sleep(3600)
                                            except KeyboardInterrupt:
                                            guard.stop()
                                            
                                            
                                                if __name__ == "__main__":
                                                main()
                                                