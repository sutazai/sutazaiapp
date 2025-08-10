#!/usr/bin/env python3
"""
Compact System Monitor - Fixed Layout, No Scrolling
===================================================

A monitor that works perfectly in 25-row terminals with static display.
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import sys
import time
import psutil
import subprocess
from datetime import datetime


class CompactMonitor:
    """Compact monitor optimized for 25-row terminals"""
    
    def __init__(self):
        # Colors
        self.GREEN = '\033[92m'
        self.YELLOW = '\033[93m'
        self.RED = '\033[91m'
        self.RESET = '\033[0m'
        
        # Terminal control
        self.CLEAR = '\033[2J'
        self.HOME = '\033[H'
        self.HIDE_CURSOR = '\033[?25l'
        self.SHOW_CURSOR = '\033[?25h'
        
        # Clear screen once at startup
        print(self.CLEAR + self.HIDE_CURSOR, end='')
    
    def move_to(self, line):
        """Move cursor to specific line"""
        return f'\033[{line};1H'
    
    def clear_line(self):
        """Clear from cursor to end of line"""
        return '\033[K'
    
    def create_bar(self, percent, width=20):
        """Create progress bar"""
        filled = int((percent / 100) * width)
        bar = '█' * filled + '░' * (width - filled)
        return bar
    
    def get_color(self, value, warning, critical):
        """Get color based on thresholds"""
        if value >= critical:
            return self.RED
        elif value >= warning:
            return self.YELLOW
        else:
            return self.GREEN
    
    def get_system_stats(self):
        """Get system statistics"""
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_percent': cpu,
            'cpu_cores': psutil.cpu_count(),
            'mem_percent': memory.percent,
            'mem_used': memory.used / 1024 / 1024 / 1024,
            'mem_total': memory.total / 1024 / 1024 / 1024,
            'swap_percent': swap.percent,
            'swap_used': swap.used / 1024 / 1024,
            'swap_total': swap.total / 1024 / 1024 / 1024,
            'disk_percent': disk.percent,
            'disk_free': disk.free / 1024 / 1024 / 1024,
        }
    
    def get_docker_containers(self):
        """Get Docker container info"""
        try:
            result = subprocess.run(
                ['docker', 'ps', '--format', '{{.Names}}|{{.Status}}|{{.Image}}'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                containers = []
                running = 0
                for line in result.stdout.strip().split('\n'):
                    if '|' in line:
                        parts = line.split('|')
                        if len(parts) == 3:
                            name = parts[0][:18]
                            status = 'Up' in parts[1]
                            image = parts[2].split(':')[0].split('/')[-1][:18]
                            
                            if status:
                                icon = '🟢'
                                running += 1
                            else:
                                icon = '🔴'
                            
                            containers.append(f"{name:<18} {icon} {image}")
                
                return containers[:6], running, len(containers)
        except Exception as e:
            # Suppressed exception (was bare except)
            logger.debug(f"Suppressed exception: {e}")
            pass
        
        return ["No containers found"], 0, 0
    
    def get_ollama_models(self):
        """Get Ollama models"""
        try:
            result = subprocess.run(
                ['docker', 'exec', 'sutazai-ollama', 'ollama', 'list'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    models = []
                    for line in lines[1:]:
                        parts = line.split()
                        if parts:
                            model_name = parts[0]
                            models.append(f"  • {model_name}")
                    
                    return models[:3]
        except Exception as e:
            # Suppressed exception (was bare except)
            logger.debug(f"Suppressed exception: {e}")
            pass
        
        return ["  Unable to retrieve models"]
    
    def run(self):
        """Main monitor loop"""
        try:
            # Initial display
            print(self.HOME + "Initializing...", end='', flush=True)
            
            while True:
                # Get data
                stats = self.get_system_stats()
                containers, running, total = self.get_docker_containers()
                models = self.get_ollama_models()
                
                # Build display
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Line 1: Header
                print(f"{self.move_to(1)}🚀 SutazAI Monitor - {timestamp}{self.clear_line()}", end='')
                
                # Line 2: Separator
                print(f"{self.move_to(2)}{'=' * 60}{self.clear_line()}", end='')
                
                # Lines 3-6: System stats
                cpu_color = self.get_color(stats['cpu_percent'], 60, 80)
                mem_color = self.get_color(stats['mem_percent'], 75, 90)
                swap_color = self.get_color(stats['swap_percent'], 50, 80)
                disk_color = self.get_color(stats['disk_percent'], 80, 90)
                
                print(f"{self.move_to(3)}CPU:    {self.create_bar(stats['cpu_percent'])} {cpu_color}{stats['cpu_percent']:5.1f}%{self.RESET} ({stats['cpu_cores']} cores){self.clear_line()}", end='')
                print(f"{self.move_to(4)}Memory: {self.create_bar(stats['mem_percent'])} {mem_color}{stats['mem_percent']:5.1f}%{self.RESET} ({stats['mem_used']:.1f}GB/{stats['mem_total']:.1f}GB){self.clear_line()}", end='')
                print(f"{self.move_to(5)}Swap:   {self.create_bar(stats['swap_percent'])} {swap_color}{stats['swap_percent']:5.1f}%{self.RESET} ({stats['swap_used']:.1f}MB/{stats['swap_total']:.1f}GB){self.clear_line()}", end='')
                print(f"{self.move_to(6)}Disk:   {self.create_bar(stats['disk_percent'])} {disk_color}{stats['disk_percent']:5.1f}%{self.RESET} ({stats['disk_free']:.1f}GB free){self.clear_line()}", end='')
                
                # Line 7: Blank
                print(f"{self.move_to(7)}{self.clear_line()}", end='')
                
                # Line 8: Container header
                print(f"{self.move_to(8)}🐳 Containers ({running}/{total}):{self.clear_line()}", end='')
                
                # Lines 9-14: Containers (6 containers)
                for i in range(6):
                    if i < len(containers):
                        print(f"{self.move_to(9 + i)}{containers[i]}{self.clear_line()}", end='')
                    else:
                        print(f"{self.move_to(9 + i)}{self.clear_line()}", end='')
                
                # Line 15: Blank
                print(f"{self.move_to(15)}{self.clear_line()}", end='')
                
                # Line 16: Models header
                print(f"{self.move_to(16)}🤖 Models:{self.clear_line()}", end='')
                
                # Lines 17-19: Models (3 models)
                for i in range(3):
                    if i < len(models):
                        print(f"{self.move_to(17 + i)}{models[i]}{self.clear_line()}", end='')
                    else:
                        print(f"{self.move_to(17 + i)}{self.clear_line()}", end='')
                
                # Line 20: Blank
                print(f"{self.move_to(20)}{self.clear_line()}", end='')
                
                # Line 21: Alerts
                alert_msg = f"{self.GREEN}✅ All systems operating normally{self.RESET}"
                if stats['cpu_percent'] > 90:
                    alert_msg = f"{self.RED}⚠️  High CPU usage: {stats['cpu_percent']:.1f}%{self.RESET}"
                elif stats['mem_percent'] > 90:
                    alert_msg = f"{self.RED}⚠️  High memory usage: {stats['mem_percent']:.1f}%{self.RESET}"
                elif stats['disk_percent'] > 90:
                    alert_msg = f"{self.RED}⚠️  Low disk space: {stats['disk_percent']:.1f}% used{self.RESET}"
                
                print(f"{self.move_to(21)}🎯 Alerts: {alert_msg}{self.clear_line()}", end='')
                
                # Line 22: Blank
                print(f"{self.move_to(22)}{self.clear_line()}", end='')
                
                # Line 23: Blank  
                print(f"{self.move_to(23)}{self.clear_line()}", end='')
                
                # Line 24: Footer
                print(f"{self.move_to(24)}Press Ctrl+C to exit | Updates: 2s | v1.0{self.clear_line()}", end='')
                
                # Flush output
                sys.stdout.flush()
                
                # Wait 2 seconds
                time.sleep(2.0)
                
        except KeyboardInterrupt:
            # Clean exit
            print(self.SHOW_CURSOR)
            print(f"{self.move_to(25)}\nMonitor stopped.{self.clear_line()}")


def main():
    """Entry point"""
    # Check if terminal supports ANSI
    if not sys.stdout.isatty() and '--force' not in sys.argv:
        print("Error: This monitor requires an interactive terminal.")
        print("For testing purposes, use --force flag.")
        return 1
    
    monitor = CompactMonitor()
    monitor.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())