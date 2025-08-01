#!/usr/bin/env python3
"""
Memory Monitor Dashboard for Small Model Systems
Real-time monitoring of memory usage and small model optimization
"""

import os
import sys
import time
import json
import psutil
import requests
from datetime import datetime
import threading
from collections import deque

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_color_code(percentage, warning=80, critical=90):
    """Get color code based on percentage"""
    if percentage >= critical:
        return '\033[91m'  # Red
    elif percentage >= warning:
        return '\033[93m'  # Yellow
    else:
        return '\033[92m'  # Green

def reset_color():
    """Reset terminal color"""
    return '\033[0m'

class MemoryMonitorDashboard:
    def __init__(self):
        """Initialize the monitoring dashboard"""
        self.hardware_optimizer_url = "http://localhost:8523"
        self.ollama_url = "http://localhost:11434"
        self.running = True
        
        # History for trends (last 60 measurements)
        self.memory_history = deque(maxlen=60)
        self.cpu_history = deque(maxlen=60)
        
        # Small model configuration
        self.small_models = ['qwen2.5:3b', 'llama3.2:3b', 'qwen2.5-coder:3b']
        
    def get_system_status(self):
        """Get comprehensive system status"""
        try:
            # System resources
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Add to history
            self.memory_history.append(memory.percent)
            self.cpu_history.append(cpu_percent)
            
            # Hardware optimizer status
            hw_status = self.get_hardware_optimizer_status()
            
            # Ollama status
            ollama_status = self.get_ollama_status()
            
            # Docker containers
            containers = self.get_container_status()
            
            return {
                'timestamp': datetime.now(),
                'memory': {
                    'total_gb': round(memory.total / (1024**3), 2),
                    'used_gb': round(memory.used / (1024**3), 2),
                    'available_gb': round(memory.available / (1024**3), 2),
                    'percent': memory.percent,
                    'trend': self.calculate_trend(self.memory_history)
                },
                'cpu': {
                    'percent': cpu_percent,
                    'count': psutil.cpu_count(),
                    'trend': self.calculate_trend(self.cpu_history)
                },
                'hardware_optimizer': hw_status,
                'ollama': ollama_status,
                'containers': containers
            }
        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.now()}
    
    def get_hardware_optimizer_status(self):
        """Get hardware optimizer status"""
        try:
            response = requests.get(f"{self.hardware_optimizer_url}/system-summary", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {'status': 'unavailable'}
    
    def get_ollama_status(self):
        """Get Ollama model status"""
        try:
            response = requests.get(f"{self.ollama_url}/api/ps", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                
                # Categorize models
                small_models = []
                large_models = []
                
                for model in models:
                    model_name = model.get('name', '')
                    if any(sm in model_name for sm in self.small_models):
                        small_models.append(model_name)
                    else:
                        large_models.append(model_name)
                
                return {
                    'status': 'online',
                    'total_models': len(models),
                    'small_models': small_models,
                    'large_models': large_models,
                    'memory_efficient': len(large_models) == 0
                }
        except:
            pass
        return {'status': 'offline'}
    
    def get_container_status(self):
        """Get Docker container status"""
        try:
            import docker
            client = docker.from_env()
            containers = client.containers.list()
            
            running_containers = []
            for container in containers:
                try:
                    stats = container.stats(stream=False)
                    memory_usage = stats['memory_stats']['usage']
                    memory_limit = stats['memory_stats']['limit']
                    memory_percent = (memory_usage / memory_limit) * 100.0 if memory_limit > 0 else 0
                    
                    running_containers.append({
                        'name': container.name,
                        'status': container.status,
                        'memory_mb': round(memory_usage / (1024**2), 1),
                        'memory_percent': round(memory_percent, 1)
                    })
                except:
                    running_containers.append({
                        'name': container.name,
                        'status': container.status,
                        'memory_mb': 0,
                        'memory_percent': 0
                    })
            
            return {
                'total': len(running_containers),
                'containers': sorted(running_containers, key=lambda x: x['memory_mb'], reverse=True)[:10]
            }
        except:
            return {'total': 0, 'containers': []}
    
    def calculate_trend(self, history):
        """Calculate trend from history"""
        if len(history) < 5:
            return "stable"
        
        recent = sum(history[-5:]) / 5
        earlier = sum(history[-10:-5]) / 5 if len(history) >= 10 else recent
        
        diff = recent - earlier
        if diff > 2:
            return "increasing"
        elif diff < -2:
            return "decreasing"
        else:
            return "stable"
    
    def get_trend_symbol(self, trend):
        """Get symbol for trend"""
        symbols = {
            "increasing": "📈",
            "decreasing": "📉",
            "stable": "📊"
        }
        return symbols.get(trend, "📊")
    
    def display_dashboard(self, status):
        """Display the monitoring dashboard"""
        clear_screen()
        
        print("🖥️  SutazAI Memory Monitor - Small Model System")
        print("=" * 80)
        print(f"Last Update: {status['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        if 'error' in status:
            print(f"❌ Error: {status['error']}")
            return
        
        # Memory Status
        memory = status['memory']
        mem_color = get_color_code(memory['percent'])
        trend_symbol = self.get_trend_symbol(memory['trend'])
        
        print(f"💾 MEMORY STATUS {trend_symbol}")
        print(f"   {mem_color}Usage: {memory['percent']:.1f}%{reset_color()} "
              f"({memory['used_gb']:.1f}GB / {memory['total_gb']:.1f}GB)")
        print(f"   Available: {memory['available_gb']:.1f}GB")
        print(f"   Trend: {memory['trend']}")
        
        # Memory bar
        bar_length = 50
        filled_length = int(bar_length * memory['percent'] / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"   [{mem_color}{bar}{reset_color()}]")
        print()
        
        # CPU Status
        cpu = status['cpu']
        cpu_color = get_color_code(cpu['percent'])
        cpu_trend_symbol = self.get_trend_symbol(cpu['trend'])
        
        print(f"🔥 CPU STATUS {cpu_trend_symbol}")
        print(f"   {cpu_color}Usage: {cpu['percent']:.1f}%{reset_color()}")
        print(f"   Cores: {cpu['count']}")
        print(f"   Trend: {cpu['trend']}")
        print()
        
        # Ollama Status
        ollama = status['ollama']
        print("🤖 OLLAMA STATUS")
        if ollama['status'] == 'online':
            efficient_color = '\033[92m' if ollama['memory_efficient'] else '\033[93m'
            print(f"   Status: {efficient_color}Online{reset_color()}")
            print(f"   Total Models: {ollama['total_models']}")
            print(f"   Small Models: {len(ollama['small_models'])}")
            print(f"   Large Models: {len(ollama['large_models'])}")
            
            if ollama['small_models']:
                print("   ✅ Loaded Small Models:")
                for model in ollama['small_models']:
                    print(f"      • {model}")
            
            if ollama['large_models']:
                print("   ⚠️  Large Models (should unload):")
                for model in ollama['large_models']:
                    print(f"      • {model}")
            
            efficiency_status = "✅ Memory Efficient" if ollama['memory_efficient'] else "⚠️  Not Optimized"
            print(f"   {efficiency_status}")
        else:
            print("   Status: 🔴 Offline")
        print()
        
        # Hardware Optimizer Status
        hw_opt = status['hardware_optimizer']
        print("⚙️  HARDWARE OPTIMIZER")
        if hw_opt.get('status') == 'success':
            print("   Status: ✅ Online")
            if 'small_model_mode' in hw_opt:
                mode_status = "✅ Enabled" if hw_opt['small_model_mode'] else "❌ Disabled"
                print(f"   Small Model Mode: {mode_status}")
            if 'memory_efficient' in hw_opt:
                efficiency = "✅ Efficient" if hw_opt['memory_efficient'] else "⚠️  Needs Optimization"
                print(f"   System Efficiency: {efficiency}")
        else:
            print("   Status: 🔴 Unavailable")
        print()
        
        # Container Status
        containers = status['containers']
        print(f"🐳 CONTAINERS ({containers['total']} running)")
        if containers['containers']:
            print("   Top Memory Users:")
            for i, container in enumerate(containers['containers'][:5]):
                mem_color = get_color_code(container['memory_percent'], 50, 80)
                status_emoji = "✅" if container['status'] == 'running' else "❌"
                print(f"   {i+1:2d}. {status_emoji} {container['name'][:25]:25} "
                      f"{mem_color}{container['memory_mb']:6.1f}MB{reset_color()}")
        print()
        
        # System Health Summary
        print("🏥 SYSTEM HEALTH SUMMARY")
        
        # Overall health score
        health_score = 100
        health_issues = []
        
        if memory['percent'] > 90:
            health_score -= 30
            health_issues.append("Critical memory usage")
        elif memory['percent'] > 80:
            health_score -= 15
            health_issues.append("High memory usage")
        
        if cpu['percent'] > 90:
            health_score -= 20
            health_issues.append("High CPU usage")
        
        if not ollama.get('memory_efficient', True):
            health_score -= 25
            health_issues.append("Large models loaded")
        
        if ollama['status'] != 'online':
            health_score -= 30
            health_issues.append("Ollama offline")
        
        # Display health score
        if health_score >= 90:
            health_color = '\033[92m'  # Green
            health_status = "Excellent"
        elif health_score >= 70:
            health_color = '\033[93m'  # Yellow
            health_status = "Good"
        else:
            health_color = '\033[91m'  # Red
            health_status = "Poor"
        
        print(f"   Overall Health: {health_color}{health_score}% ({health_status}){reset_color()}")
        
        if health_issues:
            print("   Issues:")
            for issue in health_issues:
                print(f"      ⚠️  {issue}")
        else:
            print("   ✅ No issues detected")
        
        print()
        
        # Memory optimization recommendations
        if memory['percent'] > 85:
            print("🔧 RECOMMENDATIONS")
            print("   • Unload large models: curl http://localhost:8523/unload-large-models")
            print("   • Force optimization: curl http://localhost:8523/optimize")
            print("   • Emergency cleanup: curl http://localhost:8523/emergency-scale-down")
            print()
        
        # Footer
        print("=" * 80)
        print("Press Ctrl+C to exit | Updates every 10 seconds")
    
    def start_monitoring(self):
        """Start the monitoring loop"""
        print("Starting SutazAI Memory Monitor Dashboard...")
        
        try:
            while self.running:
                status = self.get_system_status()
                self.display_dashboard(status)
                time.sleep(10)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"\nError in monitoring: {e}")
        finally:
            self.running = False

def main():
    """Main function"""
    dashboard = MemoryMonitorDashboard()
    dashboard.start_monitoring()

if __name__ == "__main__":
    main()