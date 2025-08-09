#!/usr/bin/env python3
"""
Test Enhanced Monitor Components
===============================

Tests the enhanced monitor components without requiring a TTY.
"""

import sys
import json
import tempfile
from pathlib import Path

# Add the monitoring directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from static_monitor import EnhancedMonitor
    print("✅ Successfully imported EnhancedMonitor")
except ImportError as e:
    print(f"❌ Failed to import EnhancedMonitor: {e}")
    sys.exit(1)

def test_config_loading():
    """Test configuration loading"""
    print("\n🔧 Testing configuration loading...")
    
    # Test with no config
    monitor = EnhancedMonitor()
    assert monitor.config is not None
    print("✅ Default configuration loaded")
    
    # Test with custom config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_config = {
            "refresh_rate": 1.0,
            "thresholds": {
                "cpu_warning": 60,
                "cpu_critical": 80
            }
        }
        json.dump(test_config, f)
        f.flush()
        
        monitor2 = EnhancedMonitor(f.name)
        assert monitor2.config['refresh_rate'] == 1.0
        assert monitor2.config['thresholds']['cpu_warning'] == 60
        print("✅ Custom configuration loaded")
    
    Path(f.name).unlink()  # Clean up

def test_system_stats():
    """Test system statistics gathering"""
    print("\n📊 Testing system statistics...")
    
    monitor = EnhancedMonitor()
    stats = monitor.get_system_stats()
    
    required_keys = [
        'cpu_percent', 'cpu_cores', 'cpu_trend',
        'mem_percent', 'mem_used', 'mem_total', 'mem_trend',
        'disk_percent', 'disk_free',
        'network', 'connections'
    ]
    
    for key in required_keys:
        assert key in stats, f"Missing key: {key}"
    
    assert isinstance(stats['cpu_percent'], (int, float))
    assert isinstance(stats['cpu_cores'], int)
    assert stats['cpu_trend'] in ['↑', '↓', '→']
    assert isinstance(stats['network'], dict)
    
    print("✅ System statistics gathering works")

def test_agent_registry_loading():
    """Test AI agent registry loading"""
    print("\n🤖 Testing AI agent registry loading...")
    
    monitor = EnhancedMonitor()
    registry = monitor.agent_registry
    
    assert isinstance(registry, dict)
    assert 'agents' in registry
    
    if registry['agents']:
        print(f"✅ Loaded {len(registry['agents'])} agents from registry")
    else:
        print("✅ Agent registry loaded (empty)")

def test_color_and_display():
    """Test color and display functions"""
    print("\n🎨 Testing display functions...")
    
    monitor = EnhancedMonitor()
    
    # Test color selection
    green_color = monitor.get_color(50, 60, 80)  # Below warning
    yellow_color = monitor.get_color(70, 60, 80)  # Warning level
    red_color = monitor.get_color(90, 60, 80)  # Critical level
    
    assert green_color == monitor.GREEN
    assert yellow_color == monitor.YELLOW
    assert red_color == monitor.RED
    
    # Test progress bar
    bar = monitor.create_bar(75, 20)
    assert len(bar) == 20
    assert '█' in bar and '░' in bar
    
    print("✅ Display functions work correctly")

def test_network_calculations():
    """Test network statistics calculations"""
    print("\n🌐 Testing network calculations...")
    
    monitor = EnhancedMonitor()
    
    # Mock network data
    class MockNetwork:
        def __init__(self, bytes_sent, bytes_recv):
            self.bytes_sent = bytes_sent
            self.bytes_recv = bytes_recv
    
    # First call should initialize baseline
    net1 = MockNetwork(1000000, 2000000)
    result1 = monitor._calculate_network_stats(net1)
    assert result1['bandwidth_mbps'] == 0.0  # First call
    
    print("✅ Network calculations work")

def test_trend_calculation():
    """Test trend calculation"""
    print("\n📈 Testing trend calculations...")
    
    monitor = EnhancedMonitor()
    
    # Test increasing trend
    monitor.history['cpu'].extend([10, 20, 30])
    trend = monitor._get_trend(monitor.history['cpu'])
    assert trend == '↑'
    
    # Test decreasing trend
    monitor.history['memory'].extend([30, 20, 10])
    trend = monitor._get_trend(monitor.history['memory'])
    assert trend == '↓'
    
    # Test stable trend
    monitor.history['network'].extend([50, 45, 55])
    trend = monitor._get_trend(monitor.history['network'])
    assert trend == '→'
    
    print("✅ Trend calculations work")

def test_adaptive_refresh():
    """Test adaptive refresh rate"""
    print("\n⚡ Testing adaptive refresh rate...")
    
    monitor = EnhancedMonitor()
    original_rate = monitor.current_refresh_rate
    
    # Test high activity scenario
    monitor._update_refresh_rate(85, 80)  # High CPU and memory
    assert monitor.current_refresh_rate < original_rate
    
    # Test normal activity scenario
    monitor._update_refresh_rate(30, 40)  # Normal CPU and memory
    
    print("✅ Adaptive refresh rate works")

def test_alert_generation():
    """Test alert message generation"""
    print("\n🚨 Testing alert generation...")
    
    monitor = EnhancedMonitor()
    
    # Test normal conditions
    stats = {
        'cpu_percent': 30,
        'mem_percent': 40,
        'disk_percent': 50,
        'network': {'bandwidth_mbps': 10}
    }
    alert = monitor._generate_alert_message(stats, 5, 5)  # All agents healthy
    assert 'operational' in alert.lower()
    
    # Test high CPU
    stats['cpu_percent'] = 90
    alert = monitor._generate_alert_message(stats, 5, 5)
    assert 'HIGH CPU' in alert
    
    print("✅ Alert generation works")

def main():
    """Run all tests"""
    print("🧪 Testing Enhanced Monitor Components")
    print("=" * 50)
    
    try:
        test_config_loading()
        test_system_stats()
        test_agent_registry_loading()
        test_color_and_display()
        test_network_calculations()
        test_trend_calculation()
        test_adaptive_refresh()
        test_alert_generation()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! Enhanced Monitor is ready for use.")
        print("\nTo run the monitor:")
        print("  ./run_enhanced_monitor.sh")
        print("  python3 static_monitor.py --force")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())