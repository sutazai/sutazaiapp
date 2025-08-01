#!/usr/bin/env python3
"""
Test Static Display - Demonstrates the fixed layout approach
"""

import time
import random
from static_monitor import StaticMonitor

def demonstrate_static_display():
    """Show how the static display works with simulated updates"""
    
    monitor = StaticMonitor()
    
    print("=== Static Monitor Display Test ===")
    print("This test shows how the monitor updates values in-place without scrolling.")
    print("In a real terminal, you would see a fixed layout with only numbers changing.")
    print()
    
    # Show the layout structure
    print("Display Layout Structure:")
    print("-" * 60)
    for i, line in enumerate(monitor.display_lines):
        if '{' in line:
            print(f"Line {i+1:2d}: {line[:60]}...")
        elif line.strip():
            print(f"Line {i+1:2d}: {line[:60]}")
    
    print("\n=== Key Improvements in the Fixed Version ===")
    print("1. ✓ Fixed number of lines (35 lines total)")
    print("2. ✓ Each section has fixed slots (5 containers, 3 alerts)")
    print("3. ✓ Uses cursor positioning to update individual lines")
    print("4. ✓ Pads lines to 80 characters to prevent artifacts")
    print("5. ✓ Only updates lines with dynamic content ({} placeholders)")
    print("6. ✓ Clears screen once at startup, never again")
    print("7. ✓ Hides cursor during operation to prevent flickering")
    print()
    
    # Simulate a few updates to show the concept
    print("=== Simulated Updates (showing only changed values) ===")
    for update in range(5):
        print(f"\nUpdate {update + 1}:")
        
        # Simulate changing system data
        cpu_percent = random.uniform(10, 95)
        mem_percent = random.uniform(40, 85)
        
        cpu_bar = monitor.create_bar(cpu_percent)
        mem_bar = monitor.create_bar(mem_percent)
        
        cpu_color = monitor.get_color(cpu_percent, 60, 80)
        mem_color = monitor.get_color(mem_percent, 75, 90)
        
        # Show what would be updated
        print(f"  CPU:    {cpu_bar}  {cpu_color}{cpu_percent:5.1f}%{monitor.RESET}")
        print(f"  Memory: {mem_bar}  {mem_color}{mem_percent:5.1f}%{monitor.RESET}")
        print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        time.sleep(1)
    
    print("\n=== How to Use ===")
    print("1. Run in a real terminal: python3 static_monitor.py")
    print("2. The screen will clear once and show the fixed layout")
    print("3. Only the numeric values and bars will update every 2 seconds")
    print("4. The layout stays completely static - no scrolling ever")
    print("5. Press Ctrl+C to exit cleanly")
    print()
    print("This works like professional monitors (top, htop, etc.)")

if __name__ == "__main__":
    demonstrate_static_display()