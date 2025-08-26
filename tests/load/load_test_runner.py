#!/usr/bin/env python3
"""Load test runner for performance testing."""

import sys
import argparse
import time
import requests
from concurrent.futures import ThreadPoolExecutor
import statistics

def run_load_test(host, users, spawn_rate, run_time):
    """Run a simple load test."""
    print(f"🚀 Running load test...")
    print(f"   Host: {host}")
    print(f"   Users: {users}")
    print(f"   Spawn rate: {spawn_rate}")
    print(f"   Duration: {run_time}")
    print("=" * 40)
    
    # Parse run time (e.g., "300s" -> 300)
    if run_time.endswith('s'):
        duration = int(run_time[:-1])
    else:
        duration = int(run_time)
    
    start_time = time.time()
    response_times = []
    errors = 0
    
    def make_request():
        """Make a single request."""
        nonlocal errors
        try:
            start = time.time()
            response = requests.get(f"{host}/health", timeout=5)
            elapsed = time.time() - start
            response_times.append(elapsed)
            if response.status_code != 200:
                errors += 1
        except Exception:
            errors += 1
            response_times.append(5.0)  # Timeout
    
    # Simulate load
    with ThreadPoolExecutor(max_workers=users) as executor:
        end_time = time.time() + duration
        request_count = 0
        
        while time.time() < end_time:
            executor.submit(make_request)
            request_count += 1
            time.sleep(1.0 / spawn_rate)  # Control spawn rate
            
            if request_count % 10 == 0:
                print(f"   Requests sent: {request_count}")
    
    # Calculate statistics
    if response_times:
        avg_response = statistics.mean(response_times)
        median_response = statistics.median(response_times)
        max_response = max(response_times)
        min_response = min(response_times)
        
        print("\n📊 Load Test Results:")
        print("=" * 40)
        print(f"Total requests: {len(response_times)}")
        print(f"Failed requests: {errors}")
        print(f"Average response time: {avg_response:.3f}s")
        print(f"Median response time: {median_response:.3f}s")
        print(f"Min response time: {min_response:.3f}s")
        print(f"Max response time: {max_response:.3f}s")
        print(f"Error rate: {(errors/len(response_times)*100):.1f}%")
    else:
        print("❌ No successful requests")
    
    return 0 if errors < len(response_times) * 0.1 else 1  # Fail if >10% errors

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Load test runner')
    parser.add_argument('--host', default='http://localhost:8000', help='Target host')
    parser.add_argument('--users', type=int, default=10, help='Number of concurrent users')
    parser.add_argument('--spawn-rate', type=int, default=2, help='Users spawned per second')
    parser.add_argument('--run-time', default='60s', help='Test duration')
    
    args = parser.parse_args()
    
    return run_load_test(args.host, args.users, args.spawn_rate, args.run_time)

if __name__ == "__main__":
    sys.exit(main())