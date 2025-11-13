#!/usr/bin/env python3
"""
Comprehensive Full Stack Integration Test
Tests all deployed services and their connectivity
"""

import sys
sys.path.insert(0, '/usr/lib/python3/dist-packages')

import requests
from typing import Dict, List, Tuple

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def test_service(name: str, url: str, expected_status: int = 200) -> Tuple[bool, str]:
    """Test if a service is accessible"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == expected_status:
            return True, f"Status {response.status_code}"
        else:
            return False, f"Unexpected status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Connection refused"
    except requests.exceptions.Timeout:
        return False, "Request timeout"
    except Exception as e:
        return False, f"Error: {str(e)}"

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}{text:^70}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

def print_result(service: str, status: bool, message: str):
    """Print test result"""
    status_icon = f"{GREEN}✓{RESET}" if status else f"{RED}✗{RESET}"
    status_text = f"{GREEN}PASSED{RESET}" if status else f"{RED}FAILED{RESET}"
    print(f"{status_icon} {service:<30} {status_text:<20} {message}")

def main():
    print_header("SutazAI Platform - Full Stack Integration Test")
    
    # Define all services to test
    services = {
        "Infrastructure Services": [
            ("PostgreSQL", "http://localhost:10000", 404),  # No HTTP endpoint
            ("Redis", "http://localhost:10001", 404),       # No HTTP endpoint
            ("Neo4j", "http://localhost:10002"),
            ("RabbitMQ Management", "http://localhost:10005"),
            ("Consul", "http://localhost:10006/v1/status/leader"),
            ("Kong Admin API", "http://localhost:10009"),
        ],
        "Vector Databases": [
            ("ChromaDB", "http://localhost:10100/api/v1"),
            ("Qdrant", "http://localhost:10101"),
            ("FAISS Service", "http://localhost:10103/health"),
        ],
        "Application Layer": [
            ("Backend API - Health", "http://localhost:10200/health"),
            ("Backend API - Detailed Health", "http://localhost:10200/health/detailed"),
            ("Frontend - Streamlit Health", "http://localhost:11000/_stcore/health"),
        ]
    }
    
    total_tests = 0
    passed_tests = 0
    failed_services = []
    
    # Run tests for each category
    for category, service_list in services.items():
        print(f"\n{YELLOW}{category}{RESET}")
        print(f"{'-'*70}")
        
        for service_name, url, *expected in service_list:
            total_tests += 1
            expected_status = expected[0] if expected else 200
            success, message = test_service(service_name, url, expected_status)
            print_result(service_name, success, message)
            
            if success:
                passed_tests += 1
            else:
                failed_services.append(service_name)
    
    # Test backend service connectivity
    print(f"\n{YELLOW}Backend Service Connectivity{RESET}")
    print(f"{'-'*70}")
    
    try:
        response = requests.get("http://localhost:10200/health/detailed", timeout=5)
        if response.status_code == 200:
            data = response.json()
            services_status = data.get('services', {})
            
            for service, is_connected in services_status.items():
                total_tests += 1
                print_result(f"Backend → {service}", is_connected, 
                           "Connected" if is_connected else "Not connected")
                if is_connected:
                    passed_tests += 1
                else:
                    failed_services.append(f"Backend → {service}")
    except Exception as e:
        print_result("Backend connectivity check", False, str(e))
    
    # Print summary
    print_header("Test Summary")
    
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"{GREEN}Passed: {passed_tests}{RESET}")
    print(f"{RED}Failed: {len(failed_services)}{RESET}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    
    if failed_services:
        print(f"\n{RED}Failed Services:{RESET}")
        for service in failed_services:
            print(f"  - {service}")
    
    print(f"\n{BLUE}{'='*70}{RESET}\n")
    
    # Return exit code based on results
    return 0 if len(failed_services) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
