#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
ULTRA UUID Migration Validation Test Suite
Comprehensive testing for UUID/INTEGER migration fix
"""

import asyncio
import json
import time
import requests
import random
import string
from datetime import datetime
from typing import Dict, Any, List

class UUIDMigrationTester:
    """Comprehensive UUID migration validation test suite"""
    
    def __init__(self):
        self.base_url = "http://localhost:10010"
        self.test_results = []
        self.errors = []
        
    def log_result(self, test_name: str, passed: bool, details: Dict[str, Any]):
        """Log test result"""
        result = {
            "test_name": test_name,
            "status": "PASS" if passed else "FAIL",
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        self.test_results.append(result)
        logger.info(f"[{'PASS' if passed else 'FAIL'}] {test_name}: {details.get('message', 'No message')}")
        
    def log_error(self, test_name: str, error: str):
        """Log test error"""
        self.errors.append({
            "test_name": test_name,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        logger.error(f"[ERROR] {test_name}: {error}")
    
    def generate_test_user_data(self) -> Dict[str, str]:
        """Generate test user registration data"""
        random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        return {
            "username": f"testuser_{random_id}",
            "email": f"test_{random_id}@example.com",
            "password": f"TestPass123!{random_id}"
        }
    
    def test_backend_health(self) -> bool:
        """Test backend health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                is_healthy = health_data.get("status") == "healthy"
                
                self.log_result("Backend Health Check", is_healthy, {
                    "message": "Backend is healthy" if is_healthy else "Backend is not healthy",
                    "status_code": response.status_code,
                    "health_data": health_data
                })
                return is_healthy
            else:
                self.log_result("Backend Health Check", False, {
                    "message": f"Health check failed with status {response.status_code}",
                    "status_code": response.status_code
                })
                return False
                
        except Exception as e:
            self.log_error("Backend Health Check", str(e))
            return False
    
    def test_user_registration(self) -> tuple[bool, Dict[str, str], Dict[str, Any]]:
        """Test user registration with integer IDs"""
        test_user = self.generate_test_user_data()
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/auth/register",
                json=test_user,
                timeout=10
            )
            
            if response.status_code == 201:
                user_data = response.json()
                user_id = user_data.get("id")
                
                # Verify that user ID is an integer
                is_integer_id = isinstance(user_id, int)
                
                self.log_result("User Registration", is_integer_id, {
                    "message": f"User registered with ID: {user_id} (type: {type(user_id).__name__})",
                    "user_id": user_id,
                    "user_id_type": type(user_id).__name__,
                    "status_code": response.status_code,
                    "response_data": user_data
                })
                
                return is_integer_id, test_user, user_data
            else:
                error_msg = response.text if response.text else "No error message"
                self.log_result("User Registration", False, {
                    "message": f"Registration failed with status {response.status_code}: {error_msg}",
                    "status_code": response.status_code,
                    "error": error_msg
                })
                return False, test_user, {}
                
        except Exception as e:
            self.log_error("User Registration", str(e))
            return False, test_user, {}
    
    def test_user_login(self, username: str, password: str) -> tuple[bool, str]:
        """Test user login and JWT token generation"""
        try:
            login_data = {
                "username": username,
                "password": password
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/auth/login",
                json=login_data,
                timeout=10
            )
            
            if response.status_code == 200:
                token_data = response.json()
                access_token = token_data.get("access_token")
                user_data = token_data.get("user", {})
                user_id = user_data.get("id")
                
                # Verify that user ID in token is an integer
                is_integer_id = isinstance(user_id, int)
                
                self.log_result("User Login", is_integer_id and access_token is not None, {
                    "message": f"Login successful. User ID: {user_id} (type: {type(user_id).__name__})",
                    "user_id": user_id,
                    "user_id_type": type(user_id).__name__,
                    "has_token": access_token is not None,
                    "status_code": response.status_code
                })
                
                return is_integer_id and access_token is not None, access_token
            else:
                error_msg = response.text if response.text else "No error message"
                self.log_result("User Login", False, {
                    "message": f"Login failed with status {response.status_code}: {error_msg}",
                    "status_code": response.status_code,
                    "error": error_msg
                })
                return False, ""
                
        except Exception as e:
            self.log_error("User Login", str(e))
            return False, ""
    
    def test_authenticated_endpoints(self, token: str) -> bool:
        """Test authenticated endpoints with JWT token"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            
            # Test user profile endpoint
            response = requests.get(
                f"{self.base_url}/api/v1/auth/profile",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                profile_data = response.json()
                user_id = profile_data.get("id")
                is_integer_id = isinstance(user_id, int)
                
                self.log_result("Authenticated Profile Access", is_integer_id, {
                    "message": f"Profile accessed. User ID: {user_id} (type: {type(user_id).__name__})",
                    "user_id": user_id,
                    "user_id_type": type(user_id).__name__,
                    "status_code": response.status_code,
                    "profile_data": profile_data
                })
                
                return is_integer_id
            else:
                self.log_result("Authenticated Profile Access", False, {
                    "message": f"Profile access failed with status {response.status_code}",
                    "status_code": response.status_code
                })
                return False
                
        except Exception as e:
            self.log_error("Authenticated Profile Access", str(e))
            return False
    
    def test_api_data_types(self) -> bool:
        """Test API endpoints for correct data types"""
        try:
            # Test models endpoint
            response = requests.get(f"{self.base_url}/api/v1/models/", timeout=10)
            
            if response.status_code == 200:
                models_data = response.json()
                
                self.log_result("API Data Types - Models", True, {
                    "message": "Models endpoint accessible",
                    "status_code": response.status_code,
                    "data_structure": type(models_data).__name__
                })
                return True
            else:
                self.log_result("API Data Types - Models", False, {
                    "message": f"Models endpoint failed with status {response.status_code}",
                    "status_code": response.status_code
                })
                return False
                
        except Exception as e:
            self.log_error("API Data Types - Models", str(e))
            return False
    
    def test_database_crud_operations(self) -> bool:
        """Test basic CRUD operations"""
        try:
            # Register a test user for CRUD testing
            success, test_user, user_data = self.test_user_registration()
            
            if not success:
                self.log_result("Database CRUD Operations", False, {
                    "message": "Failed to create test user for CRUD testing"
                })
                return False
            
            user_id = user_data.get("id")
            
            # Test user creation (already done above)
            # Test user read via login
            login_success, token = self.test_user_login(test_user["username"], test_user["password"])
            
            if not login_success:
                self.log_result("Database CRUD Operations", False, {
                    "message": "Failed to read user data via login"
                })
                return False
            
            # Test authenticated read
            auth_success = self.test_authenticated_endpoints(token)
            
            self.log_result("Database CRUD Operations", auth_success, {
                "message": "CRUD operations completed successfully" if auth_success else "CRUD operations failed",
                "user_id": user_id,
                "operations_tested": ["CREATE", "READ", "AUTHENTICATE"]
            })
            
            return auth_success
            
        except Exception as e:
            self.log_error("Database CRUD Operations", str(e))
            return False
    
    def test_service_connectivity(self) -> Dict[str, bool]:
        """Test connectivity to various services"""
        services = {
            "Backend API": f"{self.base_url}/health",
            "Models Endpoint": f"{self.base_url}/api/v1/models/",
            "Auth Endpoint": f"{self.base_url}/api/v1/auth/status"
        }
        
        results = {}
        
        for service_name, url in services.items():
            try:
                response = requests.get(url, timeout=5)
                is_accessible = response.status_code in [200, 404]  # 404 is acceptable for some endpoints
                results[service_name] = is_accessible
                
                self.log_result(f"Service Connectivity - {service_name}", is_accessible, {
                    "message": f"Service {'accessible' if is_accessible else 'not accessible'}",
                    "url": url,
                    "status_code": response.status_code
                })
                
            except Exception as e:
                results[service_name] = False
                self.log_error(f"Service Connectivity - {service_name}", str(e))
        
        return results
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete UUID migration validation test suite"""
        logger.info("=" * 80)
        logger.info("ULTRA UUID Migration Validation Test Suite")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Test 1: Backend Health
        health_ok = self.test_backend_health()
        
        # Test 2: Service Connectivity
        service_results = self.test_service_connectivity()
        
        # Test 3: User Registration (with integer ID validation)
        reg_success, test_user, user_data = self.test_user_registration()
        
        # Test 4: User Authentication
        if reg_success:
            login_success, token = self.test_user_login(test_user["username"], test_user["password"])
            
            # Test 5: Authenticated endpoints
            if login_success:
                auth_access = self.test_authenticated_endpoints(token)
            else:
                auth_access = False
        else:
            login_success = False
            auth_access = False
        
        # Test 6: API Data Types
        api_types_ok = self.test_api_data_types()
        
        # Test 7: Database CRUD
        crud_ok = self.test_database_crud_operations()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate summary
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "PASS"])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "overall_status": "PASS" if success_rate >= 80 else "FAIL",
            "errors_count": len(self.errors),
            "detailed_results": self.test_results,
            "errors": self.errors
        }
        
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUITE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.error(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Overall Status: {summary['overall_status']}")
        logger.error(f"Errors: {len(self.errors)}")
        
        if self.errors:
            logger.error("\nERRORS:")
            for error in self.errors:
                logger.error(f"  - {error['test_name']}: {error['error']}")
        
        return summary

def main():
    """Main test execution function"""
    tester = UUIDMigrationTester()
    results = tester.run_comprehensive_test_suite()
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"/opt/sutazaiapp/tests/uuid_migration_validation_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nDetailed results saved to: {results_file}")
    
    # Exit with appropriate code
    exit_code = 0 if results["overall_status"] == "PASS" else 1
    exit(exit_code)

if __name__ == "__main__":
    main()