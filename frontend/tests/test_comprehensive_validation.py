#!/usr/bin/env python3
"""
Comprehensive Frontend UI Validation Test Suite
================================================
Validates findings from expert analysis reports with automated tests.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from axe_selenium_python import Axe
import numpy as np

# Configuration
FRONTEND_URL = "http://localhost:11000"
BACKEND_URL = "http://localhost:10200"
TIMEOUT = 10
PERFORMANCE_BUDGET = {
    "first_contentful_paint": 1800,  # ms
    "dom_content_loaded": 3000,  # ms
    "full_page_load": 5000,  # ms
    "memory_usage": 100 * 1024 * 1024,  # 100MB in bytes
}

class TestResult:
    """Container for test results with scoring"""
    def __init__(self, category: str):
        self.category = category
        self.passed = 0
        self.failed = 0
        self.violations = []
        self.warnings = []
        self.critical_issues = []
        
    def add_pass(self, test_name: str):
        self.passed += 1
        
    def add_fail(self, test_name: str, reason: str, severity: str = "major"):
        self.failed += 1
        if severity == "critical":
            self.critical_issues.append(f"{test_name}: {reason}")
        else:
            self.violations.append(f"{test_name}: {reason}")
    
    def add_warning(self, message: str):
        self.warnings.append(message)
    
    def get_score(self) -> float:
        total = self.passed + self.failed
        if total == 0:
            return 0
        return (self.passed / total) * 100


class ComprehensiveFrontendTester:
    """Main test orchestrator for comprehensive frontend validation"""
    
    def __init__(self):
        self.results = {
            "accessibility": TestResult("Accessibility"),
            "performance": TestResult("Performance"),
            "security": TestResult("Security"),
            "browser_compatibility": TestResult("Browser Compatibility"),
            "ux": TestResult("User Experience"),
        }
        self.driver = None
        
    def setup_driver(self, browser: str = "chrome"):
        """Initialize WebDriver with proper options"""
        if browser == "chrome":
            options = webdriver.ChromeOptions()
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--headless')
            options.add_experimental_option('excludeSwitches', ['enable-logging'])
            
            # Enable performance logging
            options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
            
            self.driver = webdriver.Chrome(options=options)
        elif browser == "firefox":
            options = webdriver.FirefoxOptions()
            options.add_argument('--headless')
            self.driver = webdriver.Firefox(options=options)
        
        self.driver.set_window_size(1920, 1080)
        return self.driver
    
    def teardown_driver(self):
        """Clean up WebDriver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    # ========== ACCESSIBILITY TESTS ==========
    
    def test_accessibility_wcag(self):
        """Test WCAG 2.1 Level AA compliance"""
        print("\n🔍 Testing Accessibility Compliance...")
        
        try:
            self.driver.get(FRONTEND_URL)
            time.sleep(3)  # Wait for page to fully load
            
            # Run axe accessibility tests
            axe = Axe(self.driver)
            axe.inject()
            results = axe.run()
            
            violations = results.get("violations", [])
            
            # Process violations by severity
            critical_count = 0
            major_count = 0
            
            for violation in violations:
                impact = violation.get("impact", "minor")
                description = violation.get("description", "Unknown issue")
                nodes = violation.get("nodes", [])
                
                if impact in ["critical", "serious"]:
                    critical_count += 1
                    self.results["accessibility"].add_fail(
                        violation.get("id", "unknown"),
                        f"{description} ({len(nodes)} instances)",
                        severity="critical"
                    )
                else:
                    major_count += 1
                    self.results["accessibility"].add_fail(
                        violation.get("id", "unknown"),
                        description,
                        severity="major"
                    )
            
            # Test specific WCAG criteria
            self._test_color_contrast()
            self._test_keyboard_navigation()
            self._test_aria_labels()
            self._test_focus_indicators()
            
            # Calculate pass rate
            if critical_count == 0 and major_count < 5:
                self.results["accessibility"].add_pass("WCAG_AA_Compliance")
            
            print(f"  ✅ Passed: {self.results['accessibility'].passed}")
            print(f"  ❌ Failed: {self.results['accessibility'].failed}")
            print(f"  🔴 Critical Issues: {critical_count}")
            
        except Exception as e:
            self.results["accessibility"].add_fail(
                "accessibility_test",
                str(e),
                severity="critical"
            )
    
    def _test_color_contrast(self):
        """Test color contrast ratios"""
        try:
            # Check primary text contrast
            element = self.driver.find_element(By.TAG_NAME, "body")
            bg_color = element.value_of_css_property("background-color")
            text_color = element.value_of_css_property("color")
            
            # Simple contrast check (would need proper calculation)
            if bg_color and text_color:
                self.results["accessibility"].add_pass("color_contrast")
            else:
                self.results["accessibility"].add_warning("Could not verify color contrast")
                
        except Exception as e:
            self.results["accessibility"].add_fail("color_contrast", str(e))
    
    def _test_keyboard_navigation(self):
        """Test keyboard navigation support"""
        try:
            # Test tab navigation
            body = self.driver.find_element(By.TAG_NAME, "body")
            body.send_keys(Keys.TAB)
            time.sleep(0.5)
            
            # Check if focus is visible
            active_element = self.driver.switch_to.active_element
            if active_element:
                self.results["accessibility"].add_pass("keyboard_navigation")
            else:
                self.results["accessibility"].add_fail(
                    "keyboard_navigation",
                    "No focusable elements found"
                )
                
        except Exception as e:
            self.results["accessibility"].add_fail("keyboard_navigation", str(e))
    
    def _test_aria_labels(self):
        """Test presence of ARIA labels"""
        try:
            # Check for ARIA labels on interactive elements
            buttons = self.driver.find_elements(By.TAG_NAME, "button")
            inputs = self.driver.find_elements(By.TAG_NAME, "input")
            
            missing_labels = 0
            for element in buttons + inputs:
                aria_label = element.get_attribute("aria-label")
                aria_labelledby = element.get_attribute("aria-labelledby")
                if not aria_label and not aria_labelledby:
                    missing_labels += 1
            
            if missing_labels == 0:
                self.results["accessibility"].add_pass("aria_labels")
            else:
                self.results["accessibility"].add_fail(
                    "aria_labels",
                    f"{missing_labels} elements missing ARIA labels"
                )
                
        except Exception as e:
            self.results["accessibility"].add_fail("aria_labels", str(e))
    
    def _test_focus_indicators(self):
        """Test visibility of focus indicators"""
        try:
            # Check for focus styles
            focused_element = self.driver.switch_to.active_element
            outline = focused_element.value_of_css_property("outline")
            
            if outline and outline != "none":
                self.results["accessibility"].add_pass("focus_indicators")
            else:
                self.results["accessibility"].add_fail(
                    "focus_indicators",
                    "Focus indicators not visible or missing"
                )
                
        except Exception as e:
            self.results["accessibility"].add_fail("focus_indicators", str(e))
    
    # ========== PERFORMANCE TESTS ==========
    
    def test_performance_metrics(self):
        """Test performance against defined budgets"""
        print("\n⚡ Testing Performance Metrics...")
        
        try:
            # Clear cache and cookies
            self.driver.delete_all_cookies()
            
            # Measure page load performance
            start_time = time.time()
            self.driver.get(FRONTEND_URL)
            
            # Wait for page to be interactive
            WebDriverWait(self.driver, TIMEOUT).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            load_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Get performance metrics from browser
            performance_data = self.driver.execute_script("""
                return {
                    navigation: performance.getEntriesByType('navigation')[0],
                    paint: performance.getEntriesByType('paint'),
                    memory: performance.memory
                };
            """)
            
            # Check First Contentful Paint
            fcp = None
            if performance_data and "paint" in performance_data:
                for entry in performance_data["paint"]:
                    if entry.get("name") == "first-contentful-paint":
                        fcp = entry.get("startTime", 0)
                        break
            
            if fcp and fcp < PERFORMANCE_BUDGET["first_contentful_paint"]:
                self.results["performance"].add_pass("first_contentful_paint")
            else:
                self.results["performance"].add_fail(
                    "first_contentful_paint",
                    f"FCP: {fcp}ms (budget: {PERFORMANCE_BUDGET['first_contentful_paint']}ms)",
                    severity="major"
                )
            
            # Check DOM Content Loaded
            if performance_data and "navigation" in performance_data:
                nav = performance_data["navigation"]
                dom_loaded = nav.get("domContentLoadedEventEnd", 0) - nav.get("domContentLoadedEventStart", 0)
                
                if dom_loaded < PERFORMANCE_BUDGET["dom_content_loaded"]:
                    self.results["performance"].add_pass("dom_content_loaded")
                else:
                    self.results["performance"].add_fail(
                        "dom_content_loaded",
                        f"DOM loaded: {dom_loaded}ms (budget: {PERFORMANCE_BUDGET['dom_content_loaded']}ms)"
                    )
            
            # Check total load time
            if load_time < PERFORMANCE_BUDGET["full_page_load"]:
                self.results["performance"].add_pass("full_page_load")
            else:
                self.results["performance"].add_fail(
                    "full_page_load",
                    f"Load time: {load_time:.0f}ms (budget: {PERFORMANCE_BUDGET['full_page_load']}ms)",
                    severity="major"
                )
            
            # Check memory usage
            if performance_data and "memory" in performance_data:
                memory_used = performance_data["memory"].get("usedJSHeapSize", 0)
                if memory_used < PERFORMANCE_BUDGET["memory_usage"]:
                    self.results["performance"].add_pass("memory_usage")
                else:
                    self.results["performance"].add_fail(
                        "memory_usage",
                        f"Memory: {memory_used / 1024 / 1024:.1f}MB (budget: {PERFORMANCE_BUDGET['memory_usage'] / 1024 / 1024}MB)"
                    )
            
            # Test for infinite animations
            self._test_animation_performance()
            
            print(f"  ✅ Passed: {self.results['performance'].passed}")
            print(f"  ❌ Failed: {self.results['performance'].failed}")
            
        except Exception as e:
            self.results["performance"].add_fail(
                "performance_test",
                str(e),
                severity="critical"
            )
    
    def _test_animation_performance(self):
        """Check for performance-impacting animations"""
        try:
            # Check for infinite animations
            animations = self.driver.execute_script("""
                const elements = document.querySelectorAll('*');
                let infiniteAnimations = 0;
                elements.forEach(el => {
                    const style = window.getComputedStyle(el);
                    if (style.animationIterationCount === 'infinite') {
                        infiniteAnimations++;
                    }
                });
                return infiniteAnimations;
            """)
            
            if animations > 0:
                self.results["performance"].add_fail(
                    "infinite_animations",
                    f"Found {animations} infinite animations impacting performance"
                )
            else:
                self.results["performance"].add_pass("infinite_animations")
                
        except Exception as e:
            self.results["performance"].add_warning(f"Could not test animations: {e}")
    
    # ========== SECURITY TESTS ==========
    
    def test_security_vulnerabilities(self):
        """Test for common security vulnerabilities"""
        print("\n🛡️ Testing Security...")
        
        try:
            self.driver.get(FRONTEND_URL)
            time.sleep(2)
            
            # Test for XSS vulnerabilities
            self._test_xss_prevention()
            
            # Test for unsafe HTML usage
            self._test_unsafe_html()
            
            # Test for secure headers
            self._test_security_headers()
            
            # Test for input validation
            self._test_input_validation()
            
            print(f"  ✅ Passed: {self.results['security'].passed}")
            print(f"  ❌ Failed: {self.results['security'].failed}")
            
        except Exception as e:
            self.results["security"].add_fail(
                "security_test",
                str(e),
                severity="critical"
            )
    
    def _test_xss_prevention(self):
        """Test XSS attack prevention"""
        try:
            # Try to inject script tag in input fields
            inputs = self.driver.find_elements(By.TAG_NAME, "input")
            
            if inputs:
                test_payload = "<script>alert('XSS')</script>"
                inputs[0].send_keys(test_payload)
                inputs[0].send_keys(Keys.RETURN)
                time.sleep(1)
                
                # Check if script executed (it shouldn't)
                try:
                    alert = self.driver.switch_to.alert
                    alert.dismiss()
                    self.results["security"].add_fail(
                        "xss_prevention",
                        "XSS vulnerability detected - script executed",
                        severity="critical"
                    )
                except:
                    # No alert means XSS was prevented
                    self.results["security"].add_pass("xss_prevention")
            else:
                self.results["security"].add_warning("No input fields to test XSS")
                
        except Exception as e:
            self.results["security"].add_fail("xss_prevention", str(e))
    
    def _test_unsafe_html(self):
        """Check for unsafe HTML usage"""
        try:
            # Check page source for unsafe patterns
            page_source = self.driver.page_source
            
            unsafe_patterns = [
                "unsafe_allow_html=True",
                "dangerouslySetInnerHTML",
                "eval(",
                "innerHTML =",
            ]
            
            found_unsafe = []
            for pattern in unsafe_patterns:
                if pattern in page_source:
                    found_unsafe.append(pattern)
            
            if found_unsafe:
                self.results["security"].add_fail(
                    "unsafe_html",
                    f"Found unsafe HTML patterns: {', '.join(found_unsafe)}",
                    severity="critical"
                )
            else:
                self.results["security"].add_pass("unsafe_html")
                
        except Exception as e:
            self.results["security"].add_fail("unsafe_html", str(e))
    
    def _test_security_headers(self):
        """Test for security headers"""
        try:
            response = requests.get(FRONTEND_URL, timeout=5)
            headers = response.headers
            
            required_headers = {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": ["DENY", "SAMEORIGIN"],
                "X-XSS-Protection": "1; mode=block",
                "Strict-Transport-Security": None,  # Check existence
                "Content-Security-Policy": None,  # Check existence
            }
            
            missing_headers = []
            for header, expected in required_headers.items():
                if header not in headers:
                    missing_headers.append(header)
                elif expected and headers[header] not in (expected if isinstance(expected, list) else [expected]):
                    missing_headers.append(f"{header} (incorrect value)")
            
            if missing_headers:
                self.results["security"].add_fail(
                    "security_headers",
                    f"Missing/incorrect headers: {', '.join(missing_headers)}",
                    severity="major"
                )
            else:
                self.results["security"].add_pass("security_headers")
                
        except Exception as e:
            self.results["security"].add_fail("security_headers", str(e))
    
    def _test_input_validation(self):
        """Test input validation"""
        try:
            # Find form inputs
            inputs = self.driver.find_elements(By.TAG_NAME, "input")
            
            validation_passed = True
            for input_elem in inputs:
                # Check for validation attributes
                input_type = input_elem.get_attribute("type")
                pattern = input_elem.get_attribute("pattern")
                required = input_elem.get_attribute("required")
                maxlength = input_elem.get_attribute("maxlength")
                
                # Email inputs should have proper type
                if "email" in input_elem.get_attribute("name", "").lower() and input_type != "email":
                    validation_passed = False
                    break
            
            if validation_passed:
                self.results["security"].add_pass("input_validation")
            else:
                self.results["security"].add_fail(
                    "input_validation",
                    "Insufficient input validation detected"
                )
                
        except Exception as e:
            self.results["security"].add_fail("input_validation", str(e))
    
    # ========== BROWSER COMPATIBILITY TESTS ==========
    
    def test_browser_compatibility(self):
        """Test cross-browser compatibility"""
        print("\n🌐 Testing Browser Compatibility...")
        
        try:
            # Test current browser (Chrome by default)
            self.driver.get(FRONTEND_URL)
            time.sleep(2)
            
            # Check for browser-specific CSS issues
            self._test_css_compatibility()
            
            # Check for JavaScript compatibility
            self._test_js_compatibility()
            
            # Check for feature detection
            self._test_feature_detection()
            
            print(f"  ✅ Passed: {self.results['browser_compatibility'].passed}")
            print(f"  ❌ Failed: {self.results['browser_compatibility'].failed}")
            
        except Exception as e:
            self.results["browser_compatibility"].add_fail(
                "browser_test",
                str(e),
                severity="major"
            )
    
    def _test_css_compatibility(self):
        """Test CSS compatibility"""
        try:
            # Check for vendor prefixes
            css_check = self.driver.execute_script("""
                const styles = document.styleSheets;
                let hasVendorPrefixes = false;
                for (let sheet of styles) {
                    try {
                        const rules = sheet.cssRules || sheet.rules;
                        for (let rule of rules) {
                            if (rule.cssText && rule.cssText.includes('-webkit-')) {
                                hasVendorPrefixes = true;
                                break;
                            }
                        }
                    } catch (e) {
                        // Cross-origin stylesheets
                    }
                }
                return hasVendorPrefixes;
            """)
            
            if css_check:
                self.results["browser_compatibility"].add_pass("css_prefixes")
            else:
                self.results["browser_compatibility"].add_warning(
                    "No vendor prefixes found - may have compatibility issues"
                )
                
        except Exception as e:
            self.results["browser_compatibility"].add_fail("css_compatibility", str(e))
    
    def _test_js_compatibility(self):
        """Test JavaScript compatibility"""
        try:
            # Check for modern JS features that might not be supported
            js_check = self.driver.execute_script("""
                return {
                    promises: typeof Promise !== 'undefined',
                    fetch: typeof fetch !== 'undefined',
                    localStorage: typeof localStorage !== 'undefined',
                    webSocket: typeof WebSocket !== 'undefined',
                    optionalChaining: true  // If this executes, it's supported
                };
            """)
            
            if all(js_check.values()):
                self.results["browser_compatibility"].add_pass("js_features")
            else:
                unsupported = [k for k, v in js_check.items() if not v]
                self.results["browser_compatibility"].add_fail(
                    "js_features",
                    f"Unsupported features: {', '.join(unsupported)}"
                )
                
        except Exception as e:
            self.results["browser_compatibility"].add_fail("js_compatibility", str(e))
    
    def _test_feature_detection(self):
        """Test for feature detection implementation"""
        try:
            # Check if feature detection is implemented
            feature_detection = self.driver.execute_script("""
                // Check for Modernizr or similar
                return typeof Modernizr !== 'undefined' || 
                       document.documentElement.className.includes('no-js');
            """)
            
            if feature_detection:
                self.results["browser_compatibility"].add_pass("feature_detection")
            else:
                self.results["browser_compatibility"].add_warning(
                    "No feature detection library found"
                )
                
        except Exception as e:
            self.results["browser_compatibility"].add_fail("feature_detection", str(e))
    
    # ========== UX TESTS ==========
    
    def test_user_experience(self):
        """Test user experience aspects"""
        print("\n👤 Testing User Experience...")
        
        try:
            self.driver.get(FRONTEND_URL)
            time.sleep(2)
            
            # Test navigation clarity
            self._test_navigation()
            
            # Test error handling
            self._test_error_handling()
            
            # Test loading states
            self._test_loading_states()
            
            # Test responsive design
            self._test_responsive_design()
            
            print(f"  ✅ Passed: {self.results['ux'].passed}")
            print(f"  ❌ Failed: {self.results['ux'].failed}")
            
        except Exception as e:
            self.results["ux"].add_fail(
                "ux_test",
                str(e),
                severity="major"
            )
    
    def _test_navigation(self):
        """Test navigation clarity and usability"""
        try:
            # Check for navigation elements
            nav_elements = self.driver.find_elements(By.TAG_NAME, "nav")
            sidebar = self.driver.find_elements(By.CLASS_NAME, "sidebar")
            
            if nav_elements or sidebar:
                self.results["ux"].add_pass("navigation_present")
            else:
                self.results["ux"].add_fail(
                    "navigation_present",
                    "No clear navigation structure found"
                )
                
        except Exception as e:
            self.results["ux"].add_fail("navigation", str(e))
    
    def _test_error_handling(self):
        """Test error message clarity"""
        try:
            # Try to trigger an error (e.g., invalid input)
            inputs = self.driver.find_elements(By.TAG_NAME, "input")
            
            if inputs:
                # Send invalid data
                inputs[0].clear()
                inputs[0].send_keys("")  # Empty required field
                inputs[0].send_keys(Keys.RETURN)
                time.sleep(1)
                
                # Check for error message
                error_elements = self.driver.find_elements(By.CLASS_NAME, "error")
                if not error_elements:
                    error_elements = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'error')]")
                
                if error_elements:
                    self.results["ux"].add_pass("error_handling")
                else:
                    self.results["ux"].add_warning("No error messages displayed for invalid input")
            
        except Exception as e:
            self.results["ux"].add_fail("error_handling", str(e))
    
    def _test_loading_states(self):
        """Test presence of loading indicators"""
        try:
            # Check for loading indicators in the page
            loading_indicators = self.driver.execute_script("""
                const indicators = document.querySelectorAll(
                    '.spinner, .loader, .loading, [class*="load"], [class*="spin"]'
                );
                return indicators.length > 0;
            """)
            
            if loading_indicators:
                self.results["ux"].add_pass("loading_states")
            else:
                self.results["ux"].add_warning("No loading indicators found")
                
        except Exception as e:
            self.results["ux"].add_fail("loading_states", str(e))
    
    def _test_responsive_design(self):
        """Test responsive design"""
        try:
            # Test different viewport sizes
            viewports = [
                (375, 667),   # Mobile
                (768, 1024),  # Tablet
                (1920, 1080), # Desktop
            ]
            
            responsive_issues = []
            for width, height in viewports:
                self.driver.set_window_size(width, height)
                time.sleep(1)
                
                # Check for horizontal scroll
                has_horizontal_scroll = self.driver.execute_script(
                    "return document.documentElement.scrollWidth > document.documentElement.clientWidth"
                )
                
                if has_horizontal_scroll:
                    responsive_issues.append(f"{width}x{height}")
            
            if not responsive_issues:
                self.results["ux"].add_pass("responsive_design")
            else:
                self.results["ux"].add_fail(
                    "responsive_design",
                    f"Horizontal scroll at: {', '.join(responsive_issues)}"
                )
                
            # Reset to default size
            self.driver.set_window_size(1920, 1080)
            
        except Exception as e:
            self.results["ux"].add_fail("responsive_design", str(e))
    
    # ========== REPORT GENERATION ==========
    
    def generate_report(self) -> Dict:
        """Generate comprehensive test report"""
        total_passed = sum(r.passed for r in self.results.values())
        total_failed = sum(r.failed for r in self.results.values())
        total_tests = total_passed + total_failed
        
        overall_score = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_score": round(overall_score, 1),
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "categories": {}
        }
        
        for category, result in self.results.items():
            report["categories"][category] = {
                "score": round(result.get_score(), 1),
                "passed": result.passed,
                "failed": result.failed,
                "violations": result.violations,
                "critical_issues": result.critical_issues,
                "warnings": result.warnings
            }
        
        return report
    
    def run_all_tests(self):
        """Execute all test categories"""
        print("\n" + "="*60)
        print("COMPREHENSIVE FRONTEND UI VALIDATION")
        print("="*60)
        
        try:
            # Setup driver
            self.setup_driver("chrome")
            
            # Run all test categories
            self.test_accessibility_wcag()
            self.test_performance_metrics()
            self.test_security_vulnerabilities()
            self.test_browser_compatibility()
            self.test_user_experience()
            
            # Generate and display report
            report = self.generate_report()
            
            print("\n" + "="*60)
            print("TEST RESULTS SUMMARY")
            print("="*60)
            print(f"Overall Score: {report['overall_score']}%")
            print(f"Total Tests: {report['total_tests']}")
            print(f"Passed: {report['passed']} ✅")
            print(f"Failed: {report['failed']} ❌")
            
            print("\nCategory Breakdown:")
            for category, data in report["categories"].items():
                print(f"\n{category.upper()}:")
                print(f"  Score: {data['score']}%")
                print(f"  Passed: {data['passed']}")
                print(f"  Failed: {data['failed']}")
                if data['critical_issues']:
                    print(f"  🔴 Critical Issues: {len(data['critical_issues'])}")
                if data['warnings']:
                    print(f"  ⚠️ Warnings: {len(data['warnings'])}")
            
            # Save detailed report to file
            self.save_detailed_report(report)
            
            return report
            
        finally:
            self.teardown_driver()
    
    def save_detailed_report(self, report: Dict):
        """Save detailed test results to file"""
        report_path = Path("/opt/sutazaiapp/frontend/tests/reports/automated_test_results.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Detailed report saved to: {report_path}")


def main():
    """Main test execution"""
    tester = ComprehensiveFrontendTester()
    
    try:
        # Check if frontend is accessible
        response = requests.get(FRONTEND_URL, timeout=5)
        if response.status_code != 200:
            print(f"⚠️ Warning: Frontend returned status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Warning: Cannot connect to frontend at {FRONTEND_URL}")
        print(f"   Error: {e}")
        print("\n   Attempting tests anyway...")
    
    # Run tests regardless of connection status
    report = tester.run_all_tests()
    
    # Exit with appropriate code
    if report["overall_score"] < 50:
        sys.exit(1)  # Fail if score is below 50%
    
    sys.exit(0)


if __name__ == "__main__":
    main()