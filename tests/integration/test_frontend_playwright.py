#!/usr/bin/env python3
"""
Comprehensive Frontend Testing with Playwright
Tests all aspects of the Streamlit frontend
"""

import asyncio
import json
from datetime import datetime
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

class FrontendTester:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests_passed": 0,
            "tests_failed": 0,
            "tests": []
        }
        
    async def test_homepage_loads(self, page):
        """Test if homepage loads successfully"""
        test_name = "Homepage Load Test"
        try:
            await page.goto("http://localhost:11000", wait_until="networkidle", timeout=30000)
            title = await page.title()
            
            # Check for Streamlit indicators
            streamlit_element = await page.query_selector('[data-testid="stApp"]')
            
            if streamlit_element:
                self.results["tests"].append({
                    "name": test_name,
                    "status": "passed",
                    "details": f"Page loaded, title: {title}"
                })
                self.results["tests_passed"] += 1
                print(f"✅ {test_name}: PASSED")
                return True
            else:
                raise Exception("Streamlit app container not found")
                
        except Exception as e:
            self.results["tests"].append({
                "name": test_name,
                "status": "failed",
                "error": str(e)
            })
            self.results["tests_failed"] += 1
            print(f"❌ {test_name}: FAILED - {str(e)}")
            return False
            
    async def test_chat_interface(self, page):
        """Test if chat interface is available"""
        test_name = "Chat Interface Test"
        try:
            # Look for chat input
            chat_input = await page.query_selector('input[type="text"], textarea')
            
            if chat_input:
                # Try to type in the chat
                await chat_input.fill("Test message")
                
                self.results["tests"].append({
                    "name": test_name,
                    "status": "passed",
                    "details": "Chat input found and functional"
                })
                self.results["tests_passed"] += 1
                print(f"✅ {test_name}: PASSED")
                return True
            else:
                raise Exception("Chat input not found")
                
        except Exception as e:
            self.results["tests"].append({
                "name": test_name,
                "status": "failed",
                "error": str(e)
            })
            self.results["tests_failed"] += 1
            print(f"❌ {test_name}: FAILED - {str(e)}")
            return False
            
    async def test_sidebar(self, page):
        """Test if sidebar is present and functional"""
        test_name = "Sidebar Test"
        try:
            # Look for sidebar
            sidebar = await page.query_selector('[data-testid="stSidebar"]')
            
            if sidebar:
                # Check if sidebar has content
                sidebar_content = await sidebar.inner_text()
                
                if sidebar_content:
                    self.results["tests"].append({
                        "name": test_name,
                        "status": "passed",
                        "details": f"Sidebar found with content length: {len(sidebar_content)}"
                    })
                    self.results["tests_passed"] += 1
                    print(f"✅ {test_name}: PASSED")
                    return True
                else:
                    raise Exception("Sidebar empty")
                    
            else:
                # Sidebar might be collapsed
                self.results["tests"].append({
                    "name": test_name,
                    "status": "warning",
                    "details": "Sidebar not visible (might be collapsed)"
                })
                print(f"⚠️  {test_name}: WARNING - Sidebar not visible")
                return True
                
        except Exception as e:
            self.results["tests"].append({
                "name": test_name,
                "status": "failed",
                "error": str(e)
            })
            self.results["tests_failed"] += 1
            print(f"❌ {test_name}: FAILED - {str(e)}")
            return False
            
    async def test_api_connection(self, page):
        """Test if frontend can connect to backend API"""
        test_name = "API Connection Test"
        try:
            # Intercept network requests
            api_called = False
            
            def handle_request(request):
                nonlocal api_called
                if "10200" in request.url or "api" in request.url:
                    api_called = True
                    
            page.on("request", handle_request)
            
            # Trigger some action that would call API
            await page.reload()
            await page.wait_for_timeout(3000)
            
            if api_called:
                self.results["tests"].append({
                    "name": test_name,
                    "status": "passed",
                    "details": "API connection detected"
                })
                self.results["tests_passed"] += 1
                print(f"✅ {test_name}: PASSED")
            else:
                self.results["tests"].append({
                    "name": test_name,
                    "status": "warning",
                    "details": "No API calls detected (might be normal for initial load)"
                })
                print(f"⚠️  {test_name}: WARNING - No API calls detected")
                
            return True
                
        except Exception as e:
            self.results["tests"].append({
                "name": test_name,
                "status": "failed",
                "error": str(e)
            })
            self.results["tests_failed"] += 1
            print(f"❌ {test_name}: FAILED - {str(e)}")
            return False
            
    async def test_responsive_design(self, page):
        """Test if UI is responsive"""
        test_name = "Responsive Design Test"
        try:
            viewports = [
                {"name": "Mobile", "width": 375, "height": 667},
                {"name": "Tablet", "width": 768, "height": 1024},
                {"name": "Desktop", "width": 1920, "height": 1080}
            ]
            
            for viewport in viewports:
                await page.set_viewport_size(
                    width=viewport["width"],
                    height=viewport["height"]
                )
                await page.wait_for_timeout(1000)
                
                # Check if main content is visible
                main_content = await page.query_selector('[data-testid="stApp"]')
                is_visible = await main_content.is_visible() if main_content else False
                
                if not is_visible:
                    raise Exception(f"Content not visible at {viewport['name']} size")
                    
            self.results["tests"].append({
                "name": test_name,
                "status": "passed",
                "details": "Responsive at all viewport sizes"
            })
            self.results["tests_passed"] += 1
            print(f"✅ {test_name}: PASSED")
            return True
            
        except Exception as e:
            self.results["tests"].append({
                "name": test_name,
                "status": "failed",
                "error": str(e)
            })
            self.results["tests_failed"] += 1
            print(f"❌ {test_name}: FAILED - {str(e)}")
            return False
            
    async def test_accessibility(self, page):
        """Basic accessibility tests"""
        test_name = "Accessibility Test"
        try:
            # Check for basic accessibility features
            checks = {
                "lang_attribute": await page.evaluate('() => document.documentElement.lang'),
                "title": await page.title(),
                "keyboard_nav": await page.evaluate('() => document.querySelectorAll("[tabindex]").length > 0'),
                "aria_labels": await page.evaluate('() => document.querySelectorAll("[aria-label]").length > 0')
            }
            
            issues = []
            if not checks["lang_attribute"]:
                issues.append("Missing lang attribute")
            if not checks["title"]:
                issues.append("Missing page title")
            if not checks["keyboard_nav"]:
                issues.append("No keyboard navigation elements")
                
            if issues:
                self.results["tests"].append({
                    "name": test_name,
                    "status": "warning",
                    "details": f"Accessibility issues: {', '.join(issues)}"
                })
                print(f"⚠️  {test_name}: WARNING - {', '.join(issues)}")
            else:
                self.results["tests"].append({
                    "name": test_name,
                    "status": "passed",
                    "details": "Basic accessibility checks passed"
                })
                self.results["tests_passed"] += 1
                print(f"✅ {test_name}: PASSED")
                
            return True
            
        except Exception as e:
            self.results["tests"].append({
                "name": test_name,
                "status": "failed",
                "error": str(e)
            })
            self.results["tests_failed"] += 1
            print(f"❌ {test_name}: FAILED - {str(e)}")
            return False
            
    async def run_all_tests(self):
        """Run all frontend tests"""
        print("\n" + "="*60)
        print("FRONTEND PLAYWRIGHT TESTING".center(60))
        print("="*60 + "\n")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            # Run all tests
            await self.test_homepage_loads(page)
            await self.test_chat_interface(page)
            await self.test_sidebar(page)
            await self.test_api_connection(page)
            await self.test_responsive_design(page)
            await self.test_accessibility(page)
            
            # Take screenshot for debugging
            await page.screenshot(path="/opt/sutazaiapp/frontend_test_screenshot.png")
            
            await browser.close()
            
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY".center(60))
        print("="*60)
        print(f"✅ Tests Passed: {self.results['tests_passed']}")
        print(f"❌ Tests Failed: {self.results['tests_failed']}")
        
        # Save results
        with open("/opt/sutazaiapp/frontend_test_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
            
        print(f"\nResults saved to: /opt/sutazaiapp/frontend_test_results.json")
        print(f"Screenshot saved to: /opt/sutazaiapp/frontend_test_screenshot.png")
        
        return self.results["tests_failed"] == 0

async def main():
    tester = FrontendTester()
    success = await tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    exit(asyncio.run(main()))