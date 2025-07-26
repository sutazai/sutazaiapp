#!/usr/bin/env python3
"""
Web Automation for SutazAI V7 Self-Supervised Learning
Browser automation for complex web interactions and JavaScript-heavy sites
"""

import os
import sys
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import time

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.keys import Keys
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    import pyppeteer
    PYPPETEER_AVAILABLE = True
except ImportError:
    PYPPETEER_AVAILABLE = False

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

@dataclass
class AutomationConfig:
    """Configuration for web automation"""
    # Browser settings
    browser_type: str = "chrome"  # chrome, firefox, headless
    headless: bool = True
    window_size: Tuple[int, int] = (1920, 1080)
    
    # Timeouts
    page_load_timeout: int = 30
    element_wait_timeout: int = 10
    implicit_wait: int = 5
    
    # Behavior
    auto_scroll: bool = True
    take_screenshots: bool = False
    save_page_source: bool = False
    
    # Rate limiting
    min_delay_between_actions: float = 1.0
    max_delay_between_actions: float = 3.0
    
    # Security
    disable_images: bool = True
    disable_javascript: bool = False
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    
    # Storage
    download_directory: str = "/tmp/sutazai_downloads"
    screenshot_directory: str = "/tmp/sutazai_screenshots"

class BrowserManager:
    """Manages browser instances and automation"""
    
    def __init__(self, config: AutomationConfig):
        self.config = config
        self.driver = None
        self.session_id = None
        
        # Create directories
        Path(config.download_directory).mkdir(parents=True, exist_ok=True)
        Path(config.screenshot_directory).mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'pages_visited': 0,
            'actions_performed': 0,
            'screenshots_taken': 0,
            'errors_encountered': 0,
            'start_time': datetime.now()
        }
    
    async def initialize(self):
        """Initialize browser driver"""
        if not SELENIUM_AVAILABLE:
            raise RuntimeError("Selenium not available. Install with: pip install selenium")
        
        try:
            if self.config.browser_type == "chrome":
                self.driver = self._create_chrome_driver()
            elif self.config.browser_type == "firefox":
                self.driver = self._create_firefox_driver()
            else:
                raise ValueError(f"Unsupported browser type: {self.config.browser_type}")
            
            # Set timeouts
            self.driver.implicitly_wait(self.config.implicit_wait)
            self.driver.set_page_load_timeout(self.config.page_load_timeout)
            
            # Set window size
            self.driver.set_window_size(*self.config.window_size)
            
            self.session_id = self.driver.session_id
            logger.info(f"Browser initialized: {self.config.browser_type} (Session: {self.session_id})")
            
        except Exception as e:
            logger.error(f"Error initializing browser: {e}")
            raise
    
    def _create_chrome_driver(self):
        """Create Chrome driver with options"""
        options = ChromeOptions()
        
        if self.config.headless:
            options.add_argument("--headless")
        
        # Security options
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-plugins")
        options.add_argument("--disable-popup-blocking")
        
        # Performance options
        if self.config.disable_images:
            options.add_argument("--disable-images")
        
        if self.config.disable_javascript:
            options.add_argument("--disable-javascript")
        
        # User agent
        options.add_argument(f"--user-agent={self.config.user_agent}")
        
        # Download directory
        prefs = {
            "download.default_directory": self.config.download_directory,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        options.add_experimental_option("prefs", prefs)
        
        return webdriver.Chrome(options=options)
    
    def _create_firefox_driver(self):
        """Create Firefox driver with options"""
        options = FirefoxOptions()
        
        if self.config.headless:
            options.add_argument("--headless")
        
        # Set user agent
        options.set_preference("general.useragent.override", self.config.user_agent)
        
        # Download settings
        options.set_preference("browser.download.folderList", 2)
        options.set_preference("browser.download.manager.showWhenStarting", False)
        options.set_preference("browser.download.dir", self.config.download_directory)
        
        # Disable images if requested
        if self.config.disable_images:
            options.set_preference("permissions.default.image", 2)
        
        # Disable JavaScript if requested
        if self.config.disable_javascript:
            options.set_preference("javascript.enabled", False)
        
        return webdriver.Firefox(options=options)
    
    async def navigate_to_url(self, url: str) -> Dict[str, Any]:
        """Navigate to a URL and return page information"""
        if not self.driver:
            raise RuntimeError("Browser not initialized")
        
        try:
            logger.debug(f"Navigating to {url}")
            self.driver.get(url)
            
            # Wait for page to load
            await self._wait_for_page_load()
            
            # Auto-scroll if enabled
            if self.config.auto_scroll:
                await self._auto_scroll_page()
            
            # Take screenshot if enabled
            screenshot_path = None
            if self.config.take_screenshots:
                screenshot_path = await self._take_screenshot(url)
            
            # Get page information
            page_info = {
                'url': url,
                'title': self.driver.title,
                'current_url': self.driver.current_url,
                'page_source': self.driver.page_source if self.config.save_page_source else None,
                'screenshot_path': screenshot_path,
                'timestamp': datetime.now().isoformat()
            }
            
            self.stats['pages_visited'] += 1
            
            return page_info
            
        except Exception as e:
            logger.error(f"Error navigating to {url}: {e}")
            self.stats['errors_encountered'] += 1
            raise
    
    async def _wait_for_page_load(self):
        """Wait for page to fully load"""
        try:
            # Wait for document ready state
            WebDriverWait(self.driver, self.config.element_wait_timeout).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            
            # Additional wait for dynamic content
            await asyncio.sleep(2)
            
        except TimeoutException:
            logger.warning("Page load timeout")
    
    async def _auto_scroll_page(self):
        """Automatically scroll through the page to load dynamic content"""
        try:
            # Get initial page height
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            
            scroll_attempts = 0
            max_attempts = 10
            
            while scroll_attempts < max_attempts:
                # Scroll to bottom
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                
                # Wait for new content to load
                await asyncio.sleep(2)
                
                # Calculate new height
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                
                # Break if no new content loaded
                if new_height == last_height:
                    break
                
                last_height = new_height
                scroll_attempts += 1
            
            # Scroll back to top
            self.driver.execute_script("window.scrollTo(0, 0);")
            
        except Exception as e:
            logger.error(f"Error auto-scrolling: {e}")
    
    async def _take_screenshot(self, url: str) -> str:
        """Take a screenshot of the current page"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            filepath = Path(self.config.screenshot_directory) / filename
            
            self.driver.save_screenshot(str(filepath))
            
            self.stats['screenshots_taken'] += 1
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return None
    
    async def find_elements(self, selector: str, by: str = "css") -> List[Dict[str, Any]]:
        """Find elements on the page"""
        try:
            # Convert by parameter to Selenium By
            by_mapping = {
                "css": By.CSS_SELECTOR,
                "xpath": By.XPATH,
                "id": By.ID,
                "name": By.NAME,
                "class": By.CLASS_NAME,
                "tag": By.TAG_NAME
            }
            
            by_type = by_mapping.get(by, By.CSS_SELECTOR)
            
            # Wait for elements to be present
            WebDriverWait(self.driver, self.config.element_wait_timeout).until(
                EC.presence_of_element_located((by_type, selector))
            )
            
            # Find all elements
            elements = self.driver.find_elements(by_type, selector)
            
            # Extract element information
            element_info = []
            for element in elements:
                try:
                    info = {
                        'tag_name': element.tag_name,
                        'text': element.text,
                        'attributes': {},
                        'is_displayed': element.is_displayed(),
                        'is_enabled': element.is_enabled(),
                        'location': element.location,
                        'size': element.size
                    }
                    
                    # Get common attributes
                    common_attrs = ['id', 'class', 'href', 'src', 'alt', 'title']
                    for attr in common_attrs:
                        value = element.get_attribute(attr)
                        if value:
                            info['attributes'][attr] = value
                    
                    element_info.append(info)
                    
                except Exception as e:
                    logger.warning(f"Error extracting element info: {e}")
            
            return element_info
            
        except TimeoutException:
            logger.warning(f"Elements not found: {selector}")
            return []
        except Exception as e:
            logger.error(f"Error finding elements: {e}")
            return []
    
    async def click_element(self, selector: str, by: str = "css") -> bool:
        """Click an element on the page"""
        try:
            by_mapping = {
                "css": By.CSS_SELECTOR,
                "xpath": By.XPATH,
                "id": By.ID,
                "name": By.NAME,
                "class": By.CLASS_NAME,
                "tag": By.TAG_NAME
            }
            
            by_type = by_mapping.get(by, By.CSS_SELECTOR)
            
            # Wait for element to be clickable
            element = WebDriverWait(self.driver, self.config.element_wait_timeout).until(
                EC.element_to_be_clickable((by_type, selector))
            )
            
            # Scroll to element
            self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
            
            # Click element
            element.click()
            
            # Wait after click
            await asyncio.sleep(1)
            
            self.stats['actions_performed'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error clicking element {selector}: {e}")
            return False
    
    async def fill_form_field(self, selector: str, value: str, by: str = "css") -> bool:
        """Fill a form field with a value"""
        try:
            by_mapping = {
                "css": By.CSS_SELECTOR,
                "xpath": By.XPATH,
                "id": By.ID,
                "name": By.NAME,
                "class": By.CLASS_NAME,
                "tag": By.TAG_NAME
            }
            
            by_type = by_mapping.get(by, By.CSS_SELECTOR)
            
            # Wait for element to be present
            element = WebDriverWait(self.driver, self.config.element_wait_timeout).until(
                EC.presence_of_element_located((by_type, selector))
            )
            
            # Clear existing value
            element.clear()
            
            # Type new value
            element.send_keys(value)
            
            self.stats['actions_performed'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error filling form field {selector}: {e}")
            return False
    
    async def execute_javascript(self, script: str) -> Any:
        """Execute JavaScript on the page"""
        try:
            result = self.driver.execute_script(script)
            return result
            
        except Exception as e:
            logger.error(f"Error executing JavaScript: {e}")
            return None
    
    async def wait_for_element(self, selector: str, by: str = "css", timeout: int = None) -> bool:
        """Wait for an element to appear on the page"""
        try:
            timeout = timeout or self.config.element_wait_timeout
            
            by_mapping = {
                "css": By.CSS_SELECTOR,
                "xpath": By.XPATH,
                "id": By.ID,
                "name": By.NAME,
                "class": By.CLASS_NAME,
                "tag": By.TAG_NAME
            }
            
            by_type = by_mapping.get(by, By.CSS_SELECTOR)
            
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by_type, selector))
            )
            
            return True
            
        except TimeoutException:
            logger.warning(f"Element not found within timeout: {selector}")
            return False
        except Exception as e:
            logger.error(f"Error waiting for element: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get automation statistics"""
        runtime = datetime.now() - self.stats['start_time']
        
        return {
            **self.stats,
            'runtime_seconds': runtime.total_seconds(),
            'pages_per_minute': self.stats['pages_visited'] / max(1, runtime.total_seconds() / 60),
            'actions_per_page': self.stats['actions_performed'] / max(1, self.stats['pages_visited']),
            'error_rate': self.stats['errors_encountered'] / max(1, self.stats['pages_visited'] + self.stats['actions_performed'])
        }
    
    async def close(self):
        """Close the browser"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Browser closed")
            except Exception as e:
                logger.error(f"Error closing browser: {e}")
        
        self.driver = None
        self.session_id = None

class WebAutomation:
    """
    High-level web automation interface for the learning pipeline
    """
    
    def __init__(self, config: AutomationConfig = None):
        self.config = config or AutomationConfig()
        self.browser_manager = BrowserManager(self.config)
        self.is_initialized = False
        
        # Task queue for automation tasks
        self.task_queue = asyncio.Queue()
        self.active_tasks = []
        
        logger.info("WebAutomation initialized")
    
    async def initialize(self):
        """Initialize the automation system"""
        if not self.is_initialized:
            await self.browser_manager.initialize()
            self.is_initialized = True
            logger.info("Web automation system initialized")
    
    async def automate_page_interaction(self, url: str, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Automate interactions with a web page
        
        Args:
            url: URL to navigate to
            interactions: List of interactions to perform
            
        Returns:
            Results of the automation
        """
        if not self.is_initialized:
            await self.initialize()
        
        results = {
            'url': url,
            'interactions_performed': 0,
            'elements_found': [],
            'page_info': None,
            'errors': []
        }
        
        try:
            # Navigate to the page
            page_info = await self.browser_manager.navigate_to_url(url)
            results['page_info'] = page_info
            
            # Perform interactions
            for interaction in interactions:
                try:
                    interaction_type = interaction.get('type')
                    
                    if interaction_type == 'click':
                        success = await self.browser_manager.click_element(
                            interaction['selector'],
                            interaction.get('by', 'css')
                        )
                        if success:
                            results['interactions_performed'] += 1
                    
                    elif interaction_type == 'fill':
                        success = await self.browser_manager.fill_form_field(
                            interaction['selector'],
                            interaction['value'],
                            interaction.get('by', 'css')
                        )
                        if success:
                            results['interactions_performed'] += 1
                    
                    elif interaction_type == 'find':
                        elements = await self.browser_manager.find_elements(
                            interaction['selector'],
                            interaction.get('by', 'css')
                        )
                        results['elements_found'].extend(elements)
                    
                    elif interaction_type == 'wait':
                        success = await self.browser_manager.wait_for_element(
                            interaction['selector'],
                            interaction.get('by', 'css'),
                            interaction.get('timeout')
                        )
                    
                    elif interaction_type == 'javascript':
                        result = await self.browser_manager.execute_javascript(
                            interaction['script']
                        )
                        results[f'js_result_{len(results)}'] = result
                    
                    # Add delay between interactions
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    error_msg = f"Error in interaction {interaction_type}: {e}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            logger.info(f"Completed automation for {url}")
            
        except Exception as e:
            error_msg = f"Error in page automation: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results
    
    async def extract_dynamic_content(self, url: str, wait_conditions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract content from JavaScript-heavy pages
        
        Args:
            url: URL to extract content from
            wait_conditions: Conditions to wait for before extracting
            
        Returns:
            Extracted content
        """
        if not self.is_initialized:
            await self.initialize()
        
        results = {
            'url': url,
            'content': None,
            'elements': [],
            'page_info': None,
            'errors': []
        }
        
        try:
            # Navigate to the page
            page_info = await self.browser_manager.navigate_to_url(url)
            results['page_info'] = page_info
            
            # Wait for specified conditions
            if wait_conditions:
                for condition in wait_conditions:
                    if condition.get('type') == 'element':
                        await self.browser_manager.wait_for_element(
                            condition['selector'],
                            condition.get('by', 'css'),
                            condition.get('timeout')
                        )
                    elif condition.get('type') == 'javascript':
                        # Wait for JavaScript condition
                        script = condition['script']
                        timeout = condition.get('timeout', 10)
                        
                        for _ in range(timeout):
                            result = await self.browser_manager.execute_javascript(script)
                            if result:
                                break
                            await asyncio.sleep(1)
            
            # Extract content
            results['content'] = page_info.get('page_source')
            
            # Find specific elements if requested
            if 'extract_elements' in results:
                for element_config in results['extract_elements']:
                    elements = await self.browser_manager.find_elements(
                        element_config['selector'],
                        element_config.get('by', 'css')
                    )
                    results['elements'].extend(elements)
            
            logger.info(f"Extracted dynamic content from {url}")
            
        except Exception as e:
            error_msg = f"Error extracting dynamic content: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get automation statistics"""
        return self.browser_manager.get_statistics()
    
    async def shutdown(self):
        """Shutdown the automation system"""
        await self.browser_manager.close()
        self.is_initialized = False
        logger.info("Web automation system shutdown")

# Factory function
def create_web_automation(config: AutomationConfig = None) -> WebAutomation:
    """Create a configured WebAutomation instance"""
    return WebAutomation(config)

# Example usage
if __name__ == "__main__":
    async def main():
        # Only run if Selenium is available
        if not SELENIUM_AVAILABLE:
            print("Selenium not available. Install with: pip install selenium")
            return
        
        # Create automation config
        config = AutomationConfig(
            browser_type="chrome",
            headless=True,
            take_screenshots=True
        )
        
        automation = create_web_automation(config)
        
        try:
            # Initialize automation
            await automation.initialize()
            
            # Example: Extract content from a JavaScript-heavy page
            results = await automation.extract_dynamic_content(
                "https://example.com",
                wait_conditions=[
                    {
                        'type': 'element',
                        'selector': 'body',
                        'timeout': 10
                    }
                ]
            )
            
            print(f"Extracted content from {results['url']}")
            print(f"Page title: {results['page_info']['title']}")
            
            # Example: Automate interactions
            interactions = [
                {
                    'type': 'find',
                    'selector': 'a',
                    'by': 'css'
                },
                {
                    'type': 'click',
                    'selector': 'a',
                    'by': 'css'
                }
            ]
            
            interaction_results = await automation.automate_page_interaction(
                "https://example.com",
                interactions
            )
            
            print(f"Performed {interaction_results['interactions_performed']} interactions")
            
            # Get statistics
            stats = automation.get_statistics()
            print(f"Pages visited: {stats['pages_visited']}")
            print(f"Actions performed: {stats['actions_performed']}")
            
        finally:
            await automation.shutdown()
    
    asyncio.run(main())