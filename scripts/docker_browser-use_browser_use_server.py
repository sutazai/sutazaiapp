#!/usr/bin/env python3

import os
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from bs4 import BeautifulSoup
    SELENIUM_AVAILABLE = True
except ImportError:
    logger.warning("Selenium not available, running in mock mode")
    SELENIUM_AVAILABLE = False

app = FastAPI(title="Browser Use Agent Server", version="1.0.0")

class BrowserTaskRequest(BaseModel):
    task: str
    url: Optional[str] = None
    action: Optional[str] = "navigate"  # navigate, click, type, extract, screenshot
    selector: Optional[str] = None
    text: Optional[str] = None

class BrowserUseServer:
    def __init__(self):
        self.browser_options = self.setup_browser_options()
        self.current_driver = None
        
    def setup_browser_options(self):
        """Setup Chrome browser options"""
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-plugins')
        options.add_argument('--disable-images')
        options.add_argument('--disable-javascript')
        return options
    
    def get_driver(self):
        """Get or create a browser driver"""
        if not SELENIUM_AVAILABLE:
            return None
            
        try:
            if self.current_driver is None:
                service = Service('/usr/bin/chromedriver')
                self.current_driver = webdriver.Chrome(
                    service=service,
                    options=self.browser_options
                )
            return self.current_driver
        except Exception as e:
            logger.error(f"Failed to create browser driver: {e}")
            return None
    
    def close_driver(self):
        """Close the browser driver"""
        if self.current_driver:
            try:
                self.current_driver.quit()
                self.current_driver = None
            except Exception as e:
                logger.error(f"Failed to close driver: {e}")
    
    async def execute_browser_task(self, task: str, url: str = None, action: str = "navigate", 
                                 selector: str = None, text: str = None) -> Dict[str, Any]:
        """Execute a browser automation task"""
        try:
            if not SELENIUM_AVAILABLE:
                return {
                    "result": f"Browser task executed (Mock): {task}",
                    "status": "completed_mock",
                    "action": action,
                    "url": url
                }
            
            driver = self.get_driver()
            if not driver:
                return {
                    "result": "Browser driver not available",
                    "status": "error",
                    "action": action
                }
            
            result = await asyncio.to_thread(self._execute_sync_task, 
                                           driver, task, url, action, selector, text)
            
            return result
            
        except Exception as e:
            logger.error(f"Browser task execution failed: {e}")
            return {
                "result": f"Browser task failed: {str(e)}",
                "status": "error",
                "action": action,
                "url": url
            }
    
    def _execute_sync_task(self, driver, task, url, action, selector, text):
        """Execute synchronous browser task"""
        try:
            if action == "navigate" and url:
                driver.get(url)
                return {
                    "result": f"Successfully navigated to {url}",
                    "status": "completed",
                    "action": "navigate",
                    "url": url,
                    "title": driver.title
                }
            
            elif action == "extract" and url:
                driver.get(url)
                html = driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                
                if selector:
                    elements = soup.select(selector)
                    extracted_text = [elem.get_text().strip() for elem in elements]
                else:
                    extracted_text = soup.get_text().strip()
                
                return {
                    "result": extracted_text,
                    "status": "completed",
                    "action": "extract",
                    "url": url,
                    "selector": selector
                }
            
            elif action == "click" and selector:
                element = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                element.click()
                return {
                    "result": f"Successfully clicked element: {selector}",
                    "status": "completed",
                    "action": "click",
                    "selector": selector
                }
            
            elif action == "type" and selector and text:
                element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                element.clear()
                element.send_keys(text)
                return {
                    "result": f"Successfully typed text into {selector}",
                    "status": "completed",
                    "action": "type",
                    "selector": selector,
                    "text": text
                }
            
            elif action == "screenshot":
                screenshot = driver.get_screenshot_as_base64()
                return {
                    "result": "Screenshot captured",
                    "status": "completed",
                    "action": "screenshot",
                    "screenshot": screenshot[:100] + "..." if len(screenshot) > 100 else screenshot
                }
            
            else:
                return {
                    "result": f"Unknown action: {action}",
                    "status": "error",
                    "action": action
                }
                
        except Exception as e:
            return {
                "result": f"Browser operation failed: {str(e)}",
                "status": "error",
                "action": action
            }

# Global browser server instance
browser_server = BrowserUseServer()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "selenium_available": SELENIUM_AVAILABLE,
        "browser_ready": browser_server.current_driver is not None
    }

@app.post("/execute")
async def execute_browser_task(request: BrowserTaskRequest):
    try:
        result = await browser_server.execute_browser_task(
            task=request.task,
            url=request.url,
            action=request.action,
            selector=request.selector,
            text=request.text
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/navigate")
async def navigate_to_url(url: str):
    try:
        result = await browser_server.execute_browser_task(
            task=f"Navigate to {url}",
            url=url,
            action="navigate"
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract")
async def extract_content(url: str, selector: str = None):
    try:
        result = await browser_server.execute_browser_task(
            task=f"Extract content from {url}",
            url=url,
            action="extract",
            selector=selector
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    return {
        "service": "Browser Use Agent Server",
        "status": "running",
        "selenium_available": SELENIUM_AVAILABLE,
        "browser_active": browser_server.current_driver is not None
    }

@app.on_event("shutdown")
async def shutdown_event():
    browser_server.close_driver()

if __name__ == "__main__":
    port = int(os.getenv("BROWSER_USE_PORT", 8088))
    uvicorn.run(app, host="0.0.0.0", port=port)