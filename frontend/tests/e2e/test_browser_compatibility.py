"""
Browser Compatibility Testing Suite for JARVIS Streamlit Frontend

Tests cross-browser compatibility across:
- Chrome (latest and -2 versions)
- Firefox (latest and ESR)
- Safari (latest)
- Edge (latest)
- Mobile browsers (Chrome Mobile, Safari iOS)

Focus areas:
1. CSS rendering (gradients, animations, flexbox/grid)
2. JavaScript compatibility
3. WebSocket support
4. Audio/voice features
5. Streamlit component rendering
6. Custom CSS variables
7. Performance variations
8. Mobile viewports
9. Browser-specific bugs
10. Progressive enhancement
"""

import pytest
import asyncio
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from playwright.async_api import async_playwright, Browser, Page, BrowserContext
from playwright.sync_api import sync_playwright
import pandas as pd
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:11000"  # Streamlit frontend URL
SCREENSHOT_DIR = Path("/opt/sutazaiapp/frontend/tests/screenshots")
REPORT_DIR = Path("/opt/sutazaiapp/frontend/tests/reports")

# Ensure directories exist
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class BrowserConfig:
    """Browser configuration for testing"""
    name: str
    channel: Optional[str] = None
    device: Optional[str] = None
    viewport: Optional[Dict[str, int]] = None
    user_agent: Optional[str] = None
    is_mobile: bool = False

@dataclass
class TestResult:
    """Test result for a specific browser/feature combination"""
    browser: str
    feature: str
    status: str  # pass, fail, partial
    issues: List[str]
    screenshots: List[str]
    performance_metrics: Dict[str, float]
    timestamp: datetime

class BrowserCompatibilityTester:
    """Comprehensive browser compatibility testing"""
    
    def __init__(self):
        self.browsers = self._get_browser_configs()
        self.test_results = []
        self.compatibility_matrix = {}
        
    def _get_browser_configs(self) -> List[BrowserConfig]:
        """Get browser configurations for testing"""
        return [
            # Desktop browsers
            BrowserConfig(name="chromium", channel="chrome"),
            BrowserConfig(name="chromium", channel="chrome-beta"),
            BrowserConfig(name="chromium", channel="chrome-dev"),
            BrowserConfig(name="firefox"),
            BrowserConfig(name="webkit"),  # Safari on macOS
            BrowserConfig(name="chromium", channel="msedge"),
            
            # Mobile browsers
            BrowserConfig(
                name="chromium",
                device="Pixel 5",
                is_mobile=True
            ),
            BrowserConfig(
                name="chromium",
                device="Galaxy S20",
                is_mobile=True
            ),
            BrowserConfig(
                name="webkit",
                device="iPhone 12",
                is_mobile=True
            ),
            BrowserConfig(
                name="webkit",
                device="iPad Pro",
                is_mobile=True
            ),
        ]
    
    async def run_all_tests(self):
        """Run all browser compatibility tests"""
        async with async_playwright() as p:
            for browser_config in self.browsers:
                try:
                    await self._test_browser(p, browser_config)
                except Exception as e:
                    print(f"Error testing {browser_config.name}: {e}")
                    self.test_results.append(TestResult(
                        browser=self._get_browser_name(browser_config),
                        feature="browser_launch",
                        status="fail",
                        issues=[str(e)],
                        screenshots=[],
                        performance_metrics={},
                        timestamp=datetime.now()
                    ))
        
        # Generate reports
        self._generate_compatibility_matrix()
        self._generate_html_report()
        self._generate_markdown_report()
    
    def _get_browser_name(self, config: BrowserConfig) -> str:
        """Get formatted browser name"""
        if config.device:
            return f"{config.name}_{config.device.replace(' ', '_')}"
        elif config.channel:
            return f"{config.name}_{config.channel}"
        else:
            return config.name
    
    async def _test_browser(self, playwright, config: BrowserConfig):
        """Test a specific browser configuration"""
        browser_name = self._get_browser_name(config)
        print(f"\n{'='*60}")
        print(f"Testing: {browser_name}")
        print(f"{'='*60}")
        
        # Launch browser
        browser_type = getattr(playwright, config.name)
        launch_options = {"headless": True}
        
        if config.channel:
            launch_options["channel"] = config.channel
        
        browser = await browser_type.launch(**launch_options)
        
        # Create context with device emulation if needed
        context_options = {}
        if config.device:
            device = playwright.devices[config.device]
            context_options.update(device)
        elif config.viewport:
            context_options["viewport"] = config.viewport
        
        context = await browser.new_context(**context_options)
        page = await context.new_page()
        
        # Enable console message capture
        console_messages = []
        page.on("console", lambda msg: console_messages.append({
            "type": msg.type,
            "text": msg.text,
            "location": msg.location
        }))
        
        # Enable error capture
        page_errors = []
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        
        try:
            # Navigate to the app
            await page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
            
            # Run all feature tests
            await self._test_css_rendering(page, browser_name)
            await self._test_javascript_compatibility(page, browser_name)
            await self._test_websocket_support(page, browser_name)
            await self._test_audio_features(page, browser_name)
            await self._test_streamlit_components(page, browser_name)
            await self._test_css_variables(page, browser_name)
            await self._test_performance(page, browser_name)
            await self._test_mobile_viewport(page, browser_name, config.is_mobile)
            await self._test_animations(page, browser_name)
            await self._test_progressive_enhancement(page, browser_name)
            
            # Check for console errors
            js_errors = [msg for msg in console_messages if msg["type"] == "error"]
            if js_errors:
                self.test_results.append(TestResult(
                    browser=browser_name,
                    feature="javascript_errors",
                    status="fail",
                    issues=[f"{err['text']} at {err['location']}" for err in js_errors],
                    screenshots=[],
                    performance_metrics={},
                    timestamp=datetime.now()
                ))
            
        finally:
            await browser.close()
    
    async def _test_css_rendering(self, page: Page, browser_name: str):
        """Test CSS rendering capabilities"""
        print(f"  Testing CSS rendering...")
        issues = []
        screenshots = []
        
        try:
            # Test CSS gradients
            gradient_element = await page.query_selector('.stApp')
            if gradient_element:
                gradient_style = await gradient_element.evaluate('''
                    el => window.getComputedStyle(el).background
                ''')
                if 'linear-gradient' not in gradient_style and 'gradient' not in gradient_style:
                    issues.append("CSS gradient not rendering properly")
            
            # Test flexbox
            flexbox_test = await page.evaluate('''
                () => {
                    const testEl = document.createElement('div');
                    testEl.style.display = 'flex';
                    document.body.appendChild(testEl);
                    const computed = window.getComputedStyle(testEl).display;
                    document.body.removeChild(testEl);
                    return computed === 'flex';
                }
            ''')
            if not flexbox_test:
                issues.append("Flexbox not supported")
            
            # Test CSS Grid
            grid_test = await page.evaluate('''
                () => {
                    const testEl = document.createElement('div');
                    testEl.style.display = 'grid';
                    document.body.appendChild(testEl);
                    const computed = window.getComputedStyle(testEl).display;
                    document.body.removeChild(testEl);
                    return computed === 'grid';
                }
            ''')
            if not grid_test:
                issues.append("CSS Grid not supported")
            
            # Test custom properties (CSS variables)
            css_var_test = await page.evaluate('''
                () => {
                    const root = document.documentElement;
                    const jarvisPrimary = getComputedStyle(root)
                        .getPropertyValue('--jarvis-primary').trim();
                    return jarvisPrimary === '#00D4FF';
                }
            ''')
            if not css_var_test:
                issues.append("CSS custom properties not working correctly")
            
            # Test box-shadow
            arc_reactor = await page.query_selector('.arc-reactor')
            if arc_reactor:
                box_shadow = await arc_reactor.evaluate('''
                    el => window.getComputedStyle(el).boxShadow
                ''')
                if box_shadow == 'none' or not box_shadow:
                    issues.append("Box-shadow effects not rendering")
            
            # Take screenshot
            screenshot_path = SCREENSHOT_DIR / f"{browser_name}_css_rendering.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            screenshots.append(str(screenshot_path))
            
            status = "pass" if not issues else "partial" if len(issues) < 3 else "fail"
            
        except Exception as e:
            issues.append(f"CSS test error: {str(e)}")
            status = "fail"
        
        self.test_results.append(TestResult(
            browser=browser_name,
            feature="css_rendering",
            status=status,
            issues=issues,
            screenshots=screenshots,
            performance_metrics={},
            timestamp=datetime.now()
        ))
    
    async def _test_javascript_compatibility(self, page: Page, browser_name: str):
        """Test JavaScript feature compatibility"""
        print(f"  Testing JavaScript compatibility...")
        issues = []
        
        try:
            # Test ES6+ features
            es6_tests = await page.evaluate('''
                () => {
                    const tests = {};
                    
                    // Arrow functions
                    try {
                        const arrow = () => true;
                        tests.arrow_functions = arrow();
                    } catch { tests.arrow_functions = false; }
                    
                    // Template literals
                    try {
                        const str = `test`;
                        tests.template_literals = true;
                    } catch { tests.template_literals = false; }
                    
                    // Destructuring
                    try {
                        const {a, b} = {a: 1, b: 2};
                        tests.destructuring = true;
                    } catch { tests.destructuring = false; }
                    
                    // Async/await
                    try {
                        const testAsync = async () => await Promise.resolve(true);
                        tests.async_await = true;
                    } catch { tests.async_await = false; }
                    
                    // Optional chaining
                    try {
                        const obj = {};
                        const val = obj?.prop?.nested;
                        tests.optional_chaining = true;
                    } catch { tests.optional_chaining = false; }
                    
                    // Nullish coalescing
                    try {
                        const val = null ?? 'default';
                        tests.nullish_coalescing = true;
                    } catch { tests.nullish_coalescing = false; }
                    
                    return tests;
                }
            ''')
            
            for feature, supported in es6_tests.items():
                if not supported:
                    issues.append(f"ES6 feature not supported: {feature}")
            
            # Test Web APIs
            api_tests = await page.evaluate('''
                () => {
                    const tests = {};
                    
                    tests.fetch = typeof fetch !== 'undefined';
                    tests.localStorage = typeof localStorage !== 'undefined';
                    tests.sessionStorage = typeof sessionStorage !== 'undefined';
                    tests.webSocket = typeof WebSocket !== 'undefined';
                    tests.promise = typeof Promise !== 'undefined';
                    tests.map = typeof Map !== 'undefined';
                    tests.set = typeof Set !== 'undefined';
                    tests.symbol = typeof Symbol !== 'undefined';
                    
                    return tests;
                }
            ''')
            
            for api, available in api_tests.items():
                if not available:
                    issues.append(f"Web API not available: {api}")
            
            status = "pass" if not issues else "partial" if len(issues) < 3 else "fail"
            
        except Exception as e:
            issues.append(f"JavaScript test error: {str(e)}")
            status = "fail"
        
        self.test_results.append(TestResult(
            browser=browser_name,
            feature="javascript_compatibility",
            status=status,
            issues=issues,
            screenshots=[],
            performance_metrics={},
            timestamp=datetime.now()
        ))
    
    async def _test_websocket_support(self, page: Page, browser_name: str):
        """Test WebSocket support and functionality"""
        print(f"  Testing WebSocket support...")
        issues = []
        
        try:
            ws_test = await page.evaluate('''
                () => {
                    return new Promise((resolve) => {
                        const results = {
                            supported: typeof WebSocket !== 'undefined',
                            connection: false,
                            binaryType: false,
                            events: false
                        };
                        
                        if (!results.supported) {
                            resolve(results);
                            return;
                        }
                        
                        try {
                            // Test WebSocket creation
                            const ws = new WebSocket('ws://localhost:10200/ws');
                            results.connection = true;
                            
                            // Test binary type support
                            results.binaryType = 'binaryType' in ws;
                            
                            // Test event handlers
                            results.events = 'onopen' in ws && 'onclose' in ws && 
                                           'onmessage' in ws && 'onerror' in ws;
                            
                            ws.close();
                        } catch (e) {
                            // Connection failure is ok, we're testing API support
                        }
                        
                        resolve(results);
                    });
                }
            ''')
            
            if not ws_test['supported']:
                issues.append("WebSocket API not supported")
            if not ws_test['binaryType']:
                issues.append("WebSocket binary type not supported")
            if not ws_test['events']:
                issues.append("WebSocket event handlers not fully supported")
            
            status = "pass" if not issues else "partial" if len(issues) < 2 else "fail"
            
        except Exception as e:
            issues.append(f"WebSocket test error: {str(e)}")
            status = "fail"
        
        self.test_results.append(TestResult(
            browser=browser_name,
            feature="websocket_support",
            status=status,
            issues=issues,
            screenshots=[],
            performance_metrics={},
            timestamp=datetime.now()
        ))
    
    async def _test_audio_features(self, page: Page, browser_name: str):
        """Test audio and voice recording features"""
        print(f"  Testing audio features...")
        issues = []
        
        try:
            audio_test = await page.evaluate('''
                () => {
                    const results = {
                        audioContext: false,
                        getUserMedia: false,
                        mediaRecorder: false,
                        audioElement: false,
                        webAudio: false
                    };
                    
                    // Test AudioContext
                    results.audioContext = typeof (window.AudioContext || 
                                                  window.webkitAudioContext) !== 'undefined';
                    
                    // Test getUserMedia
                    results.getUserMedia = !!(navigator.mediaDevices && 
                                             navigator.mediaDevices.getUserMedia);
                    
                    // Test MediaRecorder
                    results.mediaRecorder = typeof MediaRecorder !== 'undefined';
                    
                    // Test audio element
                    try {
                        const audio = document.createElement('audio');
                        results.audioElement = audio.canPlayType && 
                                              audio.canPlayType('audio/wav') !== '';
                    } catch {}
                    
                    // Test Web Audio API
                    try {
                        const AudioContext = window.AudioContext || window.webkitAudioContext;
                        if (AudioContext) {
                            const ctx = new AudioContext();
                            results.webAudio = !!ctx.createOscillator;
                            ctx.close();
                        }
                    } catch {}
                    
                    return results;
                }
            ''')
            
            if not audio_test['audioContext']:
                issues.append("AudioContext not supported")
            if not audio_test['getUserMedia']:
                issues.append("getUserMedia not supported (mic recording will fail)")
            if not audio_test['mediaRecorder']:
                issues.append("MediaRecorder not supported")
            if not audio_test['audioElement']:
                issues.append("Audio element WAV playback not supported")
            if not audio_test['webAudio']:
                issues.append("Web Audio API not fully supported")
            
            status = "pass" if not issues else "partial" if len(issues) < 3 else "fail"
            
        except Exception as e:
            issues.append(f"Audio test error: {str(e)}")
            status = "fail"
        
        self.test_results.append(TestResult(
            browser=browser_name,
            feature="audio_features",
            status=status,
            issues=issues,
            screenshots=[],
            performance_metrics={},
            timestamp=datetime.now()
        ))
    
    async def _test_streamlit_components(self, page: Page, browser_name: str):
        """Test Streamlit-specific components"""
        print(f"  Testing Streamlit components...")
        issues = []
        screenshots = []
        
        try:
            # Wait for Streamlit to load
            await page.wait_for_selector('.stApp', timeout=10000)
            
            # Test chat interface
            chat_input = await page.query_selector('[data-testid="stChatInput"]')
            if not chat_input:
                chat_input = await page.query_selector('input[placeholder*="Type your message"]')
            if not chat_input:
                issues.append("Chat input component not found")
            
            # Test sidebar
            sidebar = await page.query_selector('[data-testid="stSidebar"]')
            if not sidebar:
                issues.append("Sidebar not rendered")
            
            # Test tabs
            tabs = await page.query_selector('[role="tablist"]')
            if not tabs:
                issues.append("Tab component not rendered")
            
            # Test metrics
            metrics = await page.query_selector('[data-testid="metric-container"]')
            if not metrics:
                issues.append("Metric components not rendered")
            
            # Test plotly charts
            plotly = await page.query_selector('.plotly')
            if not plotly:
                issues.append("Plotly charts not rendered")
            
            # Test buttons
            buttons = await page.query_selector_all('button')
            if len(buttons) < 5:
                issues.append(f"Expected multiple buttons, found {len(buttons)}")
            
            # Take component screenshot
            screenshot_path = SCREENSHOT_DIR / f"{browser_name}_streamlit_components.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            screenshots.append(str(screenshot_path))
            
            status = "pass" if not issues else "partial" if len(issues) < 3 else "fail"
            
        except Exception as e:
            issues.append(f"Streamlit component test error: {str(e)}")
            status = "fail"
        
        self.test_results.append(TestResult(
            browser=browser_name,
            feature="streamlit_components",
            status=status,
            issues=issues,
            screenshots=screenshots,
            performance_metrics={},
            timestamp=datetime.now()
        ))
    
    async def _test_css_variables(self, page: Page, browser_name: str):
        """Test CSS custom properties (variables)"""
        print(f"  Testing CSS variables...")
        issues = []
        
        try:
            css_vars = await page.evaluate('''
                () => {
                    const root = document.documentElement;
                    const computed = getComputedStyle(root);
                    
                    return {
                        primary: computed.getPropertyValue('--jarvis-primary').trim(),
                        secondary: computed.getPropertyValue('--jarvis-secondary').trim(),
                        accent: computed.getPropertyValue('--jarvis-accent').trim(),
                        dark: computed.getPropertyValue('--jarvis-dark').trim(),
                        light: computed.getPropertyValue('--jarvis-light').trim()
                    };
                }
            ''')
            
            expected = {
                'primary': '#00D4FF',
                'secondary': '#0099CC',
                'accent': '#FF6B6B',
                'dark': '#0A0E27',
                'light': '#E6F3FF'
            }
            
            for var_name, expected_value in expected.items():
                if css_vars[var_name] != expected_value:
                    issues.append(f"CSS variable --jarvis-{var_name} has value '{css_vars[var_name]}', expected '{expected_value}'")
            
            # Test CSS variable usage in styles
            var_usage = await page.evaluate('''
                () => {
                    const button = document.querySelector('.stButton > button');
                    if (!button) return false;
                    
                    const styles = getComputedStyle(button);
                    const background = styles.background || styles.backgroundColor;
                    
                    // Check if it contains the primary color
                    return background.includes('0, 212, 255') || 
                           background.includes('00D4FF');
                }
            ''')
            
            if not var_usage:
                issues.append("CSS variables not being used in component styles")
            
            status = "pass" if not issues else "partial" if len(issues) < 3 else "fail"
            
        except Exception as e:
            issues.append(f"CSS variables test error: {str(e)}")
            status = "fail"
        
        self.test_results.append(TestResult(
            browser=browser_name,
            feature="css_variables",
            status=status,
            issues=issues,
            screenshots=[],
            performance_metrics={},
            timestamp=datetime.now()
        ))
    
    async def _test_performance(self, page: Page, browser_name: str):
        """Test performance metrics"""
        print(f"  Testing performance...")
        issues = []
        
        try:
            # Get performance metrics
            metrics = await page.evaluate('''
                () => {
                    const perf = performance.timing;
                    const paint = performance.getEntriesByType('paint');
                    
                    return {
                        domContentLoaded: perf.domContentLoadedEventEnd - perf.navigationStart,
                        loadComplete: perf.loadEventEnd - perf.navigationStart,
                        firstPaint: paint.find(p => p.name === 'first-paint')?.startTime || 0,
                        firstContentfulPaint: paint.find(p => p.name === 'first-contentful-paint')?.startTime || 0,
                        memoryUsage: performance.memory ? performance.memory.usedJSHeapSize / 1048576 : 0
                    };
                }
            ''')
            
            # Check performance thresholds
            if metrics['firstContentfulPaint'] > 1800:
                issues.append(f"First Contentful Paint too slow: {metrics['firstContentfulPaint']}ms (target: <1800ms)")
            
            if metrics['domContentLoaded'] > 3000:
                issues.append(f"DOM Content Loaded too slow: {metrics['domContentLoaded']}ms (target: <3000ms)")
            
            if metrics['loadComplete'] > 5000:
                issues.append(f"Page load too slow: {metrics['loadComplete']}ms (target: <5000ms)")
            
            if metrics['memoryUsage'] > 100:
                issues.append(f"High memory usage: {metrics['memoryUsage']:.1f}MB")
            
            status = "pass" if not issues else "partial" if len(issues) < 2 else "fail"
            
            self.test_results.append(TestResult(
                browser=browser_name,
                feature="performance",
                status=status,
                issues=issues,
                screenshots=[],
                performance_metrics=metrics,
                timestamp=datetime.now()
            ))
            
        except Exception as e:
            issues.append(f"Performance test error: {str(e)}")
            self.test_results.append(TestResult(
                browser=browser_name,
                feature="performance",
                status="fail",
                issues=issues,
                screenshots=[],
                performance_metrics={},
                timestamp=datetime.now()
            ))
    
    async def _test_mobile_viewport(self, page: Page, browser_name: str, is_mobile: bool):
        """Test mobile viewport and touch interactions"""
        print(f"  Testing mobile viewport...")
        issues = []
        screenshots = []
        
        if not is_mobile:
            # Skip for desktop browsers
            return
        
        try:
            # Check viewport meta tag
            viewport = await page.evaluate('''
                () => {
                    const meta = document.querySelector('meta[name="viewport"]');
                    return meta ? meta.content : null;
                }
            ''')
            
            if not viewport:
                issues.append("No viewport meta tag found")
            elif 'width=device-width' not in viewport:
                issues.append("Viewport not set to device-width")
            
            # Check touch event support
            touch_support = await page.evaluate('''
                () => {
                    return 'ontouchstart' in window || 
                           navigator.maxTouchPoints > 0;
                }
            ''')
            
            if not touch_support:
                issues.append("Touch events not supported")
            
            # Check responsive layout
            viewport_size = await page.evaluate('''
                () => ({
                    width: window.innerWidth,
                    height: window.innerHeight
                })
            ''')
            
            if viewport_size['width'] > 768:
                issues.append(f"Mobile viewport width too large: {viewport_size['width']}px")
            
            # Test sidebar collapse on mobile
            sidebar = await page.query_selector('[data-testid="stSidebar"]')
            if sidebar:
                is_collapsed = await sidebar.evaluate('''
                    el => {
                        const styles = getComputedStyle(el);
                        return styles.display === 'none' || 
                               styles.visibility === 'hidden' ||
                               parseInt(styles.width) < 100;
                    }
                ''')
                if not is_collapsed and viewport_size['width'] < 768:
                    issues.append("Sidebar not collapsed on mobile")
            
            # Take mobile screenshot
            screenshot_path = SCREENSHOT_DIR / f"{browser_name}_mobile_viewport.png"
            await page.screenshot(path=str(screenshot_path), full_page=False)
            screenshots.append(str(screenshot_path))
            
            status = "pass" if not issues else "partial" if len(issues) < 2 else "fail"
            
        except Exception as e:
            issues.append(f"Mobile viewport test error: {str(e)}")
            status = "fail"
        
        self.test_results.append(TestResult(
            browser=browser_name,
            feature="mobile_viewport",
            status=status,
            issues=issues,
            screenshots=screenshots,
            performance_metrics={},
            timestamp=datetime.now()
        ))
    
    async def _test_animations(self, page: Page, browser_name: str):
        """Test CSS animations and transitions"""
        print(f"  Testing animations...")
        issues = []
        
        try:
            # Test animation support
            animation_test = await page.evaluate('''
                () => {
                    const results = {
                        cssAnimations: false,
                        cssTransitions: false,
                        requestAnimationFrame: false,
                        webAnimations: false
                    };
                    
                    // Test CSS animations
                    const testEl = document.createElement('div');
                    testEl.style.animation = 'test 1s';
                    results.cssAnimations = testEl.style.animation !== '';
                    
                    // Test CSS transitions
                    testEl.style.transition = 'all 0.3s';
                    results.cssTransitions = testEl.style.transition !== '';
                    
                    // Test requestAnimationFrame
                    results.requestAnimationFrame = typeof requestAnimationFrame === 'function';
                    
                    // Test Web Animations API
                    results.webAnimations = typeof Element.prototype.animate === 'function';
                    
                    return results;
                }
            ''')
            
            if not animation_test['cssAnimations']:
                issues.append("CSS animations not supported")
            if not animation_test['cssTransitions']:
                issues.append("CSS transitions not supported")
            if not animation_test['requestAnimationFrame']:
                issues.append("requestAnimationFrame not supported")
            
            # Test specific animations
            arc_reactor = await page.query_selector('.arc-reactor')
            if arc_reactor:
                animation_running = await arc_reactor.evaluate('''
                    el => {
                        const styles = getComputedStyle(el);
                        return styles.animationName !== 'none' && 
                               styles.animationName !== '';
                    }
                ''')
                if not animation_running:
                    issues.append("Arc reactor animation not running")
            
            # Test voice wave animation
            voice_wave = await page.query_selector('.voice-wave')
            if voice_wave:
                wave_animation = await voice_wave.evaluate('''
                    el => {
                        const spans = el.querySelectorAll('span');
                        if (spans.length === 0) return false;
                        
                        const firstSpan = spans[0];
                        const styles = getComputedStyle(firstSpan);
                        return styles.animationName === 'wave';
                    }
                ''')
                if not wave_animation:
                    issues.append("Voice wave animation not working")
            
            status = "pass" if not issues else "partial" if len(issues) < 2 else "fail"
            
        except Exception as e:
            issues.append(f"Animation test error: {str(e)}")
            status = "fail"
        
        self.test_results.append(TestResult(
            browser=browser_name,
            feature="animations",
            status=status,
            issues=issues,
            screenshots=[],
            performance_metrics={},
            timestamp=datetime.now()
        ))
    
    async def _test_progressive_enhancement(self, page: Page, browser_name: str):
        """Test progressive enhancement and fallbacks"""
        print(f"  Testing progressive enhancement...")
        issues = []
        
        try:
            # Test JavaScript disabled fallback
            # (We can't actually disable JS in running page, but we can check for noscript)
            noscript = await page.evaluate('''
                () => {
                    const noscript = document.querySelector('noscript');
                    return noscript ? noscript.textContent : null;
                }
            ''')
            
            if not noscript:
                issues.append("No <noscript> fallback content")
            
            # Test feature detection usage
            feature_detection = await page.evaluate('''
                () => {
                    // Check if the app uses feature detection
                    const scripts = Array.from(document.scripts);
                    const hasFeatureDetection = scripts.some(script => {
                        const content = script.textContent || '';
                        return content.includes('typeof') || 
                               content.includes('in window') ||
                               content.includes('supports');
                    });
                    return hasFeatureDetection;
                }
            ''')
            
            if not feature_detection:
                issues.append("No apparent feature detection in use")
            
            # Test graceful degradation for unsupported features
            fallbacks = await page.evaluate('''
                () => {
                    const results = {};
                    
                    // Check for polyfills
                    results.hasPolyfills = typeof Promise !== 'undefined' && 
                                          Promise.toString().includes('[native code]') === false;
                    
                    // Check for vendor prefixes
                    const styles = document.createElement('div').style;
                    results.hasVendorPrefixes = '-webkit-transform' in styles || 
                                               '-moz-transform' in styles;
                    
                    return results;
                }
            ''')
            
            # Test image format fallbacks
            image_formats = await page.evaluate('''
                () => {
                    const img = new Image();
                    const formats = {};
                    
                    // Test WebP support
                    formats.webp = img.src = 'data:image/webp;base64,UklGRiIAAABXRUJQVlA4IBYAAAAwAQCdASoBAAEADsD+JaQAA3AAAAAA';
                    
                    // Test AVIF support
                    formats.avif = false; // Most browsers don't support it yet
                    
                    return formats;
                }
            ''')
            
            status = "pass" if not issues else "partial"
            
        except Exception as e:
            issues.append(f"Progressive enhancement test error: {str(e)}")
            status = "fail"
        
        self.test_results.append(TestResult(
            browser=browser_name,
            feature="progressive_enhancement",
            status=status,
            issues=issues,
            screenshots=[],
            performance_metrics={},
            timestamp=datetime.now()
        ))
    
    def _generate_compatibility_matrix(self):
        """Generate browser compatibility matrix"""
        matrix = {}
        
        for result in self.test_results:
            if result.browser not in matrix:
                matrix[result.browser] = {}
            matrix[result.browser][result.feature] = {
                'status': result.status,
                'issues': result.issues
            }
        
        self.compatibility_matrix = matrix
        
        # Save as JSON
        json_path = REPORT_DIR / "compatibility_matrix.json"
        with open(json_path, 'w') as f:
            json.dump(matrix, f, indent=2, default=str)
        
        print(f"\nCompatibility matrix saved to: {json_path}")
    
    def _generate_html_report(self):
        """Generate HTML compatibility report"""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>JARVIS Browser Compatibility Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #0A0E27 0%, #1A1F3A 100%);
            color: #E6F3FF;
            padding: 20px;
            margin: 0;
        }
        h1 {
            color: #00D4FF;
            text-align: center;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        }
        .matrix {
            overflow-x: auto;
            margin: 20px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            overflow: hidden;
        }
        th, td {
            padding: 12px;
            text-align: center;
            border: 1px solid rgba(0, 212, 255, 0.2);
        }
        th {
            background: rgba(0, 212, 255, 0.1);
            color: #00D4FF;
            font-weight: 600;
        }
        .pass {
            background: rgba(76, 175, 80, 0.3);
            color: #4CAF50;
        }
        .partial {
            background: rgba(255, 193, 7, 0.3);
            color: #FFC107;
        }
        .fail {
            background: rgba(244, 67, 54, 0.3);
            color: #F44336;
        }
        .feature-details {
            margin: 20px 0;
            padding: 20px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            border: 1px solid rgba(0, 212, 255, 0.2);
        }
        .issues {
            margin-top: 10px;
            padding-left: 20px;
        }
        .issues li {
            color: #FF6B6B;
            margin: 5px 0;
        }
        .timestamp {
            text-align: center;
            color: #999;
            margin-top: 30px;
        }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .summary-card {
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid #00D4FF;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        .summary-card h3 {
            color: #00D4FF;
            margin: 0 0 10px 0;
        }
        .summary-card .value {
            font-size: 2em;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>🤖 JARVIS Browser Compatibility Report</h1>
    """
        
        # Add summary statistics
        total_tests = len(self.test_results)
        passed = len([r for r in self.test_results if r.status == "pass"])
        partial = len([r for r in self.test_results if r.status == "partial"])
        failed = len([r for r in self.test_results if r.status == "fail"])
        
        html += f"""
    <div class="summary">
        <div class="summary-card">
            <h3>Total Tests</h3>
            <div class="value">{total_tests}</div>
        </div>
        <div class="summary-card">
            <h3>Passed</h3>
            <div class="value pass">{passed}</div>
        </div>
        <div class="summary-card">
            <h3>Partial</h3>
            <div class="value partial">{partial}</div>
        </div>
        <div class="summary-card">
            <h3>Failed</h3>
            <div class="value fail">{failed}</div>
        </div>
    </div>
    """
        
        # Create compatibility matrix table
        if self.compatibility_matrix:
            features = set()
            for browser_results in self.compatibility_matrix.values():
                features.update(browser_results.keys())
            features = sorted(list(features))
            
            html += """
    <h2>Compatibility Matrix</h2>
    <div class="matrix">
        <table>
            <thead>
                <tr>
                    <th>Browser</th>
            """
            
            for feature in features:
                feature_display = feature.replace('_', ' ').title()
                html += f"<th>{feature_display}</th>"
            
            html += """
                </tr>
            </thead>
            <tbody>
            """
            
            for browser, results in sorted(self.compatibility_matrix.items()):
                html += f"<tr><td><strong>{browser}</strong></td>"
                for feature in features:
                    if feature in results:
                        status = results[feature]['status']
                        html += f'<td class="{status}">{status.upper()}</td>'
                    else:
                        html += '<td>-</td>'
                html += "</tr>"
            
            html += """
            </tbody>
        </table>
    </div>
            """
        
        # Add detailed issues
        html += "<h2>Detailed Issues by Browser</h2>"
        
        for browser in sorted(set(r.browser for r in self.test_results)):
            browser_results = [r for r in self.test_results if r.browser == browser]
            issues_count = sum(len(r.issues) for r in browser_results)
            
            if issues_count > 0:
                html += f"""
    <div class="feature-details">
        <h3>{browser}</h3>
        <ul class="issues">
                """
                
                for result in browser_results:
                    if result.issues:
                        html += f"<li><strong>{result.feature}:</strong><ul>"
                        for issue in result.issues:
                            html += f"<li>{issue}</li>"
                        html += "</ul></li>"
                
                html += """
        </ul>
    </div>
                """
        
        # Add timestamp
        html += f"""
    <div class="timestamp">
        Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</body>
</html>
        """
        
        # Save HTML report
        html_path = REPORT_DIR / "compatibility_report.html"
        with open(html_path, 'w') as f:
            f.write(html)
        
        print(f"HTML report saved to: {html_path}")
    
    def _generate_markdown_report(self):
        """Generate Markdown compatibility report"""
        md = "# JARVIS Browser Compatibility Report\n\n"
        md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Summary
        total_tests = len(self.test_results)
        passed = len([r for r in self.test_results if r.status == "pass"])
        partial = len([r for r in self.test_results if r.status == "partial"])
        failed = len([r for r in self.test_results if r.status == "fail"])
        
        md += "## Summary\n\n"
        md += f"- **Total Tests**: {total_tests}\n"
        md += f"- **Passed**: {passed} ✅\n"
        md += f"- **Partial**: {partial} ⚠️\n"
        md += f"- **Failed**: {failed} ❌\n\n"
        
        # Compatibility Matrix
        md += "## Browser Compatibility Matrix\n\n"
        
        if self.compatibility_matrix:
            features = set()
            for browser_results in self.compatibility_matrix.values():
                features.update(browser_results.keys())
            features = sorted(list(features))
            
            # Create table header
            md += "| Browser |"
            for feature in features:
                md += f" {feature.replace('_', ' ').title()} |"
            md += "\n"
            
            # Add separator
            md += "|---------|"
            for _ in features:
                md += "----------|"
            md += "\n"
            
            # Add data rows
            for browser, results in sorted(self.compatibility_matrix.items()):
                md += f"| **{browser}** |"
                for feature in features:
                    if feature in results:
                        status = results[feature]['status']
                        emoji = "✅" if status == "pass" else "⚠️" if status == "partial" else "❌"
                        md += f" {emoji} {status} |"
                    else:
                        md += " - |"
                md += "\n"
        
        # Detailed Issues
        md += "\n## Detailed Issues by Browser\n\n"
        
        for browser in sorted(set(r.browser for r in self.test_results)):
            browser_results = [r for r in self.test_results if r.browser == browser]
            issues_list = []
            
            for result in browser_results:
                if result.issues:
                    for issue in result.issues:
                        issues_list.append(f"- **{result.feature}**: {issue}")
            
            if issues_list:
                md += f"### {browser}\n\n"
                md += "\n".join(issues_list)
                md += "\n\n"
        
        # Browser-Specific Recommendations
        md += "## Browser-Specific Recommendations\n\n"
        
        recommendations = self._generate_recommendations()
        for browser, recs in recommendations.items():
            if recs:
                md += f"### {browser}\n\n"
                for rec in recs:
                    md += f"- {rec}\n"
                md += "\n"
        
        # Performance Metrics
        md += "## Performance Metrics\n\n"
        
        perf_results = [r for r in self.test_results if r.feature == "performance" and r.performance_metrics]
        if perf_results:
            md += "| Browser | FCP (ms) | DOM Loaded (ms) | Full Load (ms) | Memory (MB) |\n"
            md += "|---------|----------|-----------------|----------------|-------------|\n"
            
            for result in perf_results:
                metrics = result.performance_metrics
                md += f"| {result.browser} |"
                md += f" {metrics.get('firstContentfulPaint', 0):.0f} |"
                md += f" {metrics.get('domContentLoaded', 0):.0f} |"
                md += f" {metrics.get('loadComplete', 0):.0f} |"
                md += f" {metrics.get('memoryUsage', 0):.1f} |\n"
        
        # Save Markdown report
        md_path = REPORT_DIR / "compatibility_report.md"
        with open(md_path, 'w') as f:
            f.write(md)
        
        print(f"Markdown report saved to: {md_path}")
    
    def _generate_recommendations(self) -> Dict[str, List[str]]:
        """Generate browser-specific recommendations based on test results"""
        recommendations = {}
        
        for browser, results in self.compatibility_matrix.items():
            browser_recs = []
            
            # Check for specific issues
            if 'audio_features' in results and results['audio_features']['status'] != 'pass':
                issues = results['audio_features']['issues']
                if any('getUserMedia' in issue for issue in issues):
                    browser_recs.append("Add fallback for microphone recording using file upload")
                if any('MediaRecorder' in issue for issue in issues):
                    browser_recs.append("Implement polyfill for MediaRecorder API")
            
            if 'css_rendering' in results and results['css_rendering']['status'] != 'pass':
                issues = results['css_rendering']['issues']
                if any('gradient' in issue.lower() for issue in issues):
                    browser_recs.append("Add vendor prefixes for CSS gradients")
                if any('grid' in issue.lower() for issue in issues):
                    browser_recs.append("Provide flexbox fallback for CSS Grid layouts")
            
            if 'javascript_compatibility' in results and results['javascript_compatibility']['status'] != 'pass':
                issues = results['javascript_compatibility']['issues']
                if any('optional_chaining' in issue for issue in issues):
                    browser_recs.append("Add Babel transform for optional chaining operator")
                if any('nullish_coalescing' in issue for issue in issues):
                    browser_recs.append("Add Babel transform for nullish coalescing operator")
            
            if 'websocket_support' in results and results['websocket_support']['status'] != 'pass':
                browser_recs.append("Implement long-polling fallback for WebSocket connections")
            
            if 'animations' in results and results['animations']['status'] != 'pass':
                browser_recs.append("Add CSS animation vendor prefixes (-webkit-, -moz-)")
                browser_recs.append("Consider using CSS transforms instead of complex animations")
            
            if browser_recs:
                recommendations[browser] = browser_recs
        
        return recommendations


# Test runner
def main():
    """Run browser compatibility tests"""
    print("="*80)
    print("JARVIS Frontend Browser Compatibility Testing")
    print("="*80)
    
    tester = BrowserCompatibilityTester()
    
    # Run tests
    asyncio.run(tester.run_all_tests())
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    total = len(tester.test_results)
    passed = len([r for r in tester.test_results if r.status == "pass"])
    partial = len([r for r in tester.test_results if r.status == "partial"])
    failed = len([r for r in tester.test_results if r.status == "fail"])
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed} ({passed/total*100:.1f}%)")
    print(f"Partial: {partial} ({partial/total*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total*100:.1f}%)")
    
    print(f"\nReports saved to: {REPORT_DIR}")
    print(f"Screenshots saved to: {SCREENSHOT_DIR}")
    
    # Return exit code based on failures
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())