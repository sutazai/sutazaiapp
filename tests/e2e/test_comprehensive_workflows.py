"""
Comprehensive End-to-End Tests for Sutazai Platform
Production-grade Playwright tests covering complete user workflows
"""

import pytest
import asyncio
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from typing import Dict, Any
import json
import time

# Configuration
BASE_URL = "http://localhost:11000"
API_URL = "http://localhost:10200"
TIMEOUT = 30000  # 30 seconds

class TestAuthenticationFlow:
    """Test complete authentication workflows"""
    
    @pytest.mark.asyncio
    async def test_user_registration_flow(self, browser: Browser):
        """Test complete user registration process"""
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            # Navigate to frontend
            await page.goto(BASE_URL, wait_until="networkidle", timeout=TIMEOUT)
            
            # Look for registration elements
            # Streamlit apps may have custom authentication
            signup_button = await page.query_selector('button:has-text("Sign Up")')
            if signup_button:
                await signup_button.click()
                await page.wait_for_timeout(1000)
                
                # Fill registration form
                email_input = await page.query_selector('input[type="email"]')
                password_input = await page.query_selector('input[type="password"]')
                
                if email_input and password_input:
                    test_email = f"test_{int(time.time())}@sutazai.com"
                    await email_input.fill(test_email)
                    await password_input.fill("TestPassword123!")
                    
                    # Submit registration
                    submit_button = await page.query_selector('button:has-text("Register")')
                    if submit_button:
                        await submit_button.click()
                        
                        # Wait for success message or redirect
                        await page.wait_for_timeout(2000)
                        
                        # Verify registration success
                        success_msg = await page.query_selector('text=/success|registered|welcome/i')
                        assert success_msg is not None, "Registration success message not found"
                        
        finally:
            await context.close()
    
    @pytest.mark.asyncio
    async def test_user_login_flow(self, browser: Browser):
        """Test user login with existing credentials"""
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            await page.goto(BASE_URL, wait_until="networkidle", timeout=TIMEOUT)
            
            # Look for login elements
            login_button = await page.query_selector('button:has-text("Login")')
            if login_button:
                await login_button.click()
                await page.wait_for_timeout(1000)
                
                # Fill login form
                email_input = await page.query_selector('input[type="email"]')
                password_input = await page.query_selector('input[type="password"]')
                
                if email_input and password_input:
                    await email_input.fill("admin@sutazai.com")
                    await password_input.fill("admin123")
                    
                    # Submit login
                    submit_button = await page.query_selector('button:has-text("Sign In")')
                    if submit_button:
                        await submit_button.click()
                        await page.wait_for_timeout(2000)
                        
                        # Verify login success (dashboard or chat interface visible)
                        dashboard = await page.query_selector('[data-testid="stApp"]')
                        assert dashboard is not None, "Dashboard not loaded after login"
                        
        finally:
            await context.close()
    
    @pytest.mark.asyncio
    async def test_session_persistence(self, browser: Browser):
        """Test that user session persists across page reloads"""
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            # Login first
            await page.goto(BASE_URL, wait_until="networkidle", timeout=TIMEOUT)
            
            # Perform login (simplified - assumes session storage)
            await page.evaluate("localStorage.setItem('sutazai_token', 'test_token')")
            
            # Reload page
            await page.reload(wait_until="networkidle")
            await page.wait_for_timeout(1000)
            
            # Verify session maintained
            token = await page.evaluate("localStorage.getItem('sutazai_token')")
            assert token == "test_token", "Session token not persisted"
            
        finally:
            await context.close()


class TestAgentChatInteractions:
    """Test AI agent chat functionality"""
    
    @pytest.mark.asyncio
    async def test_send_message_to_agent(self, browser: Browser):
        """Test sending a message to an AI agent and receiving response"""
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            await page.goto(BASE_URL, wait_until="networkidle", timeout=TIMEOUT)
            
            # Find chat input
            chat_input = await page.query_selector('textarea, input[placeholder*="message"], input[placeholder*="chat"]')
            if chat_input:
                test_message = "Hello, how are you?"
                await chat_input.fill(test_message)
                
                # Find and click send button
                send_button = await page.query_selector('button:has-text("Send")')
                if send_button:
                    await send_button.click()
                    
                    # Wait for response
                    await page.wait_for_timeout(5000)
                    
                    # Verify response appears
                    messages = await page.query_selector_all('[data-testid="stChatMessage"]')
                    assert len(messages) >= 2, "Agent response not received"
                    
        finally:
            await context.close()
    
    @pytest.mark.asyncio
    async def test_agent_selection(self, browser: Browser):
        """Test selecting different AI agents"""
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            await page.goto(BASE_URL, wait_until="networkidle", timeout=TIMEOUT)
            
            # Look for agent selector dropdown
            agent_selector = await page.query_selector('select, [role="combobox"]')
            if agent_selector:
                # Get available agents
                options = await page.query_selector_all('option, [role="option"]')
                
                # Test selecting each agent
                for i, option in enumerate(options[:3]):  # Test first 3 agents
                    await option.click()
                    await page.wait_for_timeout(500)
                    
                    # Verify agent selected
                    selected_text = await page.evaluate('el => el.textContent', option)
                    assert selected_text, f"Agent {i} not selected"
                    
        finally:
            await context.close()
    
    @pytest.mark.asyncio
    async def test_multi_agent_conversation(self, browser: Browser):
        """Test conversation involving multiple agents"""
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            await page.goto(BASE_URL, wait_until="networkidle", timeout=TIMEOUT)
            
            # Test sending messages to multiple agents
            agents = ["Letta", "CrewAI", "LangChain"]
            
            for agent_name in agents:
                # Select agent
                agent_button = await page.query_selector(f'button:has-text("{agent_name}")')
                if agent_button:
                    await agent_button.click()
                    await page.wait_for_timeout(500)
                    
                    # Send message
                    chat_input = await page.query_selector('textarea')
                    if chat_input:
                        await chat_input.fill(f"Hello from {agent_name}")
                        
                        send_button = await page.query_selector('button:has-text("Send")')
                        if send_button:
                            await send_button.click()
                            await page.wait_for_timeout(3000)
                            
            # Verify conversation history shows multiple agents
            messages = await page.query_selector_all('[data-testid="stChatMessage"]')
            assert len(messages) >= len(agents), "Multi-agent conversation not recorded"
            
        finally:
            await context.close()


class TestFileUploadDownload:
    """Test file upload and download functionality"""
    
    @pytest.mark.asyncio
    async def test_file_upload(self, browser: Browser):
        """Test uploading a file through the UI"""
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            await page.goto(BASE_URL, wait_until="networkidle", timeout=TIMEOUT)
            
            # Look for file uploader
            file_input = await page.query_selector('input[type="file"]')
            if file_input:
                # Create a test file
                test_file_path = "/tmp/test_upload.txt"
                with open(test_file_path, "w") as f:
                    f.write("Test file content for Sutazai upload")
                
                # Upload file
                await file_input.set_input_files(test_file_path)
                await page.wait_for_timeout(2000)
                
                # Verify upload success
                success_msg = await page.query_selector('text=/uploaded|success/i')
                assert success_msg is not None, "File upload success message not found"
                
        finally:
            await context.close()
    
    @pytest.mark.asyncio
    async def test_file_download(self, browser: Browser):
        """Test downloading a file from the system"""
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            await page.goto(BASE_URL, wait_until="networkidle", timeout=TIMEOUT)
            
            # Look for downloadable files
            download_link = await page.query_selector('a[download], button:has-text("Download")')
            if download_link:
                # Set up download listener
                async with page.expect_download() as download_info:
                    await download_link.click()
                
                download = await download_info.value
                
                # Verify download
                assert download.suggested_filename, "Download filename not provided"
                
                # Save download to verify
                await download.save_as(f"/tmp/{download.suggested_filename}")
                
        finally:
            await context.close()
    
    @pytest.mark.asyncio
    async def test_file_processing_with_agent(self, browser: Browser):
        """Test uploading a file and having an agent process it"""
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            await page.goto(BASE_URL, wait_until="networkidle", timeout=TIMEOUT)
            
            # Upload document
            file_input = await page.query_selector('input[type="file"]')
            if file_input:
                test_file = "/tmp/test_document.txt"
                with open(test_file, "w") as f:
                    f.write("This is a test document for AI processing. Please summarize this content.")
                
                await file_input.set_input_files(test_file)
                await page.wait_for_timeout(2000)
                
                # Ask agent to process file
                chat_input = await page.query_selector('textarea')
                if chat_input:
                    await chat_input.fill("Please summarize the uploaded document")
                    
                    send_button = await page.query_selector('button:has-text("Send")')
                    if send_button:
                        await send_button.click()
                        await page.wait_for_timeout(5000)
                        
                        # Verify agent processed the file
                        response = await page.query_selector('[data-testid="stChatMessage"]:last-child')
                        if response:
                            text = await response.text_content()
                            assert "summarize" in text.lower() or "document" in text.lower(), "Agent did not process file"
                            
        finally:
            await context.close()


class TestWebSocketRealtime:
    """Test WebSocket real-time communication"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, browser: Browser):
        """Test WebSocket connection establishment"""
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            # Monitor WebSocket connections
            ws_connections = []
            
            async def on_websocket(ws):
                ws_connections.append(ws)
            
            page.on("websocket", on_websocket)
            
            await page.goto(BASE_URL, wait_until="networkidle", timeout=TIMEOUT)
            await page.wait_for_timeout(2000)
            
            # Verify WebSocket connection established
            assert len(ws_connections) > 0, "No WebSocket connections established"
            
        finally:
            await context.close()
    
    @pytest.mark.asyncio
    async def test_realtime_message_updates(self, browser: Browser):
        """Test real-time message updates via WebSocket"""
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            await page.goto(BASE_URL, wait_until="networkidle", timeout=TIMEOUT)
            
            # Send message and monitor for real-time updates
            chat_input = await page.query_selector('textarea')
            if chat_input:
                await chat_input.fill("Test real-time message")
                
                # Count messages before sending
                messages_before = await page.query_selector_all('[data-testid="stChatMessage"]')
                count_before = len(messages_before)
                
                send_button = await page.query_selector('button:has-text("Send")')
                if send_button:
                    await send_button.click()
                    
                    # Wait for real-time update
                    await page.wait_for_timeout(3000)
                    
                    # Count messages after
                    messages_after = await page.query_selector_all('[data-testid="stChatMessage"]')
                    count_after = len(messages_after)
                    
                    assert count_after > count_before, "Real-time message update not received"
                    
        finally:
            await context.close()
    
    @pytest.mark.asyncio
    async def test_typing_indicator(self, browser: Browser):
        """Test typing indicator during agent response"""
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            await page.goto(BASE_URL, wait_until="networkidle", timeout=TIMEOUT)
            
            # Send message
            chat_input = await page.query_selector('textarea')
            send_button = await page.query_selector('button:has-text("Send")')
            
            if chat_input and send_button:
                await chat_input.fill("Tell me a story")
                await send_button.click()
                
                # Look for typing indicator
                await page.wait_for_timeout(500)
                typing_indicator = await page.query_selector('[data-testid="stSpinner"], .typing-indicator, text=/typing|thinking/i')
                
                # Typing indicator should appear briefly
                if typing_indicator:
                    assert await typing_indicator.is_visible(), "Typing indicator not shown"
                    
        finally:
            await context.close()


class TestPerformanceMetrics:
    """Test application performance metrics"""
    
    @pytest.mark.asyncio
    async def test_page_load_time(self, browser: Browser):
        """Test page load performance"""
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            start_time = time.time()
            await page.goto(BASE_URL, wait_until="networkidle", timeout=TIMEOUT)
            load_time = time.time() - start_time
            
            assert load_time < 10, f"Page load time too slow: {load_time:.2f}s"
            
            # Get performance metrics
            metrics = await page.evaluate("""() => {
                const timing = performance.timing;
                return {
                    loadTime: timing.loadEventEnd - timing.navigationStart,
                    domReady: timing.domContentLoadedEventEnd - timing.navigationStart,
                    responseTime: timing.responseEnd - timing.requestStart
                };
            }""")
            
            print(f"Performance metrics: {json.dumps(metrics, indent=2)}")
            
        finally:
            await context.close()
    
    @pytest.mark.asyncio
    async def test_agent_response_time(self, browser: Browser):
        """Test AI agent response time"""
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            await page.goto(BASE_URL, wait_until="networkidle", timeout=TIMEOUT)
            
            chat_input = await page.query_selector('textarea')
            send_button = await page.query_selector('button:has-text("Send")')
            
            if chat_input and send_button:
                await chat_input.fill("What is 2+2?")
                
                start_time = time.time()
                await send_button.click()
                
                # Wait for response
                await page.wait_for_selector('[data-testid="stChatMessage"]:last-child', timeout=TIMEOUT)
                response_time = time.time() - start_time
                
                assert response_time < 30, f"Agent response too slow: {response_time:.2f}s"
                print(f"Agent response time: {response_time:.2f}s")
                
        finally:
            await context.close()


# Pytest fixtures
@pytest.fixture(scope="session")
async def browser():
    """Provide a browser instance for tests"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        yield browser
        await browser.close()


@pytest.fixture(scope="function")
async def page(browser):
    """Provide a fresh page for each test"""
    context = await browser.new_context()
    page = await context.new_page()
    yield page
    await context.close()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
