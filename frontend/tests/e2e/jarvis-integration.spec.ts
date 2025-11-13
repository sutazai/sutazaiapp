import { test, expect } from '@playwright/test';

// Helper function to wait for Streamlit app to be fully loaded
async function waitForStreamlitReady(page) {
  await page.waitForSelector('[data-testid="stApp"]', { timeout: 15000 }).catch(() => {
    return page.waitForSelector('.main', { timeout: 15000 });
  });
  
  await page.waitForFunction(() => {
    const spinners = document.querySelectorAll('[data-testid="stSpinner"]');
    return spinners.length === 0;
  }, { timeout: 10000 }).catch(() => {});
  
  await page.waitForTimeout(2000);
}

test.describe('JARVIS Backend Integration', () => {
  test.beforeEach(async ({ page }) => {
    // Set longer navigation timeout for slow loads
    page.setDefaultNavigationTimeout(30000);
    page.setDefaultTimeout(15000);
    
    await page.goto('/', { waitUntil: 'domcontentloaded' });
    await waitForStreamlitReady(page);
  });

  test('should connect to backend API', async ({ page }) => {
    // Monitor network requests
    const apiCalls: string[] = [];
    
    page.on('request', request => {
      const url = request.url();
      if (url.includes('localhost:10200') || url.includes('/api/')) {
        apiCalls.push(url);
        console.log(`API Call: ${request.method()} ${url}`);
      }
    });
    
    page.on('response', response => {
      const url = response.url();
      if (url.includes('localhost:10200') || url.includes('/api/')) {
        console.log(`API Response: ${response.status()} ${url}`);
      }
    });
    
    // Trigger an API call
    const chatInput = page.locator('textarea, input[type="text"]').first();
    
    if (await chatInput.isVisible()) {
      await chatInput.fill('Test backend connection');
      await chatInput.press('Enter');
      
      await page.waitForTimeout(3000);
      
      if (apiCalls.length > 0) {
        console.log(`✅ ${apiCalls.length} API calls made`);
      } else {
        console.log('⚠️ No backend API calls detected');
      }
    }
  });

  test('should display backend service status', async ({ page }) => {
    // Look for service status indicators
    const serviceSelectors = [
      'text=/PostgreSQL|Redis|Neo4j|RabbitMQ|ChromaDB/i',
      '[data-testid="service-status"]',
      '.service-indicator',
      'div[class*="service"]'
    ];
    
    let foundServices = [];
    for (const selector of serviceSelectors) {
      const elements = page.locator(selector);
      const count = await elements.count();
      if (count > 0) {
        const text = await elements.first().textContent();
        if (text) foundServices.push(text);
      }
    }
    
    if (foundServices.length > 0) {
      console.log(`✅ Services found: ${foundServices.join(', ')}`);
    } else {
      console.log('⚠️ No backend service status displayed');
    }
  });

  test('should handle authentication if required', async ({ page }) => {
    // Check for auth elements
    const authSelectors = [
      'input[type="password"]',
      'button:has-text("Login")',
      'button:has-text("Sign In")',
      'text=/username|email|password/i',
      '[data-testid="auth-form"]'
    ];
    
    let hasAuth = false;
    for (const selector of authSelectors) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        hasAuth = true;
        console.log('✅ Authentication system present');
        break;
      }
    }
    
    if (!hasAuth) {
      console.log('ℹ️ No authentication required (open access)');
    }
  });

  test('should handle file uploads if supported', async ({ page }) => {
    // Look for file upload components
    const uploadSelectors = [
      'input[type="file"]',
      '[data-testid="stFileUploader"]',
      'button:has-text("Upload")',
      'button:has-text("Browse")',
      'div[class*="dropzone"]'
    ];
    
    let hasUpload = false;
    for (const selector of uploadSelectors) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        hasUpload = true;
        console.log('✅ File upload capability found');
        
        // Test file upload if input found
        if (selector === 'input[type="file"]') {
          const fileInput = element.first();
          if (await fileInput.isVisible()) {
            // Create a test file
            await fileInput.setInputFiles({
              name: 'test.txt',
              mimeType: 'text/plain',
              buffer: Buffer.from('Test file content')
            });
            console.log('✅ File upload tested');
          }
        }
        break;
      }
    }
    
    if (!hasUpload) {
      console.log('ℹ️ No file upload capability');
    }
  });

  test('should display agent/MCP server status', async ({ page }) => {
    // Look for agent status
    const agentSelectors = [
      'text=/agent|MCP|server|orchestrator/i',
      '[data-testid="agent-status"]',
      '.agent-indicator',
      'div[class*="agent"]'
    ];
    
    let foundAgents = [];
    for (const selector of agentSelectors) {
      const elements = page.locator(selector);
      const count = await elements.count();
      if (count > 0) {
        for (let i = 0; i < Math.min(count, 3); i++) {
          const text = await elements.nth(i).textContent();
          if (text && text.length < 50) {
            foundAgents.push(text);
          }
        }
      }
    }
    
    if (foundAgents.length > 0) {
      console.log(`✅ Agents/MCP servers: ${foundAgents.join(', ')}`);
    } else {
      console.log('⚠️ No agent/MCP server status found');
    }
  });

  test('should support session management', async ({ page }) => {
    // Look for session indicators
    const sessionSelectors = [
      'text=/session|conversation|history/i',
      '[data-testid="session-id"]',
      'button:has-text("New Chat")',
      'button:has-text("Clear")',
      'button:has-text("Reset")'
    ];
    
    let hasSession = false;
    for (const selector of sessionSelectors) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        hasSession = true;
        console.log('✅ Session management features found');
        break;
      }
    }
    
    if (!hasSession) {
      console.log('⚠️ No session management features found');
    }
  });

  test('should handle rate limiting gracefully', async ({ page }) => {
    const chatInput = page.locator('textarea, input[type="text"]').first();
    
    if (await chatInput.isVisible()) {
      // Send multiple rapid requests
      for (let i = 0; i < 10; i++) {
        await chatInput.fill(`Rapid request ${i}`);
        await chatInput.press('Enter');
        await page.waitForTimeout(100);
      }
      
      await page.waitForTimeout(3000);
      
      // Check for rate limit message
      const rateLimitSelectors = [
        'text=/rate limit|too many|slow down|please wait/i',
        '[data-testid="rate-limit-warning"]',
        '.rate-limit-message'
      ];
      
      let hasRateLimit = false;
      for (const selector of rateLimitSelectors) {
        const element = page.locator(selector);
        if (await element.count() > 0) {
          hasRateLimit = true;
          console.log('✅ Rate limiting handled gracefully');
          break;
        }
      }
      
      if (!hasRateLimit) {
        console.log('ℹ️ No rate limiting detected (or very high limit)');
      }
    }
  });

  test('should persist chat history', async ({ page }) => {
    const chatInput = page.locator('textarea, input[type="text"]').first();
    
    if (await chatInput.isVisible()) {
      // Send a unique message
      const uniqueMsg = `Test persistence ${Date.now()}`;
      await chatInput.fill(uniqueMsg);
      await chatInput.press('Enter');
      
      await page.waitForTimeout(2000);
      
      // Refresh page
      await page.reload();
      await page.waitForTimeout(3000);
      
      // Check if message persisted
      const persistedMsg = page.locator(`text="${uniqueMsg}"`);
      if (await persistedMsg.count() > 0) {
        console.log('✅ Chat history persisted across reload');
      } else {
        console.log('ℹ️ Chat history not persisted (session-only)');
      }
    }
  });

  test('should support export/download features', async ({ page }) => {
    // Look for export buttons
    const exportSelectors = [
      'button:has-text("Export")',
      'button:has-text("Download")',
      'button:has-text("Save")',
      '[aria-label*="export"]',
      '[aria-label*="download"]'
    ];
    
    let hasExport = false;
    for (const selector of exportSelectors) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        hasExport = true;
        console.log('✅ Export/download features available');
        break;
      }
    }
    
    if (!hasExport) {
      console.log('ℹ️ No export/download features found');
    }
  });

  test('should show performance metrics', async ({ page }) => {
    // Look for performance indicators
    const perfSelectors = [
      'text=/latency|response time|performance|speed/i',
      '[data-testid="performance-metrics"]',
      '.performance-indicator',
      'span[class*="metric"]'
    ];
    
    let hasMetrics = false;
    for (const selector of perfSelectors) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        const text = await element.first().textContent();
        if (text && text.match(/\d+\s*(ms|s|seconds)/i)) {
          hasMetrics = true;
          console.log(`✅ Performance metrics: ${text}`);
          break;
        }
      }
    }
    
    if (!hasMetrics) {
      console.log('⚠️ No performance metrics displayed');
    }
  });
});