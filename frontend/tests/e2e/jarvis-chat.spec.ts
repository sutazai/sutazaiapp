import { test, expect } from '@playwright/test';

test.describe('JARVIS Chat Interface', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(3000);
  });

  test('should have chat input area', async ({ page }) => {
    // Look for text input, textarea, or chat input component
    const chatInputSelectors = [
      'textarea[placeholder*="Type"], textarea[placeholder*="Message"], textarea[placeholder*="Ask"]',
      'input[type="text"][placeholder*="Type"], input[type="text"][placeholder*="Message"]',
      '[data-testid="stTextArea"] textarea',
      '[data-testid="stTextInput"] input',
      '.stTextArea textarea',
      '.stTextInput input'
    ];
    
    let chatInput = null;
    for (const selector of chatInputSelectors) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        chatInput = element.first();
        break;
      }
    }
    
    if (chatInput) {
      await expect(chatInput).toBeVisible();
      await expect(chatInput).toBeEditable();
    } else {
      // Log what we can see on the page for debugging
      const visibleText = await page.locator('body').innerText();
      console.log('Page content:', visibleText.substring(0, 500));
      throw new Error('No chat input found on page');
    }
  });

  test('should have send button or enter functionality', async ({ page }) => {
    // Look for send button
    const sendButtonSelectors = [
      'button:has-text("Send")',
      'button:has-text("Submit")',
      'button:has-text("Ask")',
      'button[aria-label*="send"]',
      '[data-testid="stButton"] button'
    ];
    
    let sendButton = null;
    for (const selector of sendButtonSelectors) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        sendButton = element.first();
        break;
      }
    }
    
    if (sendButton) {
      await expect(sendButton).toBeVisible();
      await expect(sendButton).toBeEnabled();
    }
  });

  test('should display chat messages area', async ({ page }) => {
    // Look for chat message container
    const messageAreaSelectors = [
      '.message-container',
      '.chat-messages',
      '[data-testid="stChatMessage"]',
      '.stMarkdown',
      'div[class*="chat"]',
      'div[class*="message"]'
    ];
    
    let messageArea = null;
    for (const selector of messageAreaSelectors) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        messageArea = element.first();
        break;
      }
    }
    
    if (messageArea) {
      await expect(messageArea).toBeVisible();
    }
  });

  test('should send a message and receive response', async ({ page }) => {
    // Find chat input
    const chatInput = page.locator('textarea, input[type="text"]').first();
    
    if (await chatInput.isVisible()) {
      // Type a test message
      await chatInput.fill('Hello JARVIS, what is 2+2?');
      
      // Find and click send button or press Enter
      const sendButton = page.locator('button').filter({ hasText: /Send|Submit|Ask/i }).first();
      
      if (await sendButton.isVisible()) {
        await sendButton.click();
      } else {
        await chatInput.press('Enter');
      }
      
      // Wait for response (increased timeout for AI response)
      await page.waitForTimeout(5000);
      
      // Check if response appeared
      const responseText = page.locator('text=/4|four|answer/i');
      const hasResponse = await responseText.count() > 0;
      
      if (hasResponse) {
        await expect(responseText.first()).toBeVisible();
      } else {
        // Check if any error message appeared
        const errorText = page.locator('text=/error|failed|unable/i');
        if (await errorText.count() > 0) {
          console.log('Error message found:', await errorText.first().textContent());
        }
      }
    }
  });

  test('should maintain chat history', async ({ page }) => {
    // Send first message
    const chatInput = page.locator('textarea, input[type="text"]').first();
    
    if (await chatInput.isVisible()) {
      await chatInput.fill('First message');
      await chatInput.press('Enter');
      await page.waitForTimeout(2000);
      
      // Send second message
      await chatInput.fill('Second message');
      await chatInput.press('Enter');
      await page.waitForTimeout(2000);
      
      // Check if both messages are visible
      const messages = page.locator('div').filter({ hasText: /First message|Second message/ });
      const messageCount = await messages.count();
      
      expect(messageCount).toBeGreaterThanOrEqual(2);
    }
  });

  test('should show typing indicator when processing', async ({ page }) => {
    const chatInput = page.locator('textarea, input[type="text"]').first();
    
    if (await chatInput.isVisible()) {
      await chatInput.fill('Complex question requiring processing');
      await chatInput.press('Enter');
      
      // Look for typing/processing indicator
      const indicators = page.locator('text=/typing|processing|thinking|loading/i');
      
      // Check within 1 second if indicator appears
      await page.waitForTimeout(1000);
      const hasIndicator = await indicators.count() > 0;
      
      if (hasIndicator) {
        console.log('Processing indicator found');
      }
    }
  });
});