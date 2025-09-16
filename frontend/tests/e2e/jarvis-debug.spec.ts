import { test, expect } from '@playwright/test';

test.describe('JARVIS Debug - What is actually running?', () => {
  test('capture actual page state and take screenshot', async ({ page }) => {
    // Navigate to the app
    await page.goto('http://localhost:11000');
    
    // Wait for page to fully load
    await page.waitForTimeout(5000);
    
    // Take a screenshot
    await page.screenshot({ path: 'jarvis-actual-state.png', fullPage: true });
    console.log('Screenshot saved as jarvis-actual-state.png');
    
    // Get page title
    const title = await page.title();
    console.log(`Page Title: "${title}"`);
    
    // Get all visible text on the page
    const bodyText = await page.locator('body').innerText();
    console.log('=== PAGE CONTENT ===');
    console.log(bodyText.substring(0, 2000));
    console.log('===================');
    
    // Check for common Streamlit elements
    const streamlitElements = {
      'Main container': await page.locator('.main').count(),
      'Sidebar': await page.locator('[data-testid="stSidebar"]').count(),
      'App container': await page.locator('[data-testid="stAppViewContainer"]').count(),
      'Markdown elements': await page.locator('[data-testid="stMarkdown"]').count(),
      'Text inputs': await page.locator('[data-testid="stTextInput"]').count(),
      'Text areas': await page.locator('[data-testid="stTextArea"]').count(),
      'Buttons': await page.locator('[data-testid="stButton"]').count(),
      'Columns': await page.locator('[data-testid="column"]').count(),
      'Expanders': await page.locator('[data-testid="stExpander"]').count(),
      'Metrics': await page.locator('[data-testid="stMetric"]').count(),
      'Alerts': await page.locator('[data-testid="stAlert"]').count(),
      'Chat messages': await page.locator('[data-testid="stChatMessage"]').count(),
    };
    
    console.log('\n=== STREAMLIT COMPONENTS FOUND ===');
    for (const [name, count] of Object.entries(streamlitElements)) {
      if (count > 0) {
        console.log(`✓ ${name}: ${count}`);
      }
    }
    
    // Look for any error messages
    const errorElements = await page.locator('text=/error|failed|exception|traceback/i').all();
    if (errorElements.length > 0) {
      console.log('\n=== ERRORS FOUND ===');
      for (const elem of errorElements) {
        const text = await elem.textContent();
        console.log(`- ${text?.substring(0, 100)}`);
      }
    }
    
    // Check for any inputs
    const inputs = await page.locator('input, textarea, select').all();
    console.log(`\n=== INPUT ELEMENTS: ${inputs.length} ===`);
    for (let i = 0; i < Math.min(inputs.length, 5); i++) {
      const input = inputs[i];
      const tagName = await input.evaluate(el => el.tagName);
      const placeholder = await input.getAttribute('placeholder');
      const type = await input.getAttribute('type');
      console.log(`${i+1}. ${tagName} - type: ${type}, placeholder: "${placeholder}"`);
    }
    
    // Check for buttons
    const buttons = await page.locator('button').all();
    console.log(`\n=== BUTTONS: ${buttons.length} ===`);
    for (let i = 0; i < Math.min(buttons.length, 10); i++) {
      const button = buttons[i];
      const text = await button.textContent();
      const ariaLabel = await button.getAttribute('aria-label');
      console.log(`${i+1}. "${text?.trim()}" (aria-label: ${ariaLabel})`);
    }
    
    // Check console for errors
    const consoleErrors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });
    
    // Reload to capture console errors
    await page.reload();
    await page.waitForTimeout(3000);
    
    if (consoleErrors.length > 0) {
      console.log('\n=== CONSOLE ERRORS ===');
      consoleErrors.forEach(err => console.log(`- ${err}`));
    }
    
    // Check network requests
    const requests: string[] = [];
    page.on('request', request => {
      const url = request.url();
      if (!url.includes('_stcore') && !url.includes('.js') && !url.includes('.css')) {
        requests.push(`${request.method()} ${url}`);
      }
    });
    
    // Trigger some activity
    const firstInput = await page.locator('input, textarea').first();
    if (await firstInput.isVisible()) {
      await firstInput.fill('test');
      await page.waitForTimeout(1000);
    }
    
    if (requests.length > 0) {
      console.log('\n=== NETWORK REQUESTS ===');
      requests.forEach(req => console.log(`- ${req}`));
    }
  });
});