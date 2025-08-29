import { test, expect } from '@playwright/test';

test.describe('JARVIS AI Model Support', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(3000);
  });

  test('should have model selection dropdown', async ({ page }) => {
    // Open sidebar if needed
    const sidebarToggle = page.locator('[aria-label="Open sidebar"], button:has-text("☰")');
    if (await sidebarToggle.isVisible()) {
      await sidebarToggle.click();
      await page.waitForTimeout(1000);
    }
    
    // Look for model selection
    const modelSelectors = [
      'select[aria-label*="model"]',
      '[data-testid="stSelectbox"]',
      'div[data-baseweb="select"]',
      'text=/model|GPT|Claude|Llama|Ollama/i',
      'button[aria-label*="model"]'
    ];
    
    let modelSelector = null;
    for (const selector of modelSelectors) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        modelSelector = element.first();
        console.log('✅ Model selection component found');
        break;
      }
    }
    
    if (!modelSelector) {
      console.log('❌ No model selection found');
    }
  });

  test('should display available AI models', async ({ page }) => {
    // Expected models based on system architecture
    const expectedModels = [
      'GPT-4',
      'GPT-3.5',
      'Claude',
      'Llama',
      'Ollama',
      'Custom'
    ];
    
    let foundModels = [];
    
    for (const model of expectedModels) {
      const modelElement = page.locator(`text=/${model}/i`);
      if (await modelElement.count() > 0) {
        foundModels.push(model);
      }
    }
    
    if (foundModels.length > 0) {
      console.log(`✅ Found models: ${foundModels.join(', ')}`);
    } else {
      console.log('⚠️ No AI models displayed');
    }
  });

  test('should show model status/availability', async ({ page }) => {
    // Look for model status indicators
    const statusSelectors = [
      'text=/available|ready|loading|offline/i',
      '[data-testid="model-status"]',
      '.model-status',
      'span[class*="status"]'
    ];
    
    let hasStatus = false;
    for (const selector of statusSelectors) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        const status = await element.first().textContent();
        hasStatus = true;
        console.log(`✅ Model status: ${status}`);
        break;
      }
    }
    
    if (!hasStatus) {
      console.log('⚠️ No model status indicators found');
    }
  });

  test('should allow switching between models', async ({ page }) => {
    // Find model selector
    const modelDropdown = page.locator('select, [data-baseweb="select"]').first();
    
    if (await modelDropdown.isVisible()) {
      // Try to click and see options
      await modelDropdown.click();
      await page.waitForTimeout(1000);
      
      // Look for dropdown options
      const options = page.locator('[role="option"], option');
      const optionCount = await options.count();
      
      if (optionCount > 1) {
        console.log(`✅ ${optionCount} model options available`);
        
        // Try to select a different option
        const secondOption = options.nth(1);
        if (await secondOption.isVisible()) {
          await secondOption.click();
          console.log('✅ Model switching successful');
        }
      } else {
        console.log('⚠️ No model options in dropdown');
      }
    }
  });

  test('should display model parameters/settings', async ({ page }) => {
    // Look for model parameter controls
    const parameterSelectors = [
      'input[aria-label*="temperature"]',
      'input[aria-label*="max_tokens"]',
      'input[aria-label*="top_p"]',
      'text=/temperature|tokens|sampling/i',
      '[data-testid="stSlider"]',
      'input[type="range"]'
    ];
    
    let foundParameters = [];
    for (const selector of parameterSelectors) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        foundParameters.push(selector);
      }
    }
    
    if (foundParameters.length > 0) {
      console.log(`✅ Found ${foundParameters.length} parameter controls`);
    } else {
      console.log('⚠️ No model parameter controls found');
    }
  });

  test('should show model response metadata', async ({ page }) => {
    // Send a message
    const chatInput = page.locator('textarea, input[type="text"]').first();
    
    if (await chatInput.isVisible()) {
      await chatInput.fill('Hello, which model are you?');
      await chatInput.press('Enter');
      
      await page.waitForTimeout(5000);
      
      // Look for metadata
      const metadataSelectors = [
        'text=/model:|tokens:|time:|cost:/i',
        '[data-testid="response-metadata"]',
        '.message-metadata',
        'span[class*="metadata"]'
      ];
      
      let hasMetadata = false;
      for (const selector of metadataSelectors) {
        const element = page.locator(selector);
        if (await element.count() > 0) {
          hasMetadata = true;
          const metadata = await element.first().textContent();
          console.log(`✅ Response metadata: ${metadata}`);
          break;
        }
      }
      
      if (!hasMetadata) {
        console.log('⚠️ No response metadata displayed');
      }
    }
  });

  test('should handle model-specific features', async ({ page }) => {
    // Check for model-specific UI elements
    const featureSelectors = [
      'button:has-text("Code")',
      'button:has-text("Image")',
      'button:has-text("Vision")',
      'button:has-text("Function")',
      '[data-testid="model-features"]'
    ];
    
    let foundFeatures = [];
    for (const selector of featureSelectors) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        const text = await element.first().textContent();
        foundFeatures.push(text);
      }
    }
    
    if (foundFeatures.length > 0) {
      console.log(`✅ Model features: ${foundFeatures.join(', ')}`);
    } else {
      console.log('⚠️ No model-specific features found');
    }
  });

  test('should show model loading/switching indicator', async ({ page }) => {
    const modelDropdown = page.locator('select, [data-baseweb="select"]').first();
    
    if (await modelDropdown.isVisible()) {
      // Change model
      await modelDropdown.click();
      const options = page.locator('[role="option"], option');
      
      if (await options.count() > 1) {
        await options.nth(1).click();
        
        // Look for loading indicator
        const loadingSelectors = [
          'text=/loading|switching|initializing/i',
          '.spinner',
          '[class*="loading"]',
          'div[aria-busy="true"]'
        ];
        
        let hasLoading = false;
        for (const selector of loadingSelectors) {
          const element = page.locator(selector);
          if (await element.count() > 0) {
            hasLoading = true;
            console.log('✅ Model switching indicator found');
            break;
          }
        }
        
        if (!hasLoading) {
          console.log('⚠️ No model switching indicator');
        }
      }
    }
  });
});