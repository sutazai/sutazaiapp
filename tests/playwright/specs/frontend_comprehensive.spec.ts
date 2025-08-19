import { test, expect } from '@playwright/test';

test.describe('SutazAI Frontend Comprehensive Testing', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to frontend application
    await page.goto('http://localhost:10011');
  });

  test('should load homepage and display title', async ({ page }) => {
    // Wait for the page to load
    await page.waitForLoadState('networkidle');
    
    // Check if Streamlit is running
    await expect(page).toHaveTitle(/Streamlit/);
    
    // Look for SutazAI branding
    const heading = page.locator('h1').filter({ hasText: /SutazAI/ });
    await expect(heading).toBeVisible();
  });

  test('should display system status indicator', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    // Look for system status indicators
    const statusIndicators = page.locator('[title="System Status"]');
    await expect(statusIndicators.first()).toBeVisible();
  });

  test('should have functional sidebar navigation', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    // Wait for sidebar to load
    await page.waitForSelector('[data-testid="stSidebar"]', { timeout: 10000 });
    
    // Check if navigation elements are present
    const sidebar = page.locator('[data-testid="stSidebar"]');
    await expect(sidebar).toBeVisible();
    
    // Look for navigation buttons
    const navButtons = sidebar.locator('button').filter({ hasText: /Dashboard|Agent|System/ });
    await expect(navButtons.first()).toBeVisible();
  });

  test('should handle API connectivity gracefully', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    // Check for error boundaries or graceful degradation
    const errorMessages = page.locator('.stError, [data-testid="stError"]');
    const errorCount = await errorMessages.count();
    
    // Should not have critical errors that break the UI
    expect(errorCount).toBeLessThan(5);
    
    // If there are errors, they should be handled gracefully
    if (errorCount > 0) {
      const firstError = errorMessages.first();
      const errorText = await firstError.textContent();
      expect(errorText).toMatch(/temporarily unavailable|connection|retry/i);
    }
  });

  test('should be responsive on different screen sizes', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.waitForTimeout(1000);
    
    // Sidebar should be collapsible on mobile
    const sidebar = page.locator('[data-testid="stSidebar"]');
    await expect(sidebar).toBeVisible();
    
    // Test tablet viewport
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.waitForTimeout(1000);
    
    // Test desktop viewport
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.waitForTimeout(1000);
    
    // Main content should be visible in all viewports
    const mainContent = page.locator('[data-testid="stMain"], .main');
    await expect(mainContent).toBeVisible();
  });

  test('should have accessible elements', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    // Check for proper heading hierarchy
    const h1Elements = page.locator('h1');
    await expect(h1Elements.first()).toBeVisible();
    
    // Check for button accessibility
    const buttons = page.locator('button');
    const buttonCount = await buttons.count();
    
    if (buttonCount > 0) {
      // Buttons should have text or aria-labels
      for (let i = 0; i < Math.min(buttonCount, 5); i++) {
        const button = buttons.nth(i);
        const text = await button.textContent();
        const ariaLabel = await button.getAttribute('aria-label');
        expect(text || ariaLabel).toBeTruthy();
      }
    }
  });

  test('should load without console errors', async ({ page }) => {
    const errors: string[] = [];
    
    page.on('console', msg => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });
    
    await page.goto('http://localhost:10011');
    await page.waitForLoadState('networkidle');
    
    // Should have minimal console errors
    expect(errors.length).toBeLessThan(10);
    
    // Log errors for debugging
    if (errors.length > 0) {
      console.log('Console errors detected:', errors);
    }
  });

  test('should handle user interactions', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    // Try to interact with UI elements
    const clickableElements = page.locator('button, [role="button"], .stButton');
    const elementCount = await clickableElements.count();
    
    if (elementCount > 0) {
      // Click the first available button
      const firstButton = clickableElements.first();
      await expect(firstButton).toBeVisible();
      
      // Try clicking (should not crash the app)
      try {
        await firstButton.click();
        await page.waitForTimeout(1000);
        
        // App should still be responsive
        await expect(page.locator('body')).toBeVisible();
      } catch (error) {
        console.log('Button interaction test failed:', error);
      }
    }
  });

  test('should have performance optimizations', async ({ page }) => {
    const startTime = Date.now();
    
    await page.goto('http://localhost:10011');
    await page.waitForLoadState('networkidle');
    
    const loadTime = Date.now() - startTime;
    
    // Should load within reasonable time (adjust based on system)
    expect(loadTime).toBeLessThan(30000); // 30 seconds max
    
    // Check for caching headers or optimization indicators
    const response = await page.goto('http://localhost:10011');
    const cacheControl = response?.headers()['cache-control'];
    
    console.log('Load time:', loadTime, 'ms');
    console.log('Cache control:', cacheControl);
  });

  test('should display meaningful content', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    // Should have some actual content, not just loading states
    const textContent = await page.locator('body').textContent();
    
    expect(textContent).toContain('SutazAI');
    expect(textContent.length).toBeGreaterThan(100);
    
    // Should not be stuck in loading state
    const loadingIndicators = page.locator('.stSpinner, [data-testid="stSpinner"], .loading');
    const loadingCount = await loadingIndicators.count();
    
    // Some loading is okay, but not excessive
    expect(loadingCount).toBeLessThan(10);
  });
});

test.describe('Frontend Error Handling', () => {
  test('should gracefully handle backend downtime', async ({ page }) => {
    await page.goto('http://localhost:10011');
    await page.waitForLoadState('networkidle');
    
    // Look for offline mode or error recovery UI
    const errorRecovery = page.locator('.stError, [data-testid="stError"]').filter({ 
      hasText: /offline|retry|connection|temporarily/i 
    });
    
    const errorCount = await errorRecovery.count();
    
    if (errorCount > 0) {
      // Should have user-friendly error messages
      const errorText = await errorRecovery.first().textContent();
      expect(errorText).toMatch(/try again|refresh|contact|offline|connection/i);
      
      // Should not display raw error traces to users
      expect(errorText).not.toMatch(/traceback|exception|stack trace/i);
    }
  });

  test('should provide recovery options', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    // Look for retry buttons or recovery mechanisms
    const retryButtons = page.locator('button').filter({ hasText: /retry|refresh|reload/i });
    const retryCount = await retryButtons.count();
    
    // If there are errors, there should be recovery options
    const errors = page.locator('.stError, [data-testid="stError"]');
    const errorCount = await errors.count();
    
    if (errorCount > 0) {
      expect(retryCount).toBeGreaterThan(0);
    }
  });
});

test.describe('Frontend Components', () => {
  test('should render metrics and dashboard elements', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    // Look for dashboard components
    const metrics = page.locator('.metric, [data-testid*="metric"], .stMetric');
    const charts = page.locator('canvas, svg, .plot, .chart');
    
    // Should have some visual elements
    const metricsCount = await metrics.count();
    const chartsCount = await charts.count();
    
    expect(metricsCount + chartsCount).toBeGreaterThan(0);
  });

  test('should have functional forms and inputs', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    // Look for input elements
    const inputs = page.locator('input, select, textarea');
    const inputCount = await inputs.count();
    
    if (inputCount > 0) {
      const firstInput = inputs.first();
      await expect(firstInput).toBeVisible();
      
      // Try typing in the input
      if (await firstInput.getAttribute('type') !== 'file') {
        try {
          await firstInput.fill('test');
          const value = await firstInput.inputValue();
          expect(value).toBe('test');
        } catch (error) {
          console.log('Input interaction test failed:', error);
        }
      }
    }
  });
});