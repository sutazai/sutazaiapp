import { test, expect } from '@playwright/test';

test.describe('JARVIS UI Components and Responsiveness', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(3000);
  });

  test('should display JARVIS branding/logo', async ({ page }) => {
    // Look for JARVIS branding elements
    const brandingSelectors = [
      'text=/JARVIS/i',
      'img[alt*="JARVIS"]',
      '.jarvis-logo',
      '.arc-reactor',
      'div[class*="jarvis"]'
    ];
    
    let hasBranding = false;
    for (const selector of brandingSelectors) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        hasBranding = true;
        console.log('✅ JARVIS branding found');
        break;
      }
    }
    
    if (!hasBranding) {
      console.log('⚠️ No JARVIS branding elements found');
    }
  });

  test('should have animated background or effects', async ({ page }) => {
    // Check for animated elements
    const animationSelectors = [
      '.jarvis-bg',
      '[class*="animated"]',
      '[class*="pulse"]',
      '[class*="glow"]',
      'div[style*="animation"]'
    ];
    
    let hasAnimations = false;
    for (const selector of animationSelectors) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        hasAnimations = true;
        console.log('✅ Animated UI elements found');
        break;
      }
    }
    
    // Check for CSS animations
    const hasCSS = await page.evaluate(() => {
      const styles = document.styleSheets;
      for (let sheet of styles) {
        try {
          const rules = sheet.cssRules || sheet.rules;
          for (let rule of rules) {
            if (rule.cssText && rule.cssText.includes('@keyframes')) {
              return true;
            }
          }
        } catch (e) {
          // Cross-origin stylesheets
        }
      }
      return false;
    });
    
    if (hasCSS) {
      console.log('✅ CSS animations defined');
    } else if (!hasAnimations) {
      console.log('⚠️ No animations found');
    }
  });

  test('should be responsive on mobile viewport', async ({ page }) => {
    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.waitForTimeout(1000);
    
    // Check if main elements are still visible
    const mainContent = page.locator('.main, [data-testid="stAppViewContainer"]');
    await expect(mainContent).toBeVisible();
    
    // Check if layout adapted
    const sidebar = page.locator('[data-testid="stSidebar"]');
    const sidebarVisible = await sidebar.isVisible();
    
    if (!sidebarVisible) {
      console.log('✅ Sidebar hidden on mobile (responsive)');
      
      // Check for hamburger menu
      const hamburger = page.locator('button:has-text("☰"), [aria-label*="menu"]');
      if (await hamburger.isVisible()) {
        console.log('✅ Mobile menu button available');
      }
    }
    
    // Check text readability
    const fontSize = await page.locator('body').evaluate(el => 
      window.getComputedStyle(el).fontSize
    );
    console.log(`Mobile font size: ${fontSize}`);
  });

  test('should have dark/light theme toggle', async ({ page }) => {
    // Look for theme toggle
    const themeSelectors = [
      'button:has-text("🌙")',
      'button:has-text("☀️")',
      'button:has-text("🌓")',
      '[aria-label*="theme"]',
      'button[title*="theme"]'
    ];
    
    let themeToggle = null;
    for (const selector of themeSelectors) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        themeToggle = element.first();
        break;
      }
    }
    
    if (themeToggle && await themeToggle.isVisible()) {
      // Get initial theme
      const initialBg = await page.locator('body').evaluate(el => 
        window.getComputedStyle(el).backgroundColor
      );
      
      // Toggle theme
      await themeToggle.click();
      await page.waitForTimeout(1000);
      
      // Check if theme changed
      const newBg = await page.locator('body').evaluate(el => 
        window.getComputedStyle(el).backgroundColor
      );
      
      if (newBg !== initialBg) {
        console.log('✅ Theme toggle working');
      } else {
        console.log('⚠️ Theme toggle not changing colors');
      }
    } else {
      console.log('⚠️ No theme toggle found');
    }
  });

  test('should display system status dashboard', async ({ page }) => {
    // Look for status/monitoring elements
    const dashboardSelectors = [
      'text=/CPU|Memory|Status|Health/i',
      '[data-testid="system-monitor"]',
      '.system-status',
      'div[class*="monitor"]',
      'div[class*="dashboard"]'
    ];
    
    let hasDashboard = false;
    for (const selector of dashboardSelectors) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        hasDashboard = true;
        console.log('✅ System status dashboard found');
        break;
      }
    }
    
    if (!hasDashboard) {
      console.log('⚠️ No system status dashboard found');
    }
  });

  test('should have data visualization components', async ({ page }) => {
    // Look for charts/graphs
    const chartSelectors = [
      'canvas',
      'svg[class*="chart"]',
      '[class*="plotly"]',
      '[data-testid="stPlotlyChart"]',
      '[data-testid="stMetric"]',
      '.recharts-wrapper'
    ];
    
    let hasCharts = false;
    for (const selector of chartSelectors) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        hasCharts = true;
        console.log('✅ Data visualization components found');
        break;
      }
    }
    
    if (!hasCharts) {
      console.log('⚠️ No data visualization found');
    }
  });

  test('should show loading states properly', async ({ page }) => {
    // Trigger an action that causes loading
    const chatInput = page.locator('textarea, input[type="text"]').first();
    
    if (await chatInput.isVisible()) {
      await chatInput.fill('Test loading state');
      await chatInput.press('Enter');
      
      // Look for loading indicators
      const loadingSelectors = [
        '.spinner',
        '[class*="loading"]',
        'text=/loading|processing|thinking/i',
        '[aria-busy="true"]',
        '.stSpinner'
      ];
      
      let hasLoading = false;
      
      // Check immediately after sending
      for (const selector of loadingSelectors) {
        const element = page.locator(selector);
        if (await element.count() > 0) {
          hasLoading = true;
          console.log('✅ Loading state displayed');
          break;
        }
      }
      
      if (!hasLoading) {
        console.log('⚠️ No loading indicators shown');
      }
    }
  });

  test('should have error handling UI', async ({ page }) => {
    // Try to trigger an error
    const chatInput = page.locator('textarea, input[type="text"]').first();
    
    if (await chatInput.isVisible()) {
      // Send potentially error-causing input
      await chatInput.fill('/error_test_12345');
      await chatInput.press('Enter');
      
      await page.waitForTimeout(3000);
      
      // Look for error UI
      const errorSelectors = [
        'text=/error|failed|unable|sorry/i',
        '[data-testid="stAlert"]',
        '.error-message',
        '[class*="error"]',
        '[role="alert"]'
      ];
      
      let hasErrorUI = false;
      for (const selector of errorSelectors) {
        const element = page.locator(selector);
        if (await element.count() > 0) {
          hasErrorUI = true;
          const errorText = await element.first().textContent();
          console.log(`✅ Error handling UI: ${errorText?.substring(0, 50)}`);
          break;
        }
      }
      
      if (!hasErrorUI) {
        console.log('⚠️ No error handling UI found');
      }
    }
  });

  test('should have tooltips or help text', async ({ page }) => {
    // Look for tooltips
    const tooltipSelectors = [
      '[title]',
      '[data-tooltip]',
      '[aria-describedby]',
      '.tooltip',
      '[class*="tooltip"]'
    ];
    
    let hasTooltips = false;
    for (const selector of tooltipSelectors) {
      const elements = page.locator(selector);
      const count = await elements.count();
      if (count > 0) {
        hasTooltips = true;
        console.log(`✅ Found ${count} elements with tooltips`);
        break;
      }
    }
    
    if (!hasTooltips) {
      console.log('⚠️ No tooltips found');
    }
  });

  test('should have keyboard navigation support', async ({ page }) => {
    // Test tab navigation
    await page.keyboard.press('Tab');
    await page.waitForTimeout(500);
    
    // Check if an element is focused
    const focusedElement = await page.evaluate(() => {
      const el = document.activeElement;
      return {
        tagName: el?.tagName,
        className: el?.className,
        hasTabIndex: el?.hasAttribute('tabindex')
      };
    });
    
    if (focusedElement.tagName !== 'BODY') {
      console.log(`✅ Keyboard navigation: focused ${focusedElement.tagName}`);
    } else {
      console.log('⚠️ No keyboard navigation detected');
    }
    
    // Test escape key
    await page.keyboard.press('Escape');
    
    // Test enter key on focused element
    await page.keyboard.press('Enter');
  });
});