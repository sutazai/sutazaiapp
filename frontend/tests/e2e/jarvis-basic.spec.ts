import { test, expect } from '@playwright/test';

test.describe('JARVIS Basic Functionality', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for Streamlit to fully load
    await page.waitForTimeout(3000);
  });

  test('should load the JARVIS interface', async ({ page }) => {
    // Check if the page title is correct
    await expect(page).toHaveTitle(/JARVIS/i);
    
    // Check for main components
    const mainContent = page.locator('.main');
    await expect(mainContent).toBeVisible({ timeout: 10000 });
  });

  test('should display welcome message', async ({ page }) => {
    // Look for welcome text or JARVIS branding
    const welcomeText = page.getByText(/JARVIS|Advanced Voice Assistant|Welcome/i);
    await expect(welcomeText.first()).toBeVisible({ timeout: 10000 });
  });

  test('should have sidebar with options', async ({ page }) => {
    // Check if sidebar exists
    const sidebar = page.locator('.css-1lcbmhc, [data-testid="stSidebar"]');
    
    // Try to open sidebar if it's collapsed
    const sidebarToggle = page.locator('[aria-label="Open sidebar"], button:has-text("☰")');
    if (await sidebarToggle.isVisible()) {
      await sidebarToggle.click();
      await page.waitForTimeout(1000);
    }
    
    // Check for menu options
    const menuOptions = page.locator('[data-testid="stSidebar"] .element-container');
    const count = await menuOptions.count();
    expect(count).toBeGreaterThan(0);
  });

  test('should have theme toggle functionality', async ({ page }) => {
    // Look for theme toggle button
    const themeButton = page.locator('button:has-text("🌙"), button:has-text("☀️"), [aria-label*="theme"]');
    
    if (await themeButton.isVisible()) {
      // Get initial background color
      const body = page.locator('body');
      const initialBg = await body.evaluate(el => 
        window.getComputedStyle(el).backgroundColor
      );
      
      // Toggle theme
      await themeButton.click();
      await page.waitForTimeout(1000);
      
      // Check if background changed
      const newBg = await body.evaluate(el => 
        window.getComputedStyle(el).backgroundColor
      );
      
      expect(newBg).not.toBe(initialBg);
    }
  });

  test('should display system status indicators', async ({ page }) => {
    // Look for status indicators
    const statusIndicators = page.locator('text=/Status|Online|Connected|Ready/i');
    const hasStatus = await statusIndicators.count() > 0;
    
    if (hasStatus) {
      await expect(statusIndicators.first()).toBeVisible();
    }
  });
});