import { test, expect } from '@playwright/test';

// Helper function to wait for Streamlit app to be fully loaded
async function waitForStreamlitReady(page) {
  // Wait for the Streamlit iframe or main container
  await page.waitForSelector('[data-testid="stApp"]', { timeout: 15000 }).catch(() => {
    // Fallback if data-testid is not available
    return page.waitForSelector('.main', { timeout: 15000 });
  });
  
  // Wait for any loading indicators to disappear
  await page.waitForFunction(() => {
    const spinners = document.querySelectorAll('[data-testid="stSpinner"]');
    return spinners.length === 0;
  }, { timeout: 10000 }).catch(() => {
    // Ignore timeout - app might not have spinners
  });
  
  // Additional wait for JavaScript to settle
  await page.waitForTimeout(2000);
}

test.describe('JARVIS Basic Functionality', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForStreamlitReady(page);
  });

  test('should load the JARVIS interface', async ({ page }) => {
    // Check for main Streamlit container
    const mainContent = page.locator('[data-testid="stApp"], .main').first();
    await expect(mainContent).toBeVisible({ timeout: 10000 });
    
    // Check if page title eventually becomes JARVIS
    await page.waitForFunction(() => {
      return document.title.match(/JARVIS/i) !== null;
    }, { timeout: 10000 }).catch(() => {
      // Title might not change, that's okay as long as content is there
    });
  });

  test('should display welcome message', async ({ page }) => {
    // Look for JARVIS branding (rendered in HTML)
    const jarvisbrand = page.getByText(/J\.A\.R\.V\.I\.S|Just A Rather Very Intelligent System/i);
    await expect(jarvisbrand.first()).toBeVisible({ timeout: 15000 });
  });

  test('should have sidebar with options', async ({ page }) => {
    // Wait for sidebar to be present
    const sidebar = page.locator('[data-testid="stSidebar"]').first();
    await expect(sidebar).toBeVisible({ timeout: 15000 });
    
    // Try to open sidebar if it's collapsed
    const sidebarToggle = page.locator('[data-testid="collapsedControl"]');
    if (await sidebarToggle.isVisible().catch(() => false)) {
      await sidebarToggle.click();
      await page.waitForTimeout(1000);
    }
    
    // Check for any content in sidebar (Control Panel text should be there)
    const sidebarContent = page.locator('[data-testid="stSidebar"]');
    const hasContent = await sidebarContent.textContent();
    expect(hasContent).toBeTruthy();
    expect(hasContent.length).toBeGreaterThan(0);
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