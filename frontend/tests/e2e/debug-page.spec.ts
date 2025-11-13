import { test } from '@playwright/test';

test('capture page content', async ({ page }) => {
  await page.goto('http://localhost:11000');
  await page.waitForTimeout(5000);
  
  // Take screenshot
  await page.screenshot({ path: '/tmp/jarvis-actual-page.png', fullPage: true });
  
  // Get all text content
  const bodyText = await page.locator('body').innerText();
  console.log('=== PAGE TEXT CONTENT ===');
  console.log(bodyText);
  
  // Get HTML
  const html = await page.content();
  require('fs').writeFileSync('/tmp/jarvis-page.html', html);
  
  console.log('=== Saved HTML to /tmp/jarvis-page.html ===');
  console.log('=== Saved screenshot to /tmp/jarvis-actual-page.png ===');
});
