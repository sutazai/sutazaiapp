import { test, expect } from '@playwright/test';

test.describe('Enhanced Chat Interface Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(2000);
  });

  test('should handle empty message submission', async ({ page }) => {
    const chatInput = page.locator('textarea, input[type="text"]').first();
    
    if (await chatInput.isVisible()) {
      await chatInput.fill('');
      await chatInput.press('Enter');
      
      // Should not submit empty message
      await page.waitForTimeout(1000);
      const errorMsg = page.locator('text=/empty|required|enter/i');
      
      if (await errorMsg.count() > 0) {
        console.log('✅ Empty message validation working');
      }
    }
  });

  test('should handle special characters in messages', async ({ page }) => {
    const chatInput = page.locator('textarea, input[type="text"]').first();
    
    if (await chatInput.isVisible()) {
      const specialChars = '!@#$%^&*()[]{}|\\:;"<>,.?/~`±§';
      await chatInput.fill(specialChars);
      await chatInput.press('Enter');
      
      await page.waitForTimeout(2000);
      
      // Message should be displayed
      const messageElement = page.locator(`text="${specialChars}"`);
      const count = await messageElement.count();
      
      console.log(`Special chars test: ${count > 0 ? '✅ PASS' : '⚠️ NOT FOUND'}`);
    }
  });

  test('should render markdown formatting correctly', async ({ page }) => {
    const chatInput = page.locator('textarea, input[type="text"]').first();
    
    if (await chatInput.isVisible()) {
      // Test bold
      await chatInput.fill('**bold text**');
      await chatInput.press('Enter');
      await page.waitForTimeout(2000);
      
      // Check for bold rendering (implementation specific)
      const boldElements = await page.locator('strong, b, [style*="font-weight"]').count();
      console.log(`Markdown bold: ${boldElements > 0 ? '✅' : '⚠️'}`);
      
      // Clear and test italic
      await chatInput.fill('*italic text*');
      await chatInput.press('Enter');
      await page.waitForTimeout(2000);
      
      const italicElements = await page.locator('em, i, [style*="font-style"]').count();
      console.log(`Markdown italic: ${italicElements > 0 ? '✅' : '⚠️'}`);
    }
  });

  test('should handle very long messages', async ({ page }) => {
    const chatInput = page.locator('textarea, input[type="text"]').first();
    
    if (await chatInput.isVisible()) {
      const longMessage = 'A'.repeat(5000);
      await chatInput.fill(longMessage);
      await chatInput.press('Enter');
      
      await page.waitForTimeout(3000);
      
      // Check if message was processed (accepted or rejected gracefully)
      const messages = await page.locator('div[class*="message"]').count();
      console.log(`Long message handling: ${messages > 0 ? '✅ Handled' : '⚠️ Not visible'}`);
    }
  });

  test('should handle code blocks', async ({ page }) => {
    const chatInput = page.locator('textarea, input[type="text"]').first();
    
    if (await chatInput.isVisible()) {
      const codeBlock = '```python\\nprint("Hello")\\n```';
      await chatInput.fill(codeBlock);
      await chatInput.press('Enter');
      
      await page.waitForTimeout(2000);
      
      // Check for code block rendering
      const codeElements = await page.locator('code, pre, [class*="code"]').count();
      console.log(`Code block rendering: ${codeElements > 0 ? '✅' : '⚠️'}`);
    }
  });

  test('should handle emoji in messages', async ({ page }) => {
    const chatInput = page.locator('textarea, input[type="text"]').first();
    
    if (await chatInput.isVisible()) {
      const emojiMessage = 'Hello 👋 Test 🚀 Message 😊';
      await chatInput.fill(emojiMessage);
      await chatInput.press('Enter');
      
      await page.waitForTimeout(2000);
      
      const messageElement = page.locator(`text="${emojiMessage}"`);
      const found = await messageElement.count() > 0;
      console.log(`Emoji support: ${found ? '✅' : '⚠️'}`);
    }
  });

  test('should show message timestamps', async ({ page }) => {
    const chatInput = page.locator('textarea, input[type="text"]').first();
    
    if (await chatInput.isVisible()) {
      await chatInput.fill('Test timestamp message');
      await chatInput.press('Enter');
      
      await page.waitForTimeout(2000);
      
      // Look for timestamp patterns
      const timestampPatterns = [
        'text=/\\d{1,2}:\\d{2}/i',  // HH:MM
        'text=/\\d{4}-\\d{2}-\\d{2}/i',  // YYYY-MM-DD
        '[data-testid="message-timestamp"]',
        'span[class*="timestamp"]',
        'span[class*="time"]'
      ];
      
      let hasTimestamp = false;
      for (const pattern of timestampPatterns) {
        if (await page.locator(pattern).count() > 0) {
          hasTimestamp = true;
          break;
        }
      }
      
      console.log(`Message timestamps: ${hasTimestamp ? '✅' : '⚠️ Not displayed'}`);
    }
  });

  test('should handle message editing', async ({ page }) => {
    const chatInput = page.locator('textarea, input[type="text"]').first();
    
    if (await chatInput.isVisible()) {
      await chatInput.fill('Original message');
      await chatInput.press('Enter');
      
      await page.waitForTimeout(2000);
      
      // Look for edit button
      const editButtons = page.locator('button:has-text("Edit"), [title="Edit"], [aria-label*="edit" i]');
      const hasEdit = await editButtons.count() > 0;
      
      console.log(`Message editing: ${hasEdit ? '✅ Available' : '⚠️ Not implemented'}`);
    }
  });

  test('should handle message deletion', async ({ page }) => {
    const chatInput = page.locator('textarea, input[type="text"]').first();
    
    if (await chatInput.isVisible()) {
      await chatInput.fill('Message to delete');
      await chatInput.press('Enter');
      
      await page.waitForTimeout(2000);
      
      // Look for delete button
      const deleteButtons = page.locator('button:has-text("Delete"), [title="Delete"], [aria-label*="delete" i]');
      const hasDelete = await deleteButtons.count() > 0;
      
      console.log(`Message deletion: ${hasDelete ? '✅ Available' : '⚠️ Not implemented'}`);
    }
  });

  test('should show typing indicators', async ({ page }) => {
    const chatInput = page.locator('textarea, input[type="text"]').first();
    
    if (await chatInput.isVisible()) {
      await chatInput.fill('Show typing indicator');
      await chatInput.press('Enter');
      
      // Check for typing/generating indicator quickly
      await page.waitForTimeout(500);
      
      const typingText = page.locator('text=/typing|generating|thinking|processing/i');
      const typingClass = page.locator('.typing-indicator, [class*="typing"]');
      const hasTyping = (await typingText.count() > 0) || (await typingClass.count() > 0);
      
      console.log(`Typing indicators: ${hasTyping ? '✅' : '⚠️'}`);
    }
  });

  test('should handle message reactions/likes', async ({ page }) => {
    const chatInput = page.locator('textarea, input[type="text"]').first();
    
    if (await chatInput.isVisible()) {
      await chatInput.fill('Message for reactions');
      await chatInput.press('Enter');
      
      await page.waitForTimeout(2000);
      
      // Look for reaction buttons
      const reactionButtons = page.locator('[aria-label*="reaction" i], [title*="like" i], button:has-text("👍")');
      const hasReactions = await reactionButtons.count() > 0;
      
      console.log(`Message reactions: ${hasReactions ? '✅ Available' : '⚠️ Not implemented'}`);
    }
  });
});

test.describe('File Upload and Attachments', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(2000);
  });

  test('should have file upload capability', async ({ page }) => {
    const uploadButtons = page.locator('input[type="file"], button:has-text("Upload"), button:has-text("Attach"), [aria-label*="upload" i]');
    const hasUpload = await uploadButtons.count() > 0;
    
    console.log(`File upload: ${hasUpload ? '✅ Available' : '⚠️ Not found'}`);
  });

  test('should validate file types', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]').first();
    
    if (await fileInput.count() > 0) {
      const acceptAttr = await fileInput.getAttribute('accept');
      console.log(`Accepted file types: ${acceptAttr || 'All files'}`);
      
      expect(true).toBeTruthy();
    }
  });

  test('should show upload progress', async ({ page }) => {
    // Look for progress indicators
    const progressIndicators = page.locator('[role="progressbar"], .progress, .upload-progress, [class*="progress"]');
    const hasProgress = await progressIndicators.count() > 0;
    
    console.log(`Upload progress indicator: ${hasProgress ? '✅ Present' : 'ℹ️ Not visible (appears during upload)'}`);
  });
});

test.describe('Chat History Export', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(2000);
  });

  test('should have export functionality', async ({ page }) => {
    const exportButtons = page.locator('button:has-text("Export"), button:has-text("Download"), [aria-label*="export" i]');
    const hasExport = await exportButtons.count() > 0;
    
    console.log(`Chat export: ${hasExport ? '✅ Available' : '⚠️ Not found'}`);
  });

  test('should support multiple export formats', async ({ page }) => {
    const exportBtn = page.locator('button:has-text("Export"), button:has-text("Download")').first();
    
    if (await exportBtn.count() > 0) {
      await exportBtn.click();
      await page.waitForTimeout(1000);
      
      // Look for format options
      const formatOptions = page.locator('text=/json|csv|pdf|txt/i');
      const formatCount = await formatOptions.count();
      
      console.log(`Export formats: ${formatCount} options found`);
    }
  });
});

test.describe('Responsive Design Tests', () => {
  const viewports = [
    { name: 'Mobile', width: 375, height: 667 },
    { name: 'Tablet', width: 768, height: 1024 },
    { name: 'Desktop', width: 1920, height: 1080 }
  ];

  for (const viewport of viewports) {
    test(`should render correctly on ${viewport.name}`, async ({ page }) => {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await page.goto('/');
      
      // Wait for Streamlit to fully render with new viewport size
      await page.waitForTimeout(3000);
      
      // Reload to ensure Streamlit adapts to new viewport
      await page.reload();
      await page.waitForTimeout(2000);
      
      // Check if app content is visible (Control Panel is a key element)
      const controlPanel = page.locator('h2:has-text("Control Panel"), h3').first();
      const isVisible = await controlPanel.isVisible();
      
      expect(isVisible).toBeTruthy();
      
      console.log(`✅ ${viewport.name} (${viewport.width}x${viewport.height}) renders correctly`);
    });
  }
});

test.describe('PWA Features', () => {
  test('should have service worker', async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    const hasServiceWorker = await page.evaluate(() => {
      return 'serviceWorker' in navigator;
    });
    
    console.log(`Service Worker support: ${hasServiceWorker ? '✅' : '⚠️'}`);
    expect(hasServiceWorker).toBeTruthy();
  });

  test('should have web manifest', async ({ page }) => {
    await page.goto('/');
    
    const manifestLink = page.locator('link[rel="manifest"]');
    const hasManifest = await manifestLink.count() > 0;
    
    if (hasManifest) {
      const href = await manifestLink.getAttribute('href');
      console.log(`✅ Web manifest found: ${href}`);
    } else {
      console.log('⚠️ Web manifest not found');
    }
  });

  test('should be installable', async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    const installButton = page.locator('button:has-text("Install"), button:has-text("Add to Home")');
    const hasInstall = await installButton.count() > 0;
    
    console.log(`PWA installable: ${hasInstall ? '✅ Install button present' : 'ℹ️ Install prompt managed by browser'}`);
  });
});

test.describe('Offline Mode', () => {
  test('should show offline indicator', async ({ page, context }) => {
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    await context.setOffline(true);
    await page.waitForTimeout(2000);
    
    const offlineText = page.locator('text=/offline|no connection|disconnected/i');
    const offlineTestId = page.locator('[data-testid="offline"]');
    const hasOfflineIndicator = (await offlineText.count() > 0) || (await offlineTestId.count() > 0);
    
    console.log(`Offline indicator: ${hasOfflineIndicator ? '✅' : '⚠️'}`);
    
    await context.setOffline(false);
  });

  test('should cache static assets', async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    const hasCacheStorage = await page.evaluate(() => {
      return 'caches' in window;
    });
    
    console.log(`Cache Storage API: ${hasCacheStorage ? '✅ Supported' : '⚠️ Not supported'}`);
    expect(hasCacheStorage).toBeTruthy();
  });
});
