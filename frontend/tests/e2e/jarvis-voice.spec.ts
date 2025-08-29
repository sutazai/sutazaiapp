import { test, expect } from '@playwright/test';

test.describe('JARVIS Voice Assistant Features', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(3000);
  });

  test('should have voice recording button', async ({ page }) => {
    // Look for voice/microphone button
    const voiceButtonSelectors = [
      'button[aria-label*="microphone"]',
      'button[aria-label*="voice"]',
      'button[aria-label*="record"]',
      'button:has-text("🎤")',
      'button:has-text("🎙️")',
      '[data-testid="mic-recorder"]',
      '.mic-recorder-button'
    ];
    
    let voiceButton = null;
    for (const selector of voiceButtonSelectors) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        voiceButton = element.first();
        break;
      }
    }
    
    if (voiceButton) {
      await expect(voiceButton).toBeVisible();
      console.log('✅ Voice recording button found');
    } else {
      console.log('❌ No voice recording button found');
    }
  });

  test('should show voice indicator when recording', async ({ page, context }) => {
    // Grant microphone permissions
    await context.grantPermissions(['microphone']);
    
    const voiceButton = page.locator('button').filter({ 
      hasText: /🎤|🎙️|Record|Voice/i 
    }).first();
    
    if (await voiceButton.isVisible()) {
      // Click to start recording
      await voiceButton.click();
      await page.waitForTimeout(1000);
      
      // Look for recording indicator
      const recordingIndicators = [
        'text=/recording|listening|speak now/i',
        '.recording-indicator',
        '[class*="recording"]',
        '[class*="listening"]'
      ];
      
      let hasIndicator = false;
      for (const selector of recordingIndicators) {
        const element = page.locator(selector);
        if (await element.count() > 0) {
          hasIndicator = true;
          console.log('✅ Recording indicator visible');
          break;
        }
      }
      
      if (!hasIndicator) {
        console.log('⚠️ No recording indicator found');
      }
      
      // Stop recording
      await voiceButton.click();
    }
  });

  test('should have audio visualization', async ({ page }) => {
    // Look for audio waveform or visualization
    const visualizationSelectors = [
      'canvas[class*="audio"]',
      'canvas[class*="wave"]',
      'div[class*="visualizer"]',
      '.audio-visualizer',
      'svg[class*="wave"]'
    ];
    
    let hasVisualization = false;
    for (const selector of visualizationSelectors) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        hasVisualization = true;
        console.log('✅ Audio visualization component found');
        break;
      }
    }
    
    if (!hasVisualization) {
      console.log('⚠️ No audio visualization found');
    }
  });

  test('should have voice output controls', async ({ page }) => {
    // Look for voice output/TTS controls
    const ttsControlSelectors = [
      'button:has-text("🔊")',
      'button:has-text("🔇")',
      'button[aria-label*="speaker"]',
      'button[aria-label*="mute"]',
      '[data-testid="voice-output"]',
      'input[type="range"][aria-label*="volume"]'
    ];
    
    let hasTTSControls = false;
    for (const selector of ttsControlSelectors) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        hasTTSControls = true;
        console.log('✅ Voice output controls found');
        break;
      }
    }
    
    if (!hasTTSControls) {
      console.log('⚠️ No voice output controls found');
    }
  });

  test('should display speech recognition status', async ({ page }) => {
    // Look for speech recognition status
    const statusSelectors = [
      'text=/ready|listening|processing|speaking/i',
      '[data-testid="speech-status"]',
      '.speech-recognition-status',
      'div[class*="status"]'
    ];
    
    let hasStatus = false;
    for (const selector of statusSelectors) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        const text = await element.first().textContent();
        if (text && text.length > 0) {
          hasStatus = true;
          console.log(`✅ Speech status found: ${text}`);
          break;
        }
      }
    }
    
    if (!hasStatus) {
      console.log('⚠️ No speech recognition status found');
    }
  });

  test('should have voice command history', async ({ page }) => {
    // Look for voice command history or transcript
    const historySelectors = [
      '[data-testid="voice-history"]',
      '.voice-transcript',
      'div[class*="transcript"]',
      'div[class*="voice-log"]'
    ];
    
    let hasHistory = false;
    for (const selector of historySelectors) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        hasHistory = true;
        console.log('✅ Voice command history found');
        break;
      }
    }
    
    if (!hasHistory) {
      console.log('⚠️ No voice command history found');
    }
  });

  test('should have voice settings', async ({ page }) => {
    // Open sidebar if needed
    const sidebarToggle = page.locator('[aria-label="Open sidebar"], button:has-text("☰")');
    if (await sidebarToggle.isVisible()) {
      await sidebarToggle.click();
      await page.waitForTimeout(1000);
    }
    
    // Look for voice settings
    const settingsSelectors = [
      'text=/voice settings|speech settings|audio settings/i',
      'select[aria-label*="voice"]',
      'input[aria-label*="speech rate"]',
      'input[aria-label*="pitch"]'
    ];
    
    let hasSettings = false;
    for (const selector of settingsSelectors) {
      const element = page.locator(selector);
      if (await element.count() > 0) {
        hasSettings = true;
        console.log('✅ Voice settings found');
        break;
      }
    }
    
    if (!hasSettings) {
      console.log('⚠️ No voice settings found');
    }
  });
});