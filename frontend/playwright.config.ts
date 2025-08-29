import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for JARVIS UI testing
 */
export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',
  
  use: {
    baseURL: 'http://localhost:11000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  webServer: {
    command: 'cd /opt/sutazaiapp/frontend && ./venv/bin/streamlit run app.py --server.port 11000 --server.address 0.0.0.0',
    url: 'http://localhost:11000',
    reuseExistingServer: true,
    timeout: 120 * 1000,
  },
});