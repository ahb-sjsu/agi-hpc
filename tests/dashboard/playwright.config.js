// Playwright config for the Atlas dashboard render tests.
// Runs against the live URL (override via DASHBOARD_URL env).
const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: '.',
  timeout: 60_000,
  retries: 1,
  reporter: [['list']],
  use: {
    baseURL: process.env.DASHBOARD_URL || 'https://atlas-sjsu.duckdns.org',
    ignoreHTTPSErrors: true,
    trace: 'retain-on-failure',
  },
  projects: [
    { name: 'chromium', use: { browserName: 'chromium' } },
  ],
});
