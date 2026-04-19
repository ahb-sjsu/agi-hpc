// Headless render smoke test for schematic.html.
//
// Catches the silent-regression class where HTML is correct but JS
// rendering fails — panels go blank without any server error. The
// deploy-smoke CI workflow checks raw HTML; this checks rendered state.

const { test, expect } = require('@playwright/test');

test.describe('Atlas dashboard — schematic.html', () => {
  test('loads without console errors', async ({ page }) => {
    const errors = [];
    page.on('pageerror', (err) => errors.push(String(err)));
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
    });

    await page.goto('/schematic.html', { waitUntil: 'networkidle', timeout: 30_000 });
    // Filter out known-noisy errors that don't affect rendering.
    const significant = errors.filter(
      (e) => !/favicon|net::ERR_ABORTED|blocked by CORS/i.test(e)
    );
    expect(significant, `console errors: ${significant.join('\n')}`).toHaveLength(0);
  });

  test('key widgets are present in the DOM', async ({ page }) => {
    await page.goto('/schematic.html', { waitUntil: 'domcontentloaded' });

    // Static widgets — these should be in the HTML immediately
    await expect(page.locator('text=NATS Topology').first()).toBeVisible();
    await expect(page.locator('text=NATS Live').first()).toBeVisible();
    await expect(page.locator('text=NRP Burst Jobs').first()).toBeVisible();
    await expect(page.locator('text=Erebus Cognitive Architecture').first()).toBeVisible();
  });

  test('NATS topology SVG renders nodes after polling', async ({ page }) => {
    await page.goto('/schematic.html', { waitUntil: 'networkidle' });
    // Polling runs every ~2s; give it up to 15s to populate.
    const svg = page.locator('#nats-topo-svg');
    await expect(svg).toBeVisible();
    await expect
      .poll(async () => svg.locator('g, circle, text, rect').count(), { timeout: 15_000 })
      .toBeGreaterThan(0);
  });

  test('NRP burst table shows at least one row after polling', async ({ page }) => {
    await page.goto('/schematic.html', { waitUntil: 'networkidle' });
    // Erebus worker pool should always produce at least one row.
    await expect
      .poll(
        async () => page.locator('#nrp-jobs-table tr').count(),
        { timeout: 15_000 }
      )
      .toBeGreaterThan(0);
  });

  test('version stamp reflects a recent commit', async ({ page }) => {
    await page.goto('/schematic.html', { waitUntil: 'domcontentloaded' });
    const stampText = await page.locator('text=/ui:[a-f0-9]{6,}/').first().innerText();
    // e.g. "ui:1efd2dc · 2026-04-19T02:33Z"
    expect(stampText).toMatch(/^ui:[a-f0-9]{6,}\s*·\s*\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z$/);
  });
});
