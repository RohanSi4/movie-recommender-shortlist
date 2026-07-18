import { expect, test } from "@playwright/test";

test("mobile layout stays within the viewport and completes a recommendation", async ({ page }) => {
  await page.goto("/");
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  const documentWidth = await page.evaluate(() => document.documentElement.scrollWidth);
  expect(documentWidth).toBeLessThanOrEqual(viewportWidth);

  const builder = page.locator(".taste-builder");
  await builder.getByRole("button", { name: "Arrival", exact: true }).click();
  await builder.getByRole("button", { name: "Make my shortlist" }).click();
  await expect(page.getByRole("heading", { name: "Movies that fit your mix" })).toBeVisible({ timeout: 30_000 });
  await expect(page.locator(".movie-card").first()).toBeVisible();

  const finalDocumentWidth = await page.evaluate(() => document.documentElement.scrollWidth);
  expect(finalDocumentWidth).toBeLessThanOrEqual(viewportWidth);
});
