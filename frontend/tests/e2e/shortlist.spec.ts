import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test("builds a personal movie mix and saves a result", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "Find something worth watching." })).toBeVisible();
  await expect(page.getByText("Model ready")).toBeVisible({ timeout: 20_000 });

  const builder = page.locator(".taste-builder");
  await builder.getByRole("button", { name: "Arrival", exact: true }).click();
  await builder.getByRole("button", { name: "Parasite", exact: true }).click();
  await builder.getByRole("button", { name: "Make my shortlist" }).click();

  await expect(page.getByRole("heading", { name: "Movies that fit your mix" })).toBeVisible({ timeout: 30_000 });
  const cards = page.locator(".movie-card");
  await expect(cards).toHaveCount(12);
  await expect(cards.first().getByText("Fits your movie mix")).toBeVisible();

  await cards.first().getByRole("button", { name: /^Save / }).click();
  await page.getByRole("button", { name: /Saved 1/ }).click();
  await expect(page.getByRole("heading", { name: "My shortlist" })).toBeVisible();
  await expect(page.locator(".saved-list li")).toHaveCount(1);
});

test("movie search supports keyboard selection", async ({ page }) => {
  await page.goto("/");
  const search = page.getByRole("combobox", { name: "Search for a movie you love" });
  await search.fill("Arrival");
  await expect(page.getByRole("option", { name: /Arrival.*2016/ })).toBeVisible({ timeout: 20_000 });
  await search.press("ArrowDown");
  await search.press("Enter");
  await expect(page.locator(".selected-movie")).toHaveCount(1);
});

test("initial experience has no serious accessibility violations", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Find something worth watching." })).toBeVisible();
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations).toEqual([]);
});
