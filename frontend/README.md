# Shortlist frontend

The product layer for Shortlist. Visitors pick one to five movies they love,
get a fresh set of recommendations, open richer details, dismiss misses, and
save a personal shortlist in their browser.

## Run it

Start the Go service from the repository root:

~~~bash
make service
~~~

Then start Next.js:

~~~bash
cd frontend
npm install
npm run dev
~~~

The default API target is `http://localhost:8080`. Override it with:

~~~bash
NEXT_PUBLIC_API_BASE=https://your-api.example.com npm run dev
~~~

Vercel uses the same `NEXT_PUBLIC_API_BASE` variable for production.

## Product behavior

- Search is abortable and fully keyboard operable.
- One to five chosen movies are sent as `movie_ids` to build a temporary taste
  vector. Nothing about the visitor is stored by the API.
- `Show me more` excludes every result already shown so repeated batches stay
  fresh.
- Saved movies live in `localStorage` and can be copied as a plain-text list.
- The UI reads the returned strategy and labels popularity fallback honestly.
- Health checks retry while the Render service wakes up.
- TMDB posters use Next Image with a designed local fallback.

## Quality checks

~~~bash
npm run lint
npm run typecheck
npm run build
npm run test:e2e
~~~

Playwright covers the main taste flow, saved movies, keyboard autocomplete,
mobile overflow, and Axe accessibility checks. The root GitHub Actions workflow
runs all of these on every push and pull request.
