import type {
  HealthResponse,
  MovieSummary,
  RankRequest,
  RankResponse,
} from "@/lib/types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8080";

export class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function requestJSON<T>(
  path: string,
  init: RequestInit = {},
  timeoutMs = 30_000
): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  const externalSignal = init.signal;
  const abortFromExternal = () => controller.abort();
  externalSignal?.addEventListener("abort", abortFromExternal, { once: true });

  try {
    const response = await fetch(`${API_BASE}${path}`, {
      ...init,
      signal: controller.signal,
    });
    if (!response.ok) {
      const body = (await response.json().catch(() => null)) as
        | { error?: string }
        | null;
      throw new ApiError(body?.error ?? "The movie service could not finish that request.", response.status);
    }
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    if (controller.signal.aborted && !externalSignal?.aborted) {
      throw new ApiError("The movie service is taking longer than expected. Try once more in a moment.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
    externalSignal?.removeEventListener("abort", abortFromExternal);
  }
}

export async function checkHealth(signal?: AbortSignal) {
  return requestJSON<HealthResponse>("/health", { signal }, 15_000);
}

export async function searchMovies(query: string, signal?: AbortSignal) {
  const results = await requestJSON<MovieSummary[]>(
    `/search?q=${encodeURIComponent(query)}&limit=8`,
    { signal },
    15_000
  );
  return Array.isArray(results) ? results : [];
}

export async function getRecommendations(
  request: RankRequest,
  signal?: AbortSignal
) {
  const response = await requestJSON<RankResponse>(
    "/rank",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal,
    },
    45_000
  );
  if (!Array.isArray(response.results)) {
    throw new ApiError("The movie service returned an unexpected response.");
  }
  return response;
}
