export type MovieSummary = {
  movie_id: number;
  title: string;
  poster_url?: string;
  release_date?: string;
};

export type MovieRecommendation = MovieSummary & {
  score: number;
  genres?: string[];
  vote_average?: number;
  overview?: string;
  reasons?: string[];
};

export type RankRequest = {
  movie_id?: number;
  movie_ids?: number[];
  user_id?: number;
  exclude_movie_ids?: number[];
  k?: number;
};

export type RankResponse = {
  user_id?: number;
  movie_id?: number;
  movie_ids?: number[];
  strategy: string;
  results: MovieRecommendation[];
  latency_ms: number;
};

export type HealthResponse = {
  status: string;
  retrieval_ready: boolean;
  model_run: string;
  catalog_size: number;
  profile_count: number;
};

export type RequestStatus = "idle" | "loading" | "success" | "error";
