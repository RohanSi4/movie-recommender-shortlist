const REASON_LABELS: Record<string, string> = {
  learned_movie_similarity: "Similar taste patterns",
  similar_genres: "Similar genres",
  popular_in_movielens: "A crowd favorite",
  high_vote_avg: "Highly rated",
  learned_from_user_history: "Fits this viewer",
  matches_user_taste: "Close to their ratings",
  matches_your_movie_mix: "Fits your movie mix",
  shares_genres_with_your_picks: "Shares your genres",
  well_reviewed: "Well reviewed",
  similar_ratings: "Similar ratings",
};

export function reasonLabel(reason: string) {
  return REASON_LABELS[reason] ?? reason.replaceAll("_", " ");
}

export function splitMovieTitle(title: string) {
  const match = title.match(/^(.*) \((\d{4})\)$/);
  if (!match) {
    return { title, year: null };
  }
  return { title: match[1], year: match[2] };
}

export function movieYear(title: string, releaseDate?: string) {
  if (releaseDate && /^\d{4}/.test(releaseDate)) {
    return releaseDate.slice(0, 4);
  }
  return splitMovieTitle(title).year;
}

export function compactNumber(value?: number) {
  if (value === undefined || !Number.isFinite(value)) {
    return null;
  }
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

export function strategyLabel(strategy: string) {
  if (strategy === "two_tower_taste_mix") {
    return "Blended from the movies you picked";
  }
  if (strategy === "two_tower_movie_similarity") {
    return "Learned from similar viewing patterns";
  }
  if (strategy === "two_tower" || strategy === "two_tower_then_lightgbm") {
    return "Built from this viewer's history";
  }
  if (strategy === "popularity_fallback") {
    return "Popular starter picks for a new viewer";
  }
  return "Matched with movie and audience signals";
}
