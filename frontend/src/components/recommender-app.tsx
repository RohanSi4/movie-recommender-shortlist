"use client";

import {
  ArrowDown,
  ArrowRight,
  Bookmark,
  CheckCircle2,
  Clapperboard,
  Code2,
  LoaderCircle,
  RefreshCw,
  Sparkles,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

import { MovieCard } from "@/components/movie-card";
import { MovieDetails } from "@/components/movie-details";
import { MovieSearch } from "@/components/movie-search";
import { PosterImage } from "@/components/poster-image";
import { SavedDrawer } from "@/components/saved-drawer";
import { checkHealth, getRecommendations } from "@/lib/api";
import { compactNumber, splitMovieTitle, strategyLabel } from "@/lib/format";
import type {
  HealthResponse,
  MovieRecommendation,
  MovieSummary,
  RankRequest,
  RequestStatus,
} from "@/lib/types";

const SAVED_KEY = "shortlist-saved-movies-v1";

const FEATURED_MOVIES: MovieSummary[] = [
  {
    movie_id: 2571,
    title: "Matrix, The (1999)",
    poster_url: "https://image.tmdb.org/t/p/w342/p96dm7sCMn4VYAStA6siNz30G1r.jpg",
    release_date: "1999-03-31",
  },
  {
    movie_id: 164179,
    title: "Arrival (2016)",
    poster_url: "https://image.tmdb.org/t/p/w342/iQBIXJprC8AN7Jwx7aOI0gPUqff.jpg",
    release_date: "2016-11-10",
  },
  {
    movie_id: 202439,
    title: "Parasite (2019)",
    poster_url: "https://image.tmdb.org/t/p/w342/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg",
    release_date: "2019-05-30",
  },
];

type ServiceState = "checking" | "online" | "offline";

export function RecommenderApp() {
  const [favorites, setFavorites] = useState<MovieSummary[]>([]);
  const [results, setResults] = useState<MovieRecommendation[]>([]);
  const [dismissedIds, setDismissedIds] = useState<number[]>([]);
  const [status, setStatus] = useState<RequestStatus>("idle");
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [strategy, setStrategy] = useState("");
  const [latency, setLatency] = useState<number | null>(null);
  const [lastRequest, setLastRequest] = useState<RankRequest | null>(null);
  const [profileId, setProfileId] = useState("123");
  const [serviceState, setServiceState] = useState<ServiceState>("checking");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [saved, setSaved] = useState<MovieRecommendation[]>([]);
  const [savedOpen, setSavedOpen] = useState(false);
  const [detailsMovie, setDetailsMovie] = useState<MovieRecommendation | null>(null);
  const rankController = useRef<AbortController | null>(null);
  const resultsRef = useRef<HTMLElement | null>(null);

  const savedIds = useMemo(() => new Set(saved.map((movie) => movie.movie_id)), [saved]);

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(SAVED_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as MovieRecommendation[];
        if (Array.isArray(parsed)) {
          setSaved(parsed);
        }
      }
    } catch {
      window.localStorage.removeItem(SAVED_KEY);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let retryTimer: number | undefined;
    const inspectService = async (attempt: number) => {
      try {
        const response = await checkHealth(controller.signal);
        setHealth(response);
        setServiceState(response.status === "ok" ? "online" : "offline");
      } catch {
        if (!controller.signal.aborted && attempt < 2) {
          retryTimer = window.setTimeout(() => void inspectService(attempt + 1), 4_000);
        } else if (!controller.signal.aborted) {
          setServiceState("offline");
        }
      }
    };
    void inspectService(0);
    return () => {
      controller.abort();
      if (retryTimer !== undefined) {
        window.clearTimeout(retryTimer);
      }
    };
  }, []);

  useEffect(() => {
    return () => rankController.current?.abort();
  }, []);

  const persistSaved = (next: MovieRecommendation[]) => {
    setSaved(next);
    window.localStorage.setItem(SAVED_KEY, JSON.stringify(next));
  };

  const toggleSaved = (movie: MovieRecommendation) => {
    if (savedIds.has(movie.movie_id)) {
      persistSaved(saved.filter((item) => item.movie_id !== movie.movie_id));
    } else {
      persistSaved([...saved, movie]);
    }
  };

  const runRecommendations = async (
    request: RankRequest,
    options: { append?: boolean; scroll?: boolean } = {}
  ) => {
    rankController.current?.abort();
    const controller = new AbortController();
    rankController.current = controller;
    const append = options.append ?? false;

    if (append) {
      setLoadingMore(true);
    } else {
      setStatus("loading");
      setResults([]);
      setDismissedIds([]);
      setStrategy("");
      setLatency(null);
    }
    setError(null);

    const exclusions = append
      ? Array.from(new Set([...results.map((movie) => movie.movie_id), ...dismissedIds]))
      : request.exclude_movie_ids;
    const payload = { ...request, exclude_movie_ids: exclusions, k: 12 };

    try {
      const response = await getRecommendations(payload, controller.signal);
      setResults((current) => (append ? [...current, ...response.results] : response.results));
      setStrategy(response.strategy);
      setLatency(response.latency_ms);
      setLastRequest(request);
      setStatus("success");
      if (options.scroll !== false) {
        window.setTimeout(
          () => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }),
          80
        );
      }
    } catch (caught) {
      if (!controller.signal.aborted) {
        setStatus("error");
        setError(
          caught instanceof Error
            ? caught.message
            : "The movie service could not finish that request."
        );
      }
    } finally {
      if (!controller.signal.aborted) {
        setLoadingMore(false);
      }
    }
  };

  const addFavorite = (movie: MovieSummary) => {
    setFavorites((current) => {
      if (current.some((item) => item.movie_id === movie.movie_id) || current.length >= 5) {
        return current;
      }
      return [...current, movie];
    });
  };

  const makeShortlist = (event: FormEvent) => {
    event.preventDefault();
    if (favorites.length === 0 || status === "loading") {
      return;
    }
    void runRecommendations({ movie_ids: favorites.map((movie) => movie.movie_id) });
  };

  const tryProfile = (event: FormEvent) => {
    event.preventDefault();
    const userId = Number(profileId);
    if (!Number.isInteger(userId) || userId <= 0) {
      setStatus("error");
      setError("Enter a positive MovieLens viewer ID.");
      return;
    }
    setFavorites([]);
    void runRecommendations({ user_id: userId });
  };

  const moreLikeThis = (movie: MovieRecommendation) => {
    const summary: MovieSummary = {
      movie_id: movie.movie_id,
      title: movie.title,
      poster_url: movie.poster_url,
      release_date: movie.release_date,
    };
    setDetailsMovie(null);
    setFavorites([summary]);
    void runRecommendations({ movie_ids: [movie.movie_id] });
  };

  const dismissMovie = (movie: MovieRecommendation) => {
    setDismissedIds((current) => [...current, movie.movie_id]);
    setResults((current) => current.filter((item) => item.movie_id !== movie.movie_id));
  };

  const reset = () => {
    rankController.current?.abort();
    setFavorites([]);
    setResults([]);
    setDismissedIds([]);
    setStatus("idle");
    setError(null);
    setStrategy("");
    setLatency(null);
    setLastRequest(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const visibleCatalog = compactNumber(health?.catalog_size) ?? "87K";

  return (
    <>
      <header className="site-header">
        <a href="#top" className="brand" aria-label="Shortlist home">
          <span><Clapperboard size={18} /></span>
          Shortlist
        </a>
        <nav aria-label="Project links">
          <span className="service-pill" data-state={serviceState}>
            <i />
            {serviceState === "checking" ? "Checking the model" : serviceState === "online" ? "Model ready" : "Model waking up"}
          </span>
          <button type="button" className="saved-button" onClick={() => setSavedOpen(true)}>
            <Bookmark size={16} /> Saved {saved.length > 0 && <b>{saved.length}</b>}
          </button>
        </nav>
      </header>

      <main id="main-content">
        <section className="hero relative" id="top">
          <div className="hero-copy">
            <p className="eyebrow"><Sparkles size={14} /> Movie recommendations that start with you</p>
            <h1>Find something worth watching.</h1>
            <p className="hero-lede">
              Pick a few movies you already love. I&apos;ll blend their taste patterns and look through {visibleCatalog} movies to make you a fresh shortlist.
            </p>
          </div>

          <form className="taste-builder" onSubmit={makeShortlist} aria-busy={status === "loading"}>
            <div className="builder-heading">
              <div>
                <span>01</span>
                <h2>Build your movie mix</h2>
              </div>
              <p>One movie works. Three or more gives the model a much better read.</p>
            </div>
            <MovieSearch
              selected={favorites}
              onSelect={addFavorite}
              onRemove={(movieId) => setFavorites((current) => current.filter((movie) => movie.movie_id !== movieId))}
            />
            <div className="builder-submit">
              <button
                type="submit"
                className="primary-button primary-button-large"
                disabled={favorites.length === 0 || status === "loading"}
              >
                {status === "loading" ? (
                  <><LoaderCircle className="animate-spin" size={18} /> Finding the good stuff...</>
                ) : (
                  <>Make my shortlist <ArrowRight size={18} /></>
                )}
              </button>
              <span>{favorites.length}/5 picked</span>
            </div>
          </form>

          <a className="scroll-cue" href="#recommendations">
            See how it works <ArrowDown size={15} />
          </a>
        </section>

        <section className="recommendations-section" id="recommendations" ref={resultsRef}>
          {status === "idle" && results.length === 0 && (
            <div className="empty-showcase">
              <div>
                <p className="eyebrow">Start anywhere</p>
                <h2>A favorite is all the model needs.</h2>
                <p>
                  Pick one movie for a focused match, or mix a few together to find the overlap in your taste.
                </p>
                <ol>
                  <li><span>1</span> Pick up to five favorites</li>
                  <li><span>2</span> The model blends their learned signals</li>
                  <li><span>3</span> Save the movies that make the cut</li>
                </ol>
              </div>
              <div className="featured-posters" aria-label="Starter movie examples">
                {FEATURED_MOVIES.map((movie) => (
                  <button type="button" key={movie.movie_id} onClick={() => addFavorite(movie)}>
                    <PosterImage
                      src={movie.poster_url}
                      alt={`Start with ${splitMovieTitle(movie.title).title}`}
                      sizes="(max-width: 640px) 28vw, 180px"
                    />
                    <span>{splitMovieTitle(movie.title).title}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {status === "loading" && (
            <div className="loading-state" role="status" aria-live="polite">
              <LoaderCircle className="animate-spin" size={30} />
              <div>
                <h2>Finding the good stuff...</h2>
                <p>If the service was sleeping, this first search can take a little longer.</p>
              </div>
            </div>
          )}

          {status === "error" && (
            <div className="error-state" role="alert">
              <div>
                <p className="eyebrow">That one did not land</p>
                <h2>Let&apos;s try it once more.</h2>
                <p>{error}</p>
              </div>
              {lastRequest && (
                <button type="button" className="secondary-button" onClick={() => void runRecommendations(lastRequest)}>
                  <RefreshCw size={16} /> Try again
                </button>
              )}
            </div>
          )}

          {status === "success" && results.length > 0 && (
            <>
              <div className="results-heading">
                <div>
                  <p className="eyebrow"><CheckCircle2 size={14} /> Your shortlist is ready</p>
                  <h2>{favorites.length > 0 ? "Movies that fit your mix" : "Movies for this viewer"}</h2>
                  <p>{strategyLabel(strategy)}{latency !== null ? ` in ${latency} ms.` : "."}</p>
                </div>
                <button type="button" className="text-button" onClick={reset}>Start over</button>
              </div>

              <div className="movie-grid" aria-live="polite">
                {results.map((movie, index) => (
                  <MovieCard
                    key={movie.movie_id}
                    movie={movie}
                    rank={index + 1}
                    saved={savedIds.has(movie.movie_id)}
                    onSave={toggleSaved}
                    onDetails={setDetailsMovie}
                    onMoreLikeThis={moreLikeThis}
                    onDismiss={dismissMovie}
                  />
                ))}
              </div>

              <div className="more-row">
                <button
                  type="button"
                  className="secondary-button"
                  disabled={!lastRequest || loadingMore}
                  onClick={() => lastRequest && void runRecommendations(lastRequest, { append: true, scroll: false })}
                >
                  {loadingMore ? <LoaderCircle className="animate-spin" size={17} /> : <RefreshCw size={17} />}
                  {loadingMore ? "Finding more..." : "Show me more"}
                </button>
                <p>Already seen something? Tap the minus and it will stay out of the next batch.</p>
              </div>
            </>
          )}
        </section>

        <section className="proof-section">
          <div className="proof-copy">
            <p className="eyebrow">A real model behind a simple screen</p>
            <h2>Not a hardcoded list in a nice jacket.</h2>
            <p>
              Shortlist trains on the same one-to-five favorites flow you use here, retrieves against the full catalog in Go, and adds TMDB details so the results feel like movies instead of database rows.
            </p>
            <div className="proof-links">
              <a href="https://github.com/RohanSi4/movie-recommender-shortlist" target="_blank" rel="noreferrer">
                <Code2 size={17} /> View the code
              </a>
              <a href="https://rohansingh04.com/projects/movie-recommender" target="_blank" rel="noreferrer">
                Read the case study <ArrowRight size={16} />
              </a>
            </div>
          </div>
          <dl className="proof-stats">
            <div><dt>Movies searched</dt><dd>87,585</dd></div>
            <div><dt>Learned profiles</dt><dd>186,458</dd></div>
            <div><dt>HitRate@10</dt><dd>84.1%<small>five-favorite test flow</small></dd></div>
            <div><dt>Recall@100</dt><dd>0.331<small>vs. 0.228 popularity</small></dd></div>
          </dl>
        </section>

        <section className="model-playground">
          <details>
            <summary>
              <span>
                <strong>Want to poke at the model?</strong>
                Try an anonymous MovieLens viewer profile.
              </span>
              <ArrowDown size={18} />
            </summary>
            <form onSubmit={tryProfile}>
              <label htmlFor="profile-id">Anonymous viewer ID</label>
              <input
                id="profile-id"
                inputMode="numeric"
                value={profileId}
                onChange={(event) => setProfileId(event.target.value)}
                placeholder="123"
              />
              <button type="submit" className="secondary-button" disabled={status === "loading"}>
                Try this profile <ArrowRight size={16} />
              </button>
              <p>
                Known profiles use their rating history. New IDs honestly fall back to popular starter picks.
              </p>
            </form>
          </details>
        </section>
      </main>

      <footer>
        <a href="#top" className="brand"><span><Clapperboard size={16} /></span> Shortlist</a>
        <p>Built by Rohan Singh with MovieLens, TMDB, Python, Go, and Next.js.</p>
      </footer>

      {detailsMovie && (
        <MovieDetails
          movie={detailsMovie}
          saved={savedIds.has(detailsMovie.movie_id)}
          onClose={() => setDetailsMovie(null)}
          onSave={toggleSaved}
          onMoreLikeThis={moreLikeThis}
        />
      )}
      {savedOpen && (
        <SavedDrawer
          movies={saved}
          onClose={() => setSavedOpen(false)}
          onRemove={toggleSaved}
          onClear={() => persistSaved([])}
        />
      )}
    </>
  );
}
