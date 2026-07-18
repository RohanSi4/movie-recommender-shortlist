"use client";

import Image from "next/image";
import { LoaderCircle, Plus, Search, X } from "lucide-react";
import { useEffect, useId, useMemo, useState } from "react";

import { searchMovies } from "@/lib/api";
import { movieYear, splitMovieTitle } from "@/lib/format";
import type { MovieSummary } from "@/lib/types";

const STARTER_MOVIES: MovieSummary[] = [
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
  {
    movie_id: 5618,
    title: "Spirited Away (Sen to Chihiro no kamikakushi) (2001)",
    poster_url: "https://image.tmdb.org/t/p/w342/39wmItIWsg5sZMyRUHLkWBcuVCM.jpg",
    release_date: "2001-07-20",
  },
  {
    movie_id: 58559,
    title: "Dark Knight, The (2008)",
    poster_url: "https://image.tmdb.org/t/p/w342/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
    release_date: "2008-07-16",
  },
];

type MovieSearchProps = {
  selected: MovieSummary[];
  onSelect: (movie: MovieSummary) => void;
  onRemove: (movieId: number) => void;
};

export function MovieSearch({ selected, onSelect, onRemove }: MovieSearchProps) {
  const generatedId = useId().replaceAll(":", "");
  const listboxId = `movie-options-${generatedId}`;
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState<MovieSummary[]>([]);
  const [searchState, setSearchState] = useState<"idle" | "loading" | "error">("idle");
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);

  const atLimit = selected.length >= 5;
  const selectedIds = useMemo(
    () => new Set(selected.map((movie) => movie.movie_id)),
    [selected]
  );

  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < 2 || atLimit) {
      return;
    }

    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      setSearchState("loading");
      try {
        const results = await searchMovies(trimmed, controller.signal);
        setSuggestions(results.filter((movie) => !selectedIds.has(movie.movie_id)));
        setActiveIndex(-1);
        setOpen(true);
        setSearchState("idle");
      } catch {
        if (!controller.signal.aborted) {
          setSuggestions([]);
          setSearchState("error");
          setOpen(true);
        }
      }
    }, 220);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [atLimit, query, selectedIds]);

  const chooseMovie = (movie: MovieSummary) => {
    onSelect(movie);
    setQuery("");
    setSuggestions([]);
    setOpen(false);
    setActiveIndex(-1);
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (!open || suggestions.length === 0) {
      if (event.key === "Escape") {
        setOpen(false);
      }
      return;
    }
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveIndex((current) => (current + 1) % suggestions.length);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveIndex((current) =>
        current <= 0 ? suggestions.length - 1 : current - 1
      );
    } else if (event.key === "Enter" && activeIndex >= 0) {
      event.preventDefault();
      chooseMovie(suggestions[activeIndex]);
    } else if (event.key === "Escape") {
      setOpen(false);
    }
  };

  return (
    <div className="space-y-5">
      <div>
        <label htmlFor={`movie-search-${generatedId}`} className="field-label">
          Search for a movie you love
        </label>
        <div className="search-shell">
          <Search aria-hidden="true" size={20} />
          <input
            id={`movie-search-${generatedId}`}
            role="combobox"
            aria-autocomplete="list"
            aria-controls={listboxId}
            aria-expanded={open}
            aria-activedescendant={
              activeIndex >= 0 ? `${listboxId}-${activeIndex}` : undefined
            }
            value={query}
            onChange={(event) => {
              const nextQuery = event.target.value;
              setQuery(nextQuery);
              if (nextQuery.trim().length < 2) {
                setSuggestions([]);
                setSearchState("idle");
              }
              setOpen(true);
            }}
            onFocus={() => suggestions.length > 0 && setOpen(true)}
            onBlur={() => window.setTimeout(() => setOpen(false), 120)}
            onKeyDown={handleKeyDown}
            disabled={atLimit}
            placeholder={atLimit ? "You picked five. Your mix is ready." : "Try Arrival, Parasite, or The Matrix"}
            autoComplete="off"
          />
          {searchState === "loading" && (
            <LoaderCircle className="animate-spin text-accent" aria-label="Searching" size={19} />
          )}

          {open && query.trim().length >= 2 && (
            <div className="search-menu" role="listbox" id={listboxId}>
              {searchState === "error" ? (
                <p role="alert" className="search-message">
                  Search is having a moment. Give it another try.
                </p>
              ) : searchState !== "loading" && suggestions.length === 0 ? (
                <p className="search-message">No close matches yet. Try a shorter title.</p>
              ) : (
                suggestions.map((movie, index) => {
                  const clean = splitMovieTitle(movie.title);
                  return (
                    <button
                      type="button"
                      role="option"
                      aria-selected={index === activeIndex}
                      id={`${listboxId}-${index}`}
                      key={movie.movie_id}
                      className="search-option"
                      data-active={index === activeIndex}
                      onMouseDown={(event) => event.preventDefault()}
                      onClick={() => chooseMovie(movie)}
                    >
                      <span className="search-option-poster">
                        <Image
                          src={movie.poster_url || "/poster-fallback.svg"}
                          alt=""
                          fill
                          sizes="44px"
                        />
                      </span>
                      <span className="min-w-0 flex-1">
                        <strong>{clean.title}</strong>
                        <small>{movieYear(movie.title, movie.release_date) ?? "Year unknown"}</small>
                      </span>
                      <Plus aria-hidden="true" size={18} />
                    </button>
                  );
                })
              )}
            </div>
          )}
        </div>
      </div>

      {!atLimit && (
        <div className="starter-row" aria-label="Quick picks">
          <span>{selected.length === 0 ? "Or start with" : "Quick add"}</span>
          {STARTER_MOVIES.filter((movie) => !selectedIds.has(movie.movie_id)).map((movie) => (
            <button type="button" key={movie.movie_id} onClick={() => chooseMovie(movie)}>
              {splitMovieTitle(movie.title).title}
            </button>
          ))}
        </div>
      )}
      {selected.length > 0 && (
        <div className="selected-movies" aria-label="Your favorite movies">
          {selected.map((movie) => (
            <div className="selected-movie" key={movie.movie_id}>
              <span className="selected-poster">
                <Image
                  src={movie.poster_url || "/poster-fallback.svg"}
                  alt=""
                  fill
                  sizes="40px"
                />
              </span>
              <span>{splitMovieTitle(movie.title).title}</span>
              <button
                type="button"
                onClick={() => onRemove(movie.movie_id)}
                aria-label={`Remove ${splitMovieTitle(movie.title).title}`}
              >
                <X size={15} />
              </button>
            </div>
          ))}
          {selected.length < 5 && <p>Add up to {5 - selected.length} more to sharpen the mix.</p>}
        </div>
      )}
    </div>
  );
}
