"use client";

import { Check, Clipboard, Trash2, X } from "lucide-react";
import { useEffect, useState } from "react";

import { PosterImage } from "@/components/poster-image";
import { movieYear, splitMovieTitle } from "@/lib/format";
import type { MovieRecommendation } from "@/lib/types";

type SavedDrawerProps = {
  movies: MovieRecommendation[];
  onClose: () => void;
  onRemove: (movie: MovieRecommendation) => void;
  onClear: () => void;
};

export function SavedDrawer({ movies, onClose, onRemove, onClear }: SavedDrawerProps) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = "";
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [onClose]);

  const copyList = async () => {
    const list = movies
      .map((movie, index) => `${index + 1}. ${splitMovieTitle(movie.title).title}${movieYear(movie.title, movie.release_date) ? ` (${movieYear(movie.title, movie.release_date)})` : ""}`)
      .join("\n");
    await navigator.clipboard.writeText(`My Shortlist\n\n${list}`);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  };

  return (
    <div className="drawer-backdrop" onMouseDown={onClose}>
      <aside
        className="saved-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="saved-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="drawer-header">
          <div>
            <p className="eyebrow">Keep the good ones</p>
            <h2 id="saved-title">My shortlist</h2>
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="Close saved movies" autoFocus>
            <X size={20} />
          </button>
        </div>

        {movies.length === 0 ? (
          <div className="drawer-empty">
            <p>Your saved movies will land here.</p>
            <span>Tap the bookmark on any recommendation so you do not lose it.</span>
          </div>
        ) : (
          <>
            <ol className="saved-list">
              {movies.map((movie) => {
                const clean = splitMovieTitle(movie.title);
                return (
                  <li key={movie.movie_id}>
                    <span className="saved-poster">
                      <PosterImage src={movie.poster_url} alt="" sizes="52px" />
                    </span>
                    <span className="saved-copy">
                      <strong>{clean.title}</strong>
                      <small>{movieYear(movie.title, movie.release_date) ?? "Year unknown"}</small>
                    </span>
                    <button
                      type="button"
                      className="icon-button"
                      onClick={() => onRemove(movie)}
                      aria-label={`Remove ${clean.title}`}
                    >
                      <Trash2 size={17} />
                    </button>
                  </li>
                );
              })}
            </ol>
            <div className="drawer-actions">
              <button type="button" className="primary-button" onClick={copyList}>
                {copied ? <Check size={17} /> : <Clipboard size={17} />}
                {copied ? "Copied" : "Copy my list"}
              </button>
              <button type="button" className="text-button" onClick={onClear}>
                Clear all
              </button>
            </div>
          </>
        )}
      </aside>
    </div>
  );
}
