"use client";

import { Bookmark, BookmarkCheck, Sparkles, Star, X } from "lucide-react";
import { useEffect } from "react";

import { PosterImage } from "@/components/poster-image";
import { movieYear, reasonLabel, splitMovieTitle } from "@/lib/format";
import type { MovieRecommendation } from "@/lib/types";

type MovieDetailsProps = {
  movie: MovieRecommendation;
  saved: boolean;
  onClose: () => void;
  onSave: (movie: MovieRecommendation) => void;
  onMoreLikeThis: (movie: MovieRecommendation) => void;
};

export function MovieDetails({
  movie,
  saved,
  onClose,
  onSave,
  onMoreLikeThis,
}: MovieDetailsProps) {
  const clean = splitMovieTitle(movie.title);
  const year = movieYear(movie.title, movie.release_date);

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

  return (
    <div className="dialog-backdrop" onMouseDown={onClose}>
      <div
        className="movie-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="movie-dialog-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <button type="button" className="dialog-close" onClick={onClose} aria-label="Close details" autoFocus>
          <X size={20} />
        </button>
        <div className="dialog-poster">
          <PosterImage
            src={movie.poster_url}
            alt={`Poster for ${clean.title}`}
            sizes="(max-width: 640px) 100vw, 360px"
          />
        </div>
        <div className="dialog-copy">
          <p className="eyebrow">Worth a closer look</p>
          <h2 id="movie-dialog-title">{clean.title}</h2>
          <div className="dialog-meta">
            {year && <span>{year}</span>}
            {movie.vote_average !== undefined && movie.vote_average > 0 && (
              <span>
                <Star size={14} fill="currentColor" /> {movie.vote_average.toFixed(1)}
              </span>
            )}
            {movie.genres?.slice(0, 3).map((genre) => <span key={genre}>{genre}</span>)}
          </div>
          <p className="dialog-overview">
            {movie.overview || "There is no summary for this one yet, but the model still found a strong match."}
          </p>
          {movie.reasons && movie.reasons.length > 0 && (
            <div className="dialog-reasons">
              <span>Why it made the cut</span>
              <ul>
                {movie.reasons.map((reason) => <li key={reason}>{reasonLabel(reason)}</li>)}
              </ul>
            </div>
          )}
          <div className="dialog-actions">
            <button type="button" className="primary-button" onClick={() => onMoreLikeThis(movie)}>
              <Sparkles size={17} /> More like this
            </button>
            <button type="button" className="secondary-button" onClick={() => onSave(movie)}>
              {saved ? <BookmarkCheck size={17} /> : <Bookmark size={17} />}
              {saved ? "Saved" : "Save movie"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
