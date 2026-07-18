"use client";

import {
  Bookmark,
  BookmarkCheck,
  CircleMinus,
  Info,
  Sparkles,
  Star,
} from "lucide-react";

import { PosterImage } from "@/components/poster-image";
import { movieYear, reasonLabel, splitMovieTitle } from "@/lib/format";
import type { MovieRecommendation } from "@/lib/types";

type MovieCardProps = {
  movie: MovieRecommendation;
  rank: number;
  saved: boolean;
  onSave: (movie: MovieRecommendation) => void;
  onDetails: (movie: MovieRecommendation) => void;
  onMoreLikeThis: (movie: MovieRecommendation) => void;
  onDismiss: (movie: MovieRecommendation) => void;
};

export function MovieCard({
  movie,
  rank,
  saved,
  onSave,
  onDetails,
  onMoreLikeThis,
  onDismiss,
}: MovieCardProps) {
  const clean = splitMovieTitle(movie.title);
  const year = movieYear(movie.title, movie.release_date);

  return (
    <article className="movie-card">
      <button
        type="button"
        className="movie-poster"
        onClick={() => onDetails(movie)}
        aria-label={`View details for ${clean.title}`}
      >
        <PosterImage
          src={movie.poster_url}
          alt={`Poster for ${clean.title}`}
          sizes="(max-width: 640px) 112px, (max-width: 1024px) 33vw, 22vw"
          priority={rank <= 4}
        />
        <span className="movie-rank" aria-label={`Recommendation ${rank}`}>
          {String(rank).padStart(2, "0")}
        </span>
      </button>

      <div className="movie-card-body">
        <div className="movie-heading">
          <div>
            <h3>{clean.title}</h3>
            <p>
              {year && <span>{year}</span>}
              {movie.vote_average !== undefined && movie.vote_average > 0 && (
                <span className="rating">
                  <Star aria-hidden="true" size={13} fill="currentColor" />
                  {movie.vote_average.toFixed(1)}
                </span>
              )}
            </p>
          </div>
          <button
            type="button"
            className="icon-button"
            data-saved={saved}
            onClick={() => onSave(movie)}
            aria-label={saved ? `Remove ${clean.title} from saved movies` : `Save ${clean.title}`}
          >
            {saved ? <BookmarkCheck size={19} /> : <Bookmark size={19} />}
          </button>
        </div>

        {movie.genres && movie.genres.length > 0 && (
          <p className="movie-genres">{movie.genres.slice(0, 3).join("  ·  ")}</p>
        )}

        {movie.reasons && movie.reasons.length > 0 && (
          <div className="reason-list">
            {movie.reasons.slice(0, 2).map((reason) => (
              <span key={reason}>{reasonLabel(reason)}</span>
            ))}
          </div>
        )}

        <div className="movie-actions">
          <button type="button" onClick={() => onMoreLikeThis(movie)}>
            <Sparkles aria-hidden="true" size={15} />
            More like this
          </button>
          <button type="button" onClick={() => onDetails(movie)}>
            <Info aria-hidden="true" size={15} />
            Details
          </button>
          <button
            type="button"
            className="dismiss-button"
            onClick={() => onDismiss(movie)}
            aria-label={`Not interested in ${clean.title}`}
            title="Not for me"
          >
            <CircleMinus aria-hidden="true" size={16} />
          </button>
        </div>
      </div>
    </article>
  );
}
