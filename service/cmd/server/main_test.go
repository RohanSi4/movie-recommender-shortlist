package main

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"movierec/internal/retrieval"
)

func loadTestIndex(t *testing.T, ids []int32, vectors [][]float32) *retrieval.Index {
	t.Helper()
	dim := len(vectors[0])
	buf := append([]byte("EMB1"), make([]byte, 12)...)
	binary.LittleEndian.PutUint32(buf[4:8], uint32(len(ids)))
	binary.LittleEndian.PutUint32(buf[8:12], uint32(dim))
	binary.LittleEndian.PutUint32(buf[12:16], 4)
	for _, id := range ids {
		buf = binary.LittleEndian.AppendUint32(buf, uint32(id))
	}
	for _, vector := range vectors {
		for _, value := range vector {
			buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(value))
		}
	}
	path := filepath.Join(t.TempDir(), "embeddings.bin")
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	index, err := retrieval.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	return index
}

func loadTestHistory(t *testing.T, userID int, movieIDs []int) *retrieval.History {
	t.Helper()
	payload := make([]byte, 0)
	previous := 0
	for _, movieID := range movieIDs {
		payload = binary.AppendUvarint(payload, uint64(movieID-previous))
		previous = movieID
	}
	buf := append([]byte("HST1"), make([]byte, 4)...)
	binary.LittleEndian.PutUint32(buf[4:8], 1)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(userID))
	buf = binary.LittleEndian.AppendUint64(buf, 0)
	buf = binary.LittleEndian.AppendUint64(buf, uint64(len(payload)))
	buf = append(buf, payload...)
	path := filepath.Join(t.TempDir(), "history.bin")
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	history, err := retrieval.LoadHistory(path)
	if err != nil {
		t.Fatal(err)
	}
	return history
}

func retrievalTestApp(t *testing.T) *App {
	t.Helper()
	movies := []Movie{
		{MovieID: 1, Title: "Seen", RatingMean: 5, RatingCount: 5000},
		{MovieID: 2, Title: "Closest", RatingMean: 4.5, RatingCount: 1000},
		{MovieID: 3, Title: "Different", RatingMean: 3, RatingCount: 100},
		{MovieID: 4, Title: "Second", RatingMean: 4, RatingCount: 800},
	}
	moviesByID := make(map[int]Movie, len(movies))
	for _, movie := range movies {
		moviesByID[movie.MovieID] = movie
	}
	items := loadTestIndex(t,
		[]int32{1, 999, 2, 3, 4},
		[][]float32{{1, 0}, {0.95, 0.05}, {0.9, 0.1}, {0, 1}, {0.8, 0.2}},
	)
	users := loadTestIndex(t, []int32{10}, [][]float32{{1, 0}})
	return &App{
		Movies:     movies,
		MoviesByID: moviesByID,
		UsersByID:  map[int]UserFeatures{10: {UserID: 10, RatingMean: 4.2, RatingCount: 20}},
		PosterBase: "https://example.com",
		Retrieval: &retrieval.Bundle{
			Items:   items,
			Users:   users,
			History: loadTestHistory(t, 10, []int{1}),
		},
	}
}

func TestKnownUserUsesRetrievalExcludesSeenAndBackfillsMetadata(t *testing.T) {
	app := retrievalTestApp(t)
	results, strategy := app.rankMoviesForUser(10, 2)
	if strategy != "two_tower" {
		t.Fatalf("strategy = %q, want two_tower", strategy)
	}
	if len(results) != 2 || results[0].MovieID != 2 || results[1].MovieID != 4 {
		t.Fatalf("unexpected retrieval results: %+v", results)
	}
	for _, result := range results {
		if result.MovieID == 1 || result.MovieID == 999 {
			t.Fatalf("seen or unknown movie leaked into results: %+v", results)
		}
	}
}

func TestColdStartUsesPopularityFallback(t *testing.T) {
	app := retrievalTestApp(t)
	results, strategy := app.rankMoviesForUser(999, 2)
	if strategy != "popularity_fallback" || len(results) != 2 {
		t.Fatalf("cold start did not fall back: strategy=%q results=%+v", strategy, results)
	}
}

func TestMovieSimilarityUsesEmbeddingAndExcludesSeed(t *testing.T) {
	app := retrievalTestApp(t)
	results, strategy := app.rankMoviesByMovie(app.MoviesByID[1], 2)
	if strategy != "two_tower_movie_similarity" {
		t.Fatalf("strategy = %q", strategy)
	}
	if len(results) != 2 || results[0].MovieID != 2 || results[1].MovieID != 4 {
		t.Fatalf("unexpected similar movies: %+v", results)
	}
}

func TestRankHandlerCapsK(t *testing.T) {
	if got := boundedK(1000); got != 100 {
		t.Fatalf("boundedK(1000) = %d, want 100", got)
	}
	app := retrievalTestApp(t)
	request := httptest.NewRequest(http.MethodPost, "/rank", strings.NewReader(`{"user_id":999,"k":1000}`))
	recorder := httptest.NewRecorder()
	app.handleRank(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", recorder.Code, recorder.Body.String())
	}
	var response RankResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if len(response.Results) > 100 {
		t.Fatalf("returned %d results, want at most 100", len(response.Results))
	}
}
