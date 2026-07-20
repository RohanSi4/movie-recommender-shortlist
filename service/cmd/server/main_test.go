package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
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

func TestColdStartCanBackfillAfterMaximumExclusions(t *testing.T) {
	app := &App{ColdStart: make([]RankResult, 600)}
	exclusions := make([]int, 500)
	for index := range app.ColdStart {
		app.ColdStart[index] = RankResult{MovieID: index + 1}
		if index < len(exclusions) {
			exclusions[index] = index + 1
		}
	}
	results := app.rankColdStartExcluding(100, exclusions)
	if len(results) != 100 || results[0].MovieID != 501 || results[99].MovieID != 600 {
		t.Fatalf("fallback did not refill after exclusions: %+v", results)
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

func TestTasteMixUsesMultipleMoviesAndExcludesSeeds(t *testing.T) {
	app := retrievalTestApp(t)
	results, strategy, seedIDs, err := app.rankMoviesByTaste([]int{1, 3, 1}, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if strategy != "two_tower_taste_mix" {
		t.Fatalf("strategy = %q, want two_tower_taste_mix", strategy)
	}
	if len(seedIDs) != 2 || seedIDs[0] != 1 || seedIDs[1] != 3 {
		t.Fatalf("seed ids = %v, want [1 3]", seedIDs)
	}
	if len(results) != 2 {
		t.Fatalf("results = %v, want 2", results)
	}
	for _, result := range results {
		if result.MovieID == 1 || result.MovieID == 3 {
			t.Fatalf("seed movie leaked into taste results: %+v", results)
		}
	}
}

func TestColdPopWeightGatesOnMeanSeedWarmth(t *testing.T) {
	app := retrievalTestApp(t)
	app.ColdPopWeight = 0.6
	app.WarmRef = 300
	// Item index ids are {1, 999, 2, 3, 4}. Seed 1 is well supported (warm);
	// seed 3 has no training support (cold).
	app.Retrieval.Items.AttachPopularity(
		[]int32{1, 999, 2, 3, 4},
		[]uint32{5000, 10, 800, 0, 400},
	)

	if w := app.coldPopWeight([]int{1}); w > 1e-9 {
		t.Fatalf("warm seed should yield ~0 weight, got %f", w)
	}
	if w := app.coldPopWeight([]int{3}); math.Abs(w-0.6) > 1e-9 {
		t.Fatalf("cold seed should yield the full ColdPopWeight, got %f", w)
	}
	// Mean warmth of one warm (1.0) and one cold (0.0) seed is 0.5, so the
	// weight is 0.6 * (1 - 0.5) = 0.3.
	if w := app.coldPopWeight([]int{1, 3}); math.Abs(w-0.3) > 1e-9 {
		t.Fatalf("mixed warm+cold weight should be 0.3, got %f", w)
	}
}

func TestColdPopWeightZeroWithoutStats(t *testing.T) {
	app := retrievalTestApp(t)
	app.ColdPopWeight = 0.6
	app.WarmRef = 300
	// No stats attached: the blend must stay off so behavior is unchanged.
	if w := app.coldPopWeight([]int{3}); w != 0 {
		t.Fatalf("missing stats should disable the blend, got %f", w)
	}
}

// With popularity attached and a cold seed, the ranking must lean on the
// popularity prior: the highest-support candidate should surface even though a
// different item is the nearest neighbor of the (noisy) seed vector.
func TestColdSeedRankingLeansOnPopularity(t *testing.T) {
	app := retrievalTestApp(t)
	app.ColdPopWeight = 5.0 // exaggerate so the test is unambiguous
	app.WarmRef = 300
	app.Retrieval.Items.AttachPopularity(
		[]int32{1, 999, 2, 3, 4},
		[]uint32{0, 0, 50000, 0, 0}, // movie 2 is the runaway hit
	)
	// Seed on movie 3 (cold, support 0). Its nearest neighbor by pure cosine is
	// not movie 2, but the popularity blend should still float movie 2 to the top.
	results, _ := app.rankMoviesByMovieExcluding(app.MoviesByID[3], 1, nil)
	if len(results) != 1 || results[0].MovieID != 2 {
		t.Fatalf("cold seed should surface the popular title, got %+v", results)
	}
}

func TestTasteMixHonorsExplicitExclusions(t *testing.T) {
	app := retrievalTestApp(t)
	results, _, _, err := app.rankMoviesByTaste([]int{1, 3}, 2, []int{2})
	if err != nil {
		t.Fatal(err)
	}
	for _, result := range results {
		if result.MovieID == 2 {
			t.Fatalf("explicitly excluded movie leaked into results: %+v", results)
		}
	}
}

func TestRankHandlerReturnsRichTasteResults(t *testing.T) {
	app := retrievalTestApp(t)
	movie := app.MoviesByID[2]
	movie.TMDBRelease = "2024-03-01"
	movie.TMDBGenres = "Drama|Mystery"
	movie.TMDBVoteAvg = 8.1
	movie.TMDBOverview = "A useful overview."
	app.MoviesByID[2] = movie
	for i := range app.Movies {
		if app.Movies[i].MovieID == movie.MovieID {
			app.Movies[i] = movie
		}
	}

	request := httptest.NewRequest(http.MethodPost, "/rank", strings.NewReader(`{"movie_ids":[1,3],"k":2}`))
	recorder := httptest.NewRecorder()
	app.handleRank(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", recorder.Code, recorder.Body.String())
	}
	var response RankResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response.Strategy != "two_tower_taste_mix" || len(response.MovieIDs) != 2 {
		t.Fatalf("unexpected taste response: %+v", response)
	}
	if len(response.Results) == 0 {
		t.Fatal("expected taste results")
	}
	for _, result := range response.Results {
		if result.MovieID == 2 {
			if result.ReleaseDate != "2024-03-01" || result.VoteAverage != 8.1 || result.Overview == "" || len(result.Genres) != 2 {
				t.Fatalf("movie metadata missing from result: %+v", result)
			}
			return
		}
	}
	t.Fatalf("enriched movie missing from response: %+v", response.Results)
}

func TestRankHandlerLimitsTasteInputs(t *testing.T) {
	app := retrievalTestApp(t)
	request := httptest.NewRequest(http.MethodPost, "/rank", strings.NewReader(`{"movie_ids":[1,2,3,4,5,6],"k":2}`))
	recorder := httptest.NewRecorder()
	app.handleRank(recorder, request)
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body=%s", recorder.Code, recorder.Body.String())
	}
}

func TestParseIntAcceptsIntegerShapedCSVFloats(t *testing.T) {
	row := []string{"68997.0"}
	index := map[string]int{"rating_count": 0}
	if got := parseInt(row, index, "rating_count"); got != 68997 {
		t.Fatalf("parseInt() = %d, want 68997", got)
	}
	row[0] = "3.5"
	if got := parseInt(row, index, "rating_count"); got != 0 {
		t.Fatalf("parseInt() accepted fractional count: %d", got)
	}
}

func TestSearchHandlerCapsLimit(t *testing.T) {
	app := &App{Movies: make([]Movie, 0, 75), PosterBase: "https://example.com"}
	for id := 1; id <= 75; id++ {
		app.Movies = append(app.Movies, Movie{MovieID: id, Title: fmt.Sprintf("Test Movie %d", id)})
	}
	request := httptest.NewRequest(http.MethodGet, "/search?q=test&limit=1000", nil)
	recorder := httptest.NewRecorder()
	app.handleSearch(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", recorder.Code, recorder.Body.String())
	}
	var results []SearchResult
	if err := json.Unmarshal(recorder.Body.Bytes(), &results); err != nil {
		t.Fatal(err)
	}
	if len(results) != 50 {
		t.Fatalf("returned %d search results, want 50", len(results))
	}
}

func TestMovieHandlerUsesPublicJSONFields(t *testing.T) {
	app := &App{MoviesByID: map[int]Movie{1: {MovieID: 1, Title: "Toy Story", RatingCount: 68997}}}
	request := httptest.NewRequest(http.MethodGet, "/movie/1", nil)
	recorder := httptest.NewRecorder()
	app.handleMovie(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", recorder.Code, recorder.Body.String())
	}
	body := recorder.Body.String()
	if !strings.Contains(body, `"movie_id":1`) || !strings.Contains(body, `"rating_count":68997`) {
		t.Fatalf("missing public JSON fields: %s", body)
	}
	if strings.Contains(body, "MovieID") || strings.Contains(body, "RatingCount") {
		t.Fatalf("leaked Go field names: %s", body)
	}
}
