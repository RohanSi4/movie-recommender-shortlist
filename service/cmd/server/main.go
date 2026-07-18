package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"time"

	"movierec/internal/retrieval"
)

type Movie struct {
	MovieID        int
	Title          string
	Genres         string
	RatingMean     float64
	RatingCount    int
	TMDBVoteAvg    float64
	TMDBPopularity float64
	TMDBGenres     string
	TMDBPosterPath string
	TMDBOverview   string
	TMDBRelease    string
}

type UserFeatures struct {
	UserID      int
	RatingMean  float64
	RatingCount int
}

type App struct {
	Movies         []Movie
	MoviesByID     map[int]Movie
	UsersByID      map[int]UserFeatures
	RetrievalSize  int
	Retrieval      *retrieval.Bundle
	ColdStart      []RankResult
	DataDir        string
	PosterBase     string
	ModelAPIBase   string
	AllowedOrigins map[string]bool
	ScoreWeights   ScoreWeights
}

type ScoreWeights struct {
	VoteAvg  float64
	Pop      float64
	CountLog float64
	UserBias float64
	MeanBias float64
}

type RankRequest struct {
	UserID  *int `json:"user_id,omitempty"`
	MovieID *int `json:"movie_id,omitempty"`
	K       int  `json:"k"`
}

type RankResult struct {
	MovieID   int      `json:"movie_id"`
	Score     float64  `json:"score"`
	Title     string   `json:"title"`
	PosterURL string   `json:"poster_url"`
	Reasons   []string `json:"reasons,omitempty"`
}

type RankResponse struct {
	UserID    int          `json:"user_id"`
	MovieID   int          `json:"movie_id,omitempty"`
	Strategy  string       `json:"strategy"`
	Results   []RankResult `json:"results"`
	LatencyMS int64        `json:"latency_ms"`
}

type ScoreRequest struct {
	UserID   int   `json:"user_id"`
	MovieIDs []int `json:"movie_ids"`
}

type ScoreItem struct {
	MovieID int     `json:"movie_id"`
	Score   float64 `json:"score"`
}

type ScoreResponse struct {
	Scores []ScoreItem `json:"scores"`
}

type SearchResult struct {
	MovieID int    `json:"movie_id"`
	Title   string `json:"title"`
}

func main() {
	dataDir := getEnv("MOVIE_DATA_DIR", "")
	if dataDir == "" {
		dataDir = resolveDataDir([]string{
			"service/data",
			"data",
		}, "service/data")
	}
	modelAPI := getEnv("MODEL_API_BASE", "")
	retrievalSize := getEnvInt("CANDIDATE_POOL_SIZE", 2000)
	app := &App{
		DataDir:        dataDir,
		PosterBase:     "https://image.tmdb.org/t/p/w342",
		ModelAPIBase:   modelAPI,
		RetrievalSize:  retrievalSize,
		AllowedOrigins: parseAllowedOrigins(getEnv("CORS_ALLOWED_ORIGINS", "")),
		ScoreWeights: ScoreWeights{
			VoteAvg:  0.15,
			Pop:      0.02,
			CountLog: 0.5,
			UserBias: 1.0,
			MeanBias: 1.0,
		},
	}

	log.Printf("Using data dir: %s", app.DataDir)
	if app.ModelAPIBase != "" {
		log.Printf("Model API: %s", app.ModelAPIBase)
	}
	log.Printf("CORS allowed origins: %s", strings.Join(originList(app.AllowedOrigins), ", "))
	log.Printf("Retrieval candidate size: %d", app.RetrievalSize)
	memoryLimitMB := getEnvInt("MEMORY_LIMIT_MB", 384)
	if memoryLimitMB > 0 {
		debug.SetMemoryLimit(int64(memoryLimitMB) * 1024 * 1024)
		log.Printf("Go memory limit: %d MB", memoryLimitMB)
	}
	if err := app.LoadData(); err != nil {
		log.Printf("Data load warning: %v", err)
	}
	// CSV parsing and binary widening create large short-lived buffers. Return
	// those pages before accepting traffic so small service instances keep a
	// safe steady-state memory margin.
	debug.FreeOSMemory()

	mux := http.NewServeMux()
	mux.HandleFunc("/health", app.handleHealth)
	mux.HandleFunc("/rank", app.handleRank)
	mux.HandleFunc("/search", app.handleSearch)
	mux.HandleFunc("/movie/", app.handleMovie)

	addr := getEnv("PORT", "8080")
	if !strings.HasPrefix(addr, ":") {
		addr = ":" + addr
	}

	log.Printf("Starting server on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal(err)
	}
}

func (a *App) LoadData() error {
	moviesPath := filepath.Join(a.DataDir, "movie_features.csv")
	usersPath := filepath.Join(a.DataDir, "user_features.csv")

	movies, err := loadMoviesCSV(moviesPath)
	if err != nil {
		return err
	}
	users, err := loadUsersCSV(usersPath)
	if err != nil {
		return err
	}

	if len(movies) == 0 || len(users) == 0 {
		return fmt.Errorf("loaded empty data (movies=%d users=%d)", len(movies), len(users))
	}

	a.Movies = movies
	a.MoviesByID = make(map[int]Movie, len(movies))
	for _, m := range movies {
		a.MoviesByID[m.MovieID] = m
	}

	a.UsersByID = make(map[int]UserFeatures, len(users))
	for _, u := range users {
		a.UsersByID[u.UserID] = u
	}
	a.ColdStart = a.rankMovies(nil, 100)

	log.Printf("Loaded %d movies, %d users", len(movies), len(users))
	bundle, err := retrieval.LoadBundle(a.DataDir)
	if err != nil {
		log.Printf("Retrieval unavailable; using heuristic fallbacks: %v", err)
	} else {
		a.Retrieval = bundle
		log.Printf(
			"Loaded retrieval run %s: %d items, %d users, %d histories, %d dimensions",
			bundle.ModelRun,
			bundle.Items.Len(),
			bundle.Users.Len(),
			bundle.History.Len(),
			bundle.Items.Dim(),
		)
	}
	return nil
}

func (a *App) handleHealth(w http.ResponseWriter, r *http.Request) {
	a.setCORS(w, r)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	retrievalReady := a.Retrieval != nil
	writeJSON(w, http.StatusOK, map[string]any{
		"status":          "ok",
		"retrieval_ready": retrievalReady,
		"model_run": func() string {
			if retrievalReady {
				return a.Retrieval.ModelRun
			}
			return ""
		}(),
	})
}

func (a *App) handleRank(w http.ResponseWriter, r *http.Request) {
	a.setCORS(w, r)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	start := time.Now()
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "POST required"})
		return
	}
	if len(a.Movies) == 0 {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "no data loaded"})
		return
	}

	var req RankRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid json"})
		return
	}
	req.K = boundedK(req.K)
	if req.UserID == nil && req.MovieID == nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "user_id or movie_id required"})
		return
	}

	var results []RankResult
	var response RankResponse

	if req.MovieID != nil && *req.MovieID > 0 {
		seed, ok := a.MoviesByID[*req.MovieID]
		if !ok {
			writeJSON(w, http.StatusNotFound, map[string]string{"error": "movie not found"})
			return
		}
		results, response.Strategy = a.rankMoviesByMovie(seed, req.K)
		response.MovieID = *req.MovieID
	} else if req.UserID != nil && *req.UserID > 0 {
		results, response.Strategy = a.rankMoviesForUser(*req.UserID, req.K)
		response.UserID = *req.UserID
	} else {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid user_id or movie_id"})
		return
	}
	response.LatencyMS = time.Since(start).Milliseconds()
	response.Results = results
	writeJSON(w, http.StatusOK, response)
}

func boundedK(k int) int {
	if k <= 0 {
		return 25
	}
	if k > 100 {
		return 100
	}
	return k
}

func (a *App) handleSearch(w http.ResponseWriter, r *http.Request) {
	a.setCORS(w, r)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "GET required"})
		return
	}
	query := strings.TrimSpace(r.URL.Query().Get("q"))
	if query == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "q required"})
		return
	}
	limit := 10
	if l := r.URL.Query().Get("limit"); l != "" {
		if parsed, err := strconv.Atoi(l); err == nil && parsed > 0 {
			limit = parsed
		}
	}
	results := a.searchMovies(query, limit)
	writeJSON(w, http.StatusOK, results)
}

func (a *App) handleMovie(w http.ResponseWriter, r *http.Request) {
	a.setCORS(w, r)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/movie/"), "/")
	if len(parts) == 0 || parts[0] == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "movie id required"})
		return
	}
	id, err := strconv.Atoi(parts[0])
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid movie id"})
		return
	}

	movie, ok := a.MoviesByID[id]
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
		return
	}
	writeJSON(w, http.StatusOK, movie)
}

func (a *App) rankMovies(user *UserFeatures, k int) []RankResult {
	results := make([]RankResult, 0, k)

	type scored struct {
		Movie Movie
		Score float64
	}
	scoredMovies := make([]scored, 0, len(a.Movies))
	for _, m := range a.Movies {
		scoredMovies = append(scoredMovies, scored{Movie: m, Score: a.scoreMovie(m, user)})
	}

	sort.Slice(scoredMovies, func(i, j int) bool {
		return scoredMovies[i].Score > scoredMovies[j].Score
	})

	if k > len(scoredMovies) {
		k = len(scoredMovies)
	}
	for i := 0; i < k; i++ {
		m := scoredMovies[i].Movie
		results = append(results, RankResult{
			MovieID:   m.MovieID,
			Score:     scoredMovies[i].Score,
			Title:     m.Title,
			PosterURL: joinPosterURL(a.PosterBase, m.TMDBPosterPath),
			Reasons:   buildReasons(m, user),
		})
	}

	return results
}

func (a *App) rankMoviesForUser(userID int, k int) ([]RankResult, string) {
	if a.Retrieval != nil {
		query := a.Retrieval.Users.Vector(userID)
		if query != nil {
			retrieveK := k + 64
			if a.ModelAPIBase != "" && a.RetrievalSize > retrieveK {
				retrieveK = a.RetrievalSize
			}
			exclude := a.Retrieval.History.ExcludeSet(userID)
			scores := a.Retrieval.Items.TopK(query, retrieveK, exclude)
			if len(scores) > 0 {
				if a.ModelAPIBase != "" {
					candidates := make([]int, 0, len(scores))
					for _, score := range scores {
						candidates = append(candidates, score.ID)
					}
					ranked, err := a.rankMoviesWithModel(userID, k, candidates)
					if err == nil && len(ranked) > 0 {
						return ranked, "two_tower_then_lightgbm"
					}
					if err != nil {
						log.Printf("Model API error; returning retrieval ranking: %v", err)
					} else {
						log.Printf("Model API returned no usable results; returning retrieval ranking")
					}
				}
				return a.embeddingResults(scores, k, func(movie Movie) []string {
					return buildUserEmbeddingReasons(movie)
				}), "two_tower"
			}
		}
	}

	return a.rankColdStart(k), "popularity_fallback"
}

func (a *App) rankColdStart(k int) []RankResult {
	if len(a.ColdStart) == 0 {
		a.ColdStart = a.rankMovies(nil, 100)
	}
	if k > len(a.ColdStart) {
		k = len(a.ColdStart)
	}
	return a.ColdStart[:k]
}

func (a *App) rankMoviesByMovie(seed Movie, k int) ([]RankResult, string) {
	if a.Retrieval != nil {
		query := a.Retrieval.Items.Vector(seed.MovieID)
		if query != nil {
			exclude := map[int]bool{seed.MovieID: true}
			scores := a.Retrieval.Items.TopK(query, k+64, exclude)
			results := a.embeddingResults(scores, k, func(movie Movie) []string {
				return buildMovieEmbeddingReasons(seed, movie)
			})
			if len(results) > 0 {
				return results, "two_tower_movie_similarity"
			}
		}
	}
	return a.rankMoviesByMovieHeuristic(seed, k), "movie_heuristic_fallback"
}

func (a *App) embeddingResults(
	scores []retrieval.Scored,
	k int,
	reasonBuilder func(Movie) []string,
) []RankResult {
	results := make([]RankResult, 0, k)
	for _, scored := range scores {
		movie, ok := a.MoviesByID[scored.ID]
		if !ok {
			continue
		}
		results = append(results, RankResult{
			MovieID:   movie.MovieID,
			Score:     scored.Score,
			Title:     movie.Title,
			PosterURL: joinPosterURL(a.PosterBase, movie.TMDBPosterPath),
			Reasons:   reasonBuilder(movie),
		})
		if len(results) == k {
			break
		}
	}
	return results
}

func (a *App) rankMoviesByMovieHeuristic(seed Movie, k int) []RankResult {
	results := make([]RankResult, 0, k)

	type scored struct {
		Movie Movie
		Score float64
	}
	scoredMovies := make([]scored, 0, len(a.Movies))
	for _, m := range a.Movies {
		if m.MovieID == seed.MovieID {
			continue
		}
		scoredMovies = append(scoredMovies, scored{Movie: m, Score: scoreMovieSimilarity(seed, m)})
	}

	sort.Slice(scoredMovies, func(i, j int) bool {
		return scoredMovies[i].Score > scoredMovies[j].Score
	})

	if k > len(scoredMovies) {
		k = len(scoredMovies)
	}
	for i := 0; i < k; i++ {
		m := scoredMovies[i].Movie
		results = append(results, RankResult{
			MovieID:   m.MovieID,
			Score:     scoredMovies[i].Score,
			Title:     m.Title,
			PosterURL: joinPosterURL(a.PosterBase, m.TMDBPosterPath),
			Reasons:   buildMovieReasons(seed, m),
		})
	}

	return results
}

func (a *App) rankMoviesWithModel(userID int, k int, candidates []int) ([]RankResult, error) {
	if len(candidates) == 0 {
		return nil, fmt.Errorf("no candidates available")
	}
	if k > len(candidates) {
		k = len(candidates)
	}

	payload := ScoreRequest{
		UserID:   userID,
		MovieIDs: candidates,
	}
	buf, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Post(a.ModelAPIBase+"/score", "application/json", strings.NewReader(string(buf)))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("model api status %d", resp.StatusCode)
	}

	var scoreResp ScoreResponse
	if err := json.NewDecoder(resp.Body).Decode(&scoreResp); err != nil {
		return nil, err
	}

	scored := make([]RankResult, 0, len(scoreResp.Scores))
	for _, item := range scoreResp.Scores {
		movie, ok := a.MoviesByID[item.MovieID]
		if !ok {
			continue
		}
		user, ok := a.UsersByID[userID]
		var userPtr *UserFeatures
		if ok {
			userPtr = &user
		}
		scored = append(scored, RankResult{
			MovieID:   movie.MovieID,
			Score:     item.Score,
			Title:     movie.Title,
			PosterURL: joinPosterURL(a.PosterBase, movie.TMDBPosterPath),
			Reasons:   buildReasons(movie, userPtr),
		})
	}

	sort.Slice(scored, func(i, j int) bool {
		return scored[i].Score > scored[j].Score
	})
	if k > len(scored) {
		k = len(scored)
	}
	return scored[:k], nil
}

func (a *App) scoreMovie(m Movie, user *UserFeatures) float64 {
	score := 0.0
	score += m.RatingMean
	score += a.ScoreWeights.VoteAvg * m.TMDBVoteAvg
	score += a.ScoreWeights.Pop * m.TMDBPopularity
	score += a.ScoreWeights.CountLog * math.Log1p(float64(m.RatingCount))

	if user != nil {
		score += a.ScoreWeights.UserBias * user.RatingMean
		score -= a.ScoreWeights.MeanBias * math.Abs(m.RatingMean-user.RatingMean)
	}
	return score
}

func buildReasons(m Movie, user *UserFeatures) []string {
	reasons := []string{}
	if m.TMDBVoteAvg >= 7.5 {
		reasons = append(reasons, "high_vote_avg")
	}
	if m.RatingCount >= 1000 {
		reasons = append(reasons, "popular_in_movielens")
	}
	if user != nil && math.Abs(m.RatingMean-user.RatingMean) <= 0.5 {
		reasons = append(reasons, "matches_user_taste")
	}
	return reasons
}

func buildUserEmbeddingReasons(movie Movie) []string {
	reasons := []string{"learned_from_user_history"}
	if movie.TMDBVoteAvg >= 7.5 {
		reasons = append(reasons, "high_vote_avg")
	}
	if movie.RatingCount >= 1000 {
		reasons = append(reasons, "popular_in_movielens")
	}
	return reasons
}

func buildMovieEmbeddingReasons(seed Movie, candidate Movie) []string {
	reasons := []string{"learned_movie_similarity"}
	if genreSimilarity(seed, candidate) >= 0.4 {
		reasons = append(reasons, "similar_genres")
	}
	if candidate.RatingCount >= 1000 {
		reasons = append(reasons, "popular_in_movielens")
	}
	return reasons
}

func buildMovieReasons(seed Movie, candidate Movie) []string {
	reasons := []string{}
	if genreSimilarity(seed, candidate) >= 0.4 {
		reasons = append(reasons, "similar_genres")
	}
	if math.Abs(seed.RatingMean-candidate.RatingMean) <= 0.3 {
		reasons = append(reasons, "similar_ratings")
	}
	if candidate.RatingCount >= 1000 {
		reasons = append(reasons, "popular_in_movielens")
	}
	return reasons
}

func scoreMovieSimilarity(seed Movie, candidate Movie) float64 {
	score := 0.0
	score += 2.0 * genreSimilarity(seed, candidate)
	score += 1.0 - (math.Abs(seed.RatingMean-candidate.RatingMean) / 5.0)
	if seed.TMDBVoteAvg > 0 && candidate.TMDBVoteAvg > 0 {
		score += 0.8 - (math.Abs(seed.TMDBVoteAvg-candidate.TMDBVoteAvg) / 10.0)
	}
	score += 0.15 * math.Log1p(float64(candidate.RatingCount))
	return score
}

func genreSimilarity(a Movie, b Movie) float64 {
	aGenres := parseGenres(preferGenres(a))
	bGenres := parseGenres(preferGenres(b))
	if len(aGenres) == 0 || len(bGenres) == 0 {
		return 0
	}
	intersection := 0
	for g := range aGenres {
		if bGenres[g] {
			intersection++
		}
	}
	union := len(aGenres) + len(bGenres) - intersection
	if union == 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}

func preferGenres(m Movie) string {
	if strings.TrimSpace(m.TMDBGenres) != "" {
		return m.TMDBGenres
	}
	return m.Genres
}

func parseGenres(raw string) map[string]bool {
	cleaned := strings.TrimSpace(raw)
	if cleaned == "" {
		return nil
	}
	parts := strings.Split(cleaned, "|")
	out := make(map[string]bool, len(parts))
	for _, part := range parts {
		p := strings.TrimSpace(part)
		if p == "" {
			continue
		}
		out[strings.ToLower(p)] = true
	}
	return out
}

func (a *App) searchMovies(query string, limit int) []SearchResult {
	q := strings.ToLower(strings.TrimSpace(query))
	if q == "" {
		return nil
	}
	type scored struct {
		Movie Movie
		Score float64
	}
	scoredMovies := make([]scored, 0, len(a.Movies))
	for _, m := range a.Movies {
		title := strings.ToLower(stripYear(m.Title))
		score := 0.0
		if strings.HasPrefix(title, q) {
			score += 3
		}
		if strings.Contains(title, q) {
			score += 1
		}
		if score == 0 {
			continue
		}
		score += 0.1 * math.Log1p(float64(m.RatingCount))
		scoredMovies = append(scoredMovies, scored{Movie: m, Score: score})
	}
	sort.Slice(scoredMovies, func(i, j int) bool {
		return scoredMovies[i].Score > scoredMovies[j].Score
	})
	if limit > len(scoredMovies) {
		limit = len(scoredMovies)
	}
	results := make([]SearchResult, 0, limit)
	for i := 0; i < limit; i++ {
		results = append(results, SearchResult{
			MovieID: scoredMovies[i].Movie.MovieID,
			Title:   scoredMovies[i].Movie.Title,
		})
	}
	return results
}

func stripYear(title string) string {
	trimmed := strings.TrimSpace(title)
	if len(trimmed) >= 7 && strings.HasSuffix(trimmed, ")") {
		idx := strings.LastIndex(trimmed, "(")
		if idx > 0 {
			candidate := strings.TrimSpace(trimmed[:idx])
			return candidate
		}
	}
	return trimmed
}

func loadMoviesCSV(path string) ([]Movie, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1

	header, err := reader.Read()
	if err != nil {
		return nil, err
	}
	idx := headerIndex(header)

	required := []string{"movieId", "title", "rating_mean", "rating_count"}
	for _, col := range required {
		if _, ok := idx[col]; !ok {
			return nil, fmt.Errorf("missing column %s in %s", col, path)
		}
	}

	var movies []Movie
	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			if err == csv.ErrFieldCount {
				continue
			}
			return nil, err
		}

		movieID := parseInt(row, idx, "movieId")
		title := parseString(row, idx, "title")
		ratingMean := parseFloat(row, idx, "rating_mean")
		ratingCount := parseInt(row, idx, "rating_count")

		movies = append(movies, Movie{
			MovieID:        movieID,
			Title:          title,
			Genres:         parseString(row, idx, "genres"),
			RatingMean:     ratingMean,
			RatingCount:    ratingCount,
			TMDBVoteAvg:    parseFloat(row, idx, "tmdb_vote_avg"),
			TMDBPopularity: parseFloat(row, idx, "tmdb_popularity"),
			TMDBGenres:     parseString(row, idx, "tmdb_genres"),
			TMDBPosterPath: parseString(row, idx, "tmdb_poster_path"),
			TMDBOverview:   parseString(row, idx, "tmdb_overview"),
			TMDBRelease:    parseString(row, idx, "tmdb_release_date"),
		})
	}

	return movies, nil
}

func loadUsersCSV(path string) ([]UserFeatures, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1

	header, err := reader.Read()
	if err != nil {
		return nil, err
	}
	idx := headerIndex(header)

	required := []string{"userId", "rating_mean", "rating_count"}
	for _, col := range required {
		if _, ok := idx[col]; !ok {
			return nil, fmt.Errorf("missing column %s in %s", col, path)
		}
	}

	var users []UserFeatures
	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			if err == csv.ErrFieldCount {
				continue
			}
			return nil, err
		}

		users = append(users, UserFeatures{
			UserID:      parseInt(row, idx, "userId"),
			RatingMean:  parseFloat(row, idx, "rating_mean"),
			RatingCount: parseInt(row, idx, "rating_count"),
		})
	}

	return users, nil
}

func headerIndex(header []string) map[string]int {
	idx := make(map[string]int, len(header))
	for i, col := range header {
		idx[strings.TrimSpace(col)] = i
	}
	return idx
}

func parseString(row []string, idx map[string]int, col string) string {
	i, ok := idx[col]
	if !ok || i >= len(row) {
		return ""
	}
	return row[i]
}

func parseInt(row []string, idx map[string]int, col string) int {
	i, ok := idx[col]
	if !ok || i >= len(row) {
		return 0
	}
	val := strings.TrimSpace(row[i])
	if val == "" {
		return 0
	}
	parsed, err := strconv.Atoi(val)
	if err != nil {
		return 0
	}
	return parsed
}

func parseFloat(row []string, idx map[string]int, col string) float64 {
	i, ok := idx[col]
	if !ok || i >= len(row) {
		return 0
	}
	val := strings.TrimSpace(row[i])
	if val == "" {
		return 0
	}
	parsed, err := strconv.ParseFloat(val, 64)
	if err != nil {
		return 0
	}
	return parsed
}

func joinPosterURL(base, path string) string {
	if path == "" {
		return ""
	}
	return base + path
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		log.Printf("json encode error: %v", err)
	}
}

func (a *App) setCORS(w http.ResponseWriter, r *http.Request) {
	origin := r.Header.Get("Origin")
	if origin != "" && (a.AllowedOrigins["*"] || a.AllowedOrigins[origin]) {
		w.Header().Set("Access-Control-Allow-Origin", origin)
		w.Header().Add("Vary", "Origin")
	}
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
}

func parseAllowedOrigins(raw string) map[string]bool {
	defaults := []string{
		"http://localhost:3000",
		"http://localhost:3001",
	}
	allowed := make(map[string]bool, len(defaults))
	for _, origin := range defaults {
		allowed[origin] = true
	}
	for _, part := range strings.Split(raw, ",") {
		origin := strings.TrimSpace(part)
		if origin != "" {
			allowed[origin] = true
		}
	}
	return allowed
}

func originList(origins map[string]bool) []string {
	out := make([]string, 0, len(origins))
	for origin := range origins {
		out = append(out, origin)
	}
	sort.Strings(out)
	return out
}

func resolveDataDir(candidates []string, fallback string) string {
	for _, candidate := range candidates {
		if candidate == "" {
			continue
		}
		info, err := os.Stat(candidate)
		if err == nil && info.IsDir() {
			return candidate
		}
	}
	return fallback
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	raw := os.Getenv(key)
	if raw == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return parsed
}
