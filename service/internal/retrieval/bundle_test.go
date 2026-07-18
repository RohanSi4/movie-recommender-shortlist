package retrieval

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func fixtureHash(t *testing.T, path string) string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(raw)
	return hex.EncodeToString(digest[:])
}

func TestLoadBundleVerifiesOneModelRun(t *testing.T) {
	dir := t.TempDir()
	itemsSource := writeFixture(t, []int32{1, 2}, [][]float32{{1, 0}, {0, 1}})
	usersSource := writeFixture(t, []int32{10, 20}, [][]float32{{1, 0}, {0, 1}})
	historySource := writeHistoryFixture(t, map[int][]int{10: {1}, 20: {2}})

	paths := map[string]string{
		"items":   filepath.Join(dir, "item_embeddings.bin"),
		"users":   filepath.Join(dir, "user_embeddings.bin"),
		"history": filepath.Join(dir, "user_history.bin"),
	}
	for key, source := range map[string]string{"items": itemsSource, "users": usersSource, "history": historySource} {
		raw, err := os.ReadFile(source)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(paths[key], raw, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	manifest := bundleManifest{
		FormatVersion: 1,
		ModelRun:      "fixture-run",
		Files: map[string]manifestFile{
			"items":   {Name: "item_embeddings.bin", SHA256: fixtureHash(t, paths["items"]), Count: 2, Dim: 2},
			"users":   {Name: "user_embeddings.bin", SHA256: fixtureHash(t, paths["users"]), Count: 2, Dim: 2},
			"history": {Name: "user_history.bin", SHA256: fixtureHash(t, paths["history"]), Count: 2},
		},
	}
	raw, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, manifestName), raw, 0o644); err != nil {
		t.Fatal(err)
	}

	bundle, err := LoadBundle(dir)
	if err != nil {
		t.Fatal(err)
	}
	if bundle.ModelRun != "fixture-run" || bundle.Items.Len() != 2 || bundle.Users.Len() != 2 {
		t.Fatalf("unexpected bundle: %+v", bundle)
	}
	if !bundle.History.ExcludeSet(10)[1] {
		t.Fatal("history was not loaded")
	}

	if err := os.WriteFile(paths["items"], []byte("changed"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadBundle(dir); err == nil {
		t.Fatal("changed artifact should fail manifest verification")
	}
}
