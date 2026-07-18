package retrieval

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

const manifestName = "retrieval_manifest.json"

type manifestFile struct {
	Name   string `json:"name"`
	SHA256 string `json:"sha256"`
	Count  int    `json:"count"`
	Dim    int    `json:"dim,omitempty"`
}

type bundleManifest struct {
	FormatVersion int                     `json:"format_version"`
	ModelRun      string                  `json:"model_run"`
	Files         map[string]manifestFile `json:"files"`
}

// Bundle is one verified serving export. The manifest hashes prevent item and
// user vectors from different training runs from being mixed silently.
type Bundle struct {
	Items    *Index
	Users    *Index
	History  *History
	ModelRun string
}

// LoadBundle verifies and loads a complete retrieval export.
func LoadBundle(dataDir string) (*Bundle, error) {
	manifestPath := filepath.Join(dataDir, manifestName)
	raw, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, err
	}
	var manifest bundleManifest
	if err := json.Unmarshal(raw, &manifest); err != nil {
		return nil, fmt.Errorf("%s: invalid json: %w", manifestPath, err)
	}
	if manifest.FormatVersion != 1 || manifest.ModelRun == "" {
		return nil, fmt.Errorf("%s: unsupported or incomplete manifest", manifestPath)
	}

	itemsFile, err := requiredManifestFile(manifest, "items", "item_embeddings.bin")
	if err != nil {
		return nil, err
	}
	usersFile, err := requiredManifestFile(manifest, "users", "user_embeddings.bin")
	if err != nil {
		return nil, err
	}
	historyFile, err := requiredManifestFile(manifest, "history", "user_history.bin")
	if err != nil {
		return nil, err
	}
	for _, file := range []manifestFile{itemsFile, usersFile, historyFile} {
		if err := verifySHA256(filepath.Join(dataDir, file.Name), file.SHA256); err != nil {
			return nil, err
		}
	}

	items, err := Load(filepath.Join(dataDir, itemsFile.Name))
	if err != nil {
		return nil, err
	}
	users, err := Load(filepath.Join(dataDir, usersFile.Name))
	if err != nil {
		return nil, err
	}
	history, err := LoadHistory(filepath.Join(dataDir, historyFile.Name))
	if err != nil {
		return nil, err
	}
	if items.Len() != itemsFile.Count || items.Dim() != itemsFile.Dim {
		return nil, fmt.Errorf("item embedding shape does not match manifest")
	}
	if users.Len() != usersFile.Count || users.Dim() != usersFile.Dim {
		return nil, fmt.Errorf("user embedding shape does not match manifest")
	}
	if history.Len() != historyFile.Count {
		return nil, fmt.Errorf("history count does not match manifest")
	}
	if items.Dim() != users.Dim() {
		return nil, fmt.Errorf("item dimension %d does not match user dimension %d", items.Dim(), users.Dim())
	}
	return &Bundle{Items: items, Users: users, History: history, ModelRun: manifest.ModelRun}, nil
}

func requiredManifestFile(manifest bundleManifest, key, expectedName string) (manifestFile, error) {
	file, ok := manifest.Files[key]
	if !ok || file.Name != expectedName || file.Count <= 0 || file.SHA256 == "" {
		return manifestFile{}, fmt.Errorf("manifest has invalid %s entry", key)
	}
	if key != "history" && file.Dim <= 0 {
		return manifestFile{}, fmt.Errorf("manifest has invalid %s dimensions", key)
	}
	return file, nil
}

func verifySHA256(path, expected string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	digest := sha256.New()
	if _, err := io.Copy(digest, file); err != nil {
		return err
	}
	actual := hex.EncodeToString(digest.Sum(nil))
	if actual != expected {
		return fmt.Errorf("%s: sha256 mismatch", path)
	}
	return nil
}
