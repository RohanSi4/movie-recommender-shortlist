package retrieval

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func writeHistoryFixture(t *testing.T, users map[int][]int) string {
	t.Helper()
	userIDs := []int{10, 20}
	payload := make([]byte, 0)
	offsets := []uint64{0}
	for _, userID := range userIDs {
		previous := 0
		for _, movieID := range users[userID] {
			payload = binary.AppendUvarint(payload, uint64(movieID-previous))
			previous = movieID
		}
		offsets = append(offsets, uint64(len(payload)))
	}

	buf := append([]byte(historyMagic), make([]byte, 4)...)
	binary.LittleEndian.PutUint32(buf[4:8], uint32(len(userIDs)))
	for _, userID := range userIDs {
		buf = binary.LittleEndian.AppendUint32(buf, uint32(userID))
	}
	for _, offset := range offsets {
		buf = binary.LittleEndian.AppendUint64(buf, offset)
	}
	buf = append(buf, payload...)
	path := filepath.Join(t.TempDir(), "history.bin")
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestHistoryExcludeSet(t *testing.T) {
	path := writeHistoryFixture(t, map[int][]int{
		10: {2, 9, 100},
		20: {5},
	})
	history, err := LoadHistory(path)
	if err != nil {
		t.Fatal(err)
	}
	if history.Len() != 2 {
		t.Fatalf("Len = %d, want 2", history.Len())
	}
	got := history.ExcludeSet(10)
	for _, movieID := range []int{2, 9, 100} {
		if !got[movieID] {
			t.Fatalf("movie %d missing from exclude set: %#v", movieID, got)
		}
	}
	if history.ExcludeSet(999) != nil {
		t.Fatal("unknown user should have no history")
	}
}

func TestHistoryRejectsCorruptOffsets(t *testing.T) {
	path := writeHistoryFixture(t, map[int][]int{10: {2}, 20: {5}})
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// First offset begins after magic, count, and the two user ids.
	binary.LittleEndian.PutUint64(raw[16:24], 1)
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadHistory(path); err == nil {
		t.Fatal("corrupt offsets should fail to load")
	}
}
