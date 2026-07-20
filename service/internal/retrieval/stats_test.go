package retrieval

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func writeStatsFixture(t *testing.T, ids []int32, counts []uint32) string {
	t.Helper()
	buf := make([]byte, 0, 8+len(ids)*8)
	buf = append(buf, []byte(statsMagic)...)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(ids)))
	for _, id := range ids {
		buf = binary.LittleEndian.AppendUint32(buf, uint32(id))
	}
	for _, c := range counts {
		buf = binary.LittleEndian.AppendUint32(buf, c)
	}
	path := filepath.Join(t.TempDir(), "stats.bin")
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadItemStatsRoundTrip(t *testing.T) {
	path := writeStatsFixture(t, []int32{10, 20, 30}, []uint32{0, 5, 90000})
	ids, counts, err := loadItemStats(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 3 || ids[2] != 30 || counts[2] != 90000 {
		t.Fatalf("stats round-trip wrong: ids=%v counts=%v", ids, counts)
	}
}

func TestLoadItemStatsRejectsCorrupt(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.bin")
	if err := os.WriteFile(path, []byte("STA1\x03\x00\x00\x00"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := loadItemStats(path); err == nil {
		t.Fatal("truncated stats file should fail")
	}
}

// A cold, wildly popular item should overtake a warm, mildly similar item only
// once the popularity blend is turned up; at weight 0 the pure dot order holds.
func TestTopKBlendedPromotesPopularOnColdQuery(t *testing.T) {
	// id 1 points at the query but is obscure; id 2 is off-axis but a megahit;
	// id 3 is off-axis and obscure.
	idx, err := Load(writeFixture(t,
		[]int32{1, 2, 3},
		[][]float32{{1, 0}, {0, 1}, {0, 1}},
	))
	if err != nil {
		t.Fatal(err)
	}
	idx.AttachPopularity([]int32{1, 2, 3}, []uint32{1, 50000, 1})
	if !idx.HasPopularity() {
		t.Fatal("popularity should be attached")
	}

	query := []float32{1, 0}
	pure := idx.TopKBlended(query, 1, nil, 0)
	if len(pure) != 1 || pure[0].ID != 1 {
		t.Fatalf("weight 0 must equal pure dot product, got %+v", pure)
	}

	blended := idx.TopKBlended(query, 2, nil, 2.0)
	if len(blended) != 2 || blended[0].ID != 2 {
		t.Fatalf("high popularity weight should lift the megahit, got %+v", blended)
	}
}

func TestTopKBlendedWithoutStatsEqualsTopK(t *testing.T) {
	idx, err := Load(writeFixture(t, []int32{1, 2}, [][]float32{{1, 0}, {0, 1}}))
	if err != nil {
		t.Fatal(err)
	}
	// No stats attached: even a large weight must not change the dot order.
	got := idx.TopKBlended([]float32{1, 0}, 2, nil, 5.0)
	if len(got) != 2 || got[0].ID != 1 {
		t.Fatalf("blended without stats should equal TopK, got %+v", got)
	}
}

func TestSupportAndNormalization(t *testing.T) {
	idx, err := Load(writeFixture(t, []int32{1, 2, 3}, [][]float32{{1, 0}, {0, 1}, {1, 1}}))
	if err != nil {
		t.Fatal(err)
	}
	// Stats for an id absent from the index (99) are ignored; index rows with
	// no stat default to zero support.
	idx.AttachPopularity([]int32{1, 2, 99}, []uint32{10, 1000, 7})
	if s, ok := idx.Support(2); !ok || s != 1000 {
		t.Fatalf("Support(2) = %d,%v want 1000,true", s, ok)
	}
	if s, ok := idx.Support(3); !ok || s != 0 {
		t.Fatalf("Support(3) = %d,%v want 0,true (row without a stat)", s, ok)
	}
	// The more-supported item must carry the higher normalized popularity, so a
	// positive blend weight favors it.
	hi := idx.TopKBlended([]float32{0, 0}, 1, map[int]bool{1: true, 3: true}, 1.0)
	if len(hi) != 1 || hi[0].ID != 2 {
		t.Fatalf("normalized popularity ordering wrong: %+v", hi)
	}
}

func TestSupportMissingStats(t *testing.T) {
	idx, err := Load(writeFixture(t, []int32{1}, [][]float32{{1, 0}}))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := idx.Support(1); ok {
		t.Fatal("Support should report false before stats are attached")
	}
	if idx.HasPopularity() {
		t.Fatal("HasPopularity should be false before stats are attached")
	}
}

func TestWarmth(t *testing.T) {
	if w := Warmth(0, 300); w != 0 {
		t.Fatalf("zero support should be fully cold, got %f", w)
	}
	if w := Warmth(5000, 300); w != 1 {
		t.Fatalf("support past the reference should be fully warm, got %f", w)
	}
	mid := Warmth(50, 300)
	if mid <= 0 || mid >= 1 {
		t.Fatalf("partial support should be between 0 and 1, got %f", mid)
	}
	if Warmth(20, 300) >= Warmth(200, 300) {
		t.Fatal("warmth should increase with support")
	}
	if math.Abs(Warmth(300, 300)-1) > 1e-9 {
		t.Fatalf("support equal to the reference should be fully warm, got %f", Warmth(300, 300))
	}
}
