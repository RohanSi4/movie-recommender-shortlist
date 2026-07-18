package retrieval

import (
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// writeFixture writes a float32 EMB1 file with the given ids and vectors.
func writeFixture(t *testing.T, ids []int32, vecs [][]float32) string {
	t.Helper()
	dim := len(vecs[0])
	buf := make([]byte, 0, 16+len(ids)*4+len(ids)*dim*4)
	buf = append(buf, []byte(magic)...)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(ids)))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(dim))
	buf = binary.LittleEndian.AppendUint32(buf, 4)
	for _, id := range ids {
		buf = binary.LittleEndian.AppendUint32(buf, uint32(id))
	}
	for _, vec := range vecs {
		for _, v := range vec {
			buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(v))
		}
	}
	path := filepath.Join(t.TempDir(), "fixture.bin")
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeHalfFixture(t *testing.T, ids []int32, dim int, values []uint16) string {
	t.Helper()
	buf := make([]byte, 0, 16+len(ids)*4+len(values)*2)
	buf = append(buf, []byte(magic)...)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(ids)))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(dim))
	buf = binary.LittleEndian.AppendUint32(buf, 2)
	for _, id := range ids {
		buf = binary.LittleEndian.AppendUint32(buf, uint32(id))
	}
	for _, value := range values {
		buf = binary.LittleEndian.AppendUint16(buf, value)
	}
	path := filepath.Join(t.TempDir(), "fixture-half.bin")
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadAndTopK(t *testing.T) {
	// Three unit vectors: id 10 along x, id 20 along y, id 30 halfway.
	s := float32(math.Sqrt2 / 2)
	path := writeFixture(t,
		[]int32{10, 20, 30},
		[][]float32{{1, 0}, {0, 1}, {s, s}},
	)
	idx, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if idx.Len() != 3 {
		t.Fatalf("Len = %d, want 3", idx.Len())
	}
	if idx.Dim() != 2 {
		t.Fatalf("Dim = %d, want 2", idx.Dim())
	}

	got := idx.TopK([]float32{1, 0}, 2, nil)
	if len(got) != 2 || got[0].ID != 10 || got[1].ID != 30 {
		t.Fatalf("TopK order wrong: %+v", got)
	}
	if got[0].Score < 0.99 {
		t.Fatalf("exact match should score ~1, got %f", got[0].Score)
	}
}

func TestTopKUsesStableIDTieBreak(t *testing.T) {
	path := writeFixture(t,
		[]int32{30, 10, 20},
		[][]float32{{1, 0}, {1, 0}, {1, 0}},
	)
	idx, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	got := idx.TopK([]float32{1, 0}, 2, nil)
	if len(got) != 2 || got[0].ID != 10 || got[1].ID != 20 {
		t.Fatalf("tie order wrong: %+v", got)
	}
}

func TestLoadFloat16Artifact(t *testing.T) {
	// [1, 0], [0.5, 1] in IEEE 754 half precision.
	path := writeHalfFixture(t, []int32{1, 2}, 2, []uint16{0x3c00, 0, 0x3800, 0x3c00})
	idx, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	got := idx.TopK([]float32{0, 1}, 1, nil)
	if len(got) != 1 || got[0].ID != 2 {
		t.Fatalf("float16 artifact ranked incorrectly: %+v", got)
	}
}

func TestTopKMatchesFullSortOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	const count = 300
	const dim = 8
	ids := make([]int32, count)
	vectors := make([][]float32, count)
	query := make([]float32, dim)
	for d := range query {
		query[d] = rng.Float32()*2 - 1
	}
	for row := 0; row < count; row++ {
		ids[row] = int32(row + 1)
		vectors[row] = make([]float32, dim)
		for d := range vectors[row] {
			vectors[row][d] = rng.Float32()*2 - 1
		}
	}
	idx, err := Load(writeFixture(t, ids, vectors))
	if err != nil {
		t.Fatal(err)
	}
	exclude := map[int]bool{3: true, 9: true, 100: true}
	want := make([]Scored, 0, count-len(exclude))
	for row, id := range ids {
		if exclude[int(id)] {
			continue
		}
		var dot float64
		for d := range query {
			dot += float64(query[d] * vectors[row][d])
		}
		want = append(want, Scored{ID: int(id), Score: dot})
	}
	sort.Slice(want, func(i, j int) bool { return better(want[i], want[j]) })
	got := idx.TopK(query, 25, exclude)
	for i := range got {
		if got[i].ID != want[i].ID || math.Abs(got[i].Score-want[i].Score) > 1e-6 {
			t.Fatalf("rank %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestTopKHandlesAllExcludedAndWrongDimension(t *testing.T) {
	idx, err := Load(writeFixture(t, []int32{1, 2}, [][]float32{{1, 0}, {0, 1}}))
	if err != nil {
		t.Fatal(err)
	}
	if got := idx.TopK([]float32{1, 0}, 10, map[int]bool{1: true, 2: true}); len(got) != 0 {
		t.Fatalf("all-excluded query returned %+v", got)
	}
	if got := idx.TopK([]float32{1}, 1, nil); got != nil {
		t.Fatalf("wrong-dimension query returned %+v", got)
	}
}

func TestTopKExcludes(t *testing.T) {
	path := writeFixture(t,
		[]int32{1, 2},
		[][]float32{{1, 0}, {0.9, 0.1}},
	)
	idx, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	got := idx.TopK([]float32{1, 0}, 5, map[int]bool{1: true})
	if len(got) != 1 || got[0].ID != 2 {
		t.Fatalf("exclusion failed: %+v", got)
	}
}

func TestMissingIDIsColdStart(t *testing.T) {
	path := writeFixture(t, []int32{1}, [][]float32{{1, 0}})
	idx, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if idx.Has(999) {
		t.Fatal("Has(999) should be false")
	}
	if idx.Vector(999) != nil {
		t.Fatal("Vector(999) should be nil")
	}
	// A nil index (embeddings never loaded) must degrade, not panic.
	var none *Index
	if none.Has(1) || none.Vector(1) != nil || none.TopK([]float32{1}, 3, nil) != nil || none.Len() != 0 {
		t.Fatal("nil index should behave as empty")
	}
}

func TestLoadRejectsCorruptFiles(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.bin")
	if err := os.WriteFile(path, []byte("EMB1xxxx"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil {
		t.Fatal("truncated file should fail to load")
	}
}

func TestLoadRejectsDuplicateIDsAndNonFiniteValues(t *testing.T) {
	duplicate := writeFixture(t, []int32{1, 1}, [][]float32{{1, 0}, {0, 1}})
	if _, err := Load(duplicate); err == nil {
		t.Fatal("duplicate ids should fail to load")
	}

	nonFinite := writeFixture(t, []int32{1}, [][]float32{{float32(math.NaN()), 0}})
	if _, err := Load(nonFinite); err == nil {
		t.Fatal("non-finite values should fail to load")
	}
}

func TestFloat16Roundtrip(t *testing.T) {
	cases := map[uint16]float32{
		0x3c00: 1.0,
		0xbc00: -1.0,
		0x3800: 0.5,
		0x0000: 0.0,
	}
	for h, want := range cases {
		if got := float16to32(h); got != want {
			t.Fatalf("float16to32(%#x) = %f, want %f", h, got, want)
		}
	}
}

func BenchmarkTopK87585x64(b *testing.B) {
	const count = 87585
	const dim = 64
	idx := &Index{
		dim:  dim,
		ids:  make([]int32, count),
		rows: make(map[int]int, count),
		vecs: make([]float32, count*dim),
	}
	for row := 0; row < count; row++ {
		idx.ids[row] = int32(row + 1)
		idx.rows[row+1] = row
		for d := 0; d < dim; d++ {
			idx.vecs[row*dim+d] = float32((row+d)%101) / 101
		}
	}
	query := make([]float32, dim)
	for i := range query {
		query[i] = float32(i+1) / dim
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.TopK(query, 25, nil)
	}
}
