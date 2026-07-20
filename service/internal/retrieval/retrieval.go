// Package retrieval serves two-tower embedding lookups: personalized
// candidate generation (user vector against every item vector) and learned
// item-to-item similarity. Vectors are exact, not approximate: at 87,585
// items x 64 dims a full top-25 scan takes about 3 to 4 milliseconds on the
// development machine, which keeps ANN complexity unnecessary at this scale.
package retrieval

import (
	"container/heap"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
)

// File layout written by ml/scripts/export_embeddings.py:
// magic "EMB1", uint32 count, uint32 dim, uint32 bytes-per-value
// (2 = float16, 4 = float32), count int32 ids, count*dim values, row-major,
// all little-endian.
const magic = "EMB1"

// Index holds one embedding table in float32, plus an id lookup. When item
// stats are attached (see stats.go), pop and support are aligned to the same
// row order as vecs and drive the cold-seed popularity blend.
type Index struct {
	dim     int
	ids     []int32
	rows    map[int]int // external id -> row offset
	vecs    []float32   // len = len(ids) * dim, row-major
	pop     []float32   // normalized popularity score per row, nil until attached
	support []int32     // training-support count per row, nil until attached
}

// Scored pairs an external id with a dot-product score.
type Scored struct {
	ID    int
	Score float64
}

// Load reads one embedding file. It returns an error rather than panicking
// so the caller can treat missing embeddings as a degraded mode.
func Load(path string) (*Index, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) < 16 || string(raw[:4]) != magic {
		return nil, fmt.Errorf("%s: not an EMB1 embedding file", path)
	}
	count := int(binary.LittleEndian.Uint32(raw[4:8]))
	dim := int(binary.LittleEndian.Uint32(raw[8:12]))
	itemSize := int(binary.LittleEndian.Uint32(raw[12:16]))
	if count <= 0 || dim <= 0 {
		return nil, fmt.Errorf("%s: invalid shape %d x %d", path, count, dim)
	}
	if itemSize != 2 && itemSize != 4 {
		return nil, fmt.Errorf("%s: unsupported value size %d", path, itemSize)
	}
	want := uint64(16) + uint64(count)*4 + uint64(count)*uint64(dim)*uint64(itemSize)
	if want > uint64(^uint(0)>>1) || uint64(len(raw)) != want {
		return nil, fmt.Errorf("%s: size %d does not match header (want %d)", path, len(raw), want)
	}

	idx := &Index{
		dim:  dim,
		ids:  make([]int32, count),
		rows: make(map[int]int, count),
		vecs: make([]float32, count*dim),
	}
	off := 16
	for i := 0; i < count; i++ {
		idx.ids[i] = int32(binary.LittleEndian.Uint32(raw[off : off+4]))
		id := int(idx.ids[i])
		if _, exists := idx.rows[id]; exists {
			return nil, fmt.Errorf("%s: duplicate id %d", path, id)
		}
		idx.rows[id] = i
		off += 4
	}
	for i := range idx.vecs {
		if itemSize == 2 {
			idx.vecs[i] = float16to32(binary.LittleEndian.Uint16(raw[off : off+2]))
			off += 2
		} else {
			idx.vecs[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[off : off+4]))
			off += 4
		}
		if math.IsNaN(float64(idx.vecs[i])) || math.IsInf(float64(idx.vecs[i]), 0) {
			return nil, fmt.Errorf("%s: non-finite embedding value at offset %d", path, i)
		}
	}
	return idx, nil
}

// Len reports how many vectors the index holds.
func (x *Index) Len() int {
	if x == nil {
		return 0
	}
	return len(x.ids)
}

// Dim reports the width of each vector.
func (x *Index) Dim() int {
	if x == nil {
		return 0
	}
	return x.dim
}

// Has reports whether the id has an embedding.
func (x *Index) Has(id int) bool {
	if x == nil {
		return false
	}
	_, ok := x.rows[id]
	return ok
}

// Vector returns the embedding for id, or nil when absent.
func (x *Index) Vector(id int) []float32 {
	if x == nil {
		return nil
	}
	row, ok := x.rows[id]
	if !ok {
		return nil
	}
	return x.vecs[row*x.dim : (row+1)*x.dim]
}

// TopK scores query against every vector in the index and returns the k
// best by dot product, skipping any id in exclude. Vectors are exported
// L2-normalized, so dot product equals cosine similarity.
func (x *Index) TopK(query []float32, k int, exclude map[int]bool) []Scored {
	return x.topK(query, k, exclude, 0)
}

// TopKBlended scores dot(query, item) + popWeight * popScore(item). It pulls
// results toward popular titles when the query is unreliable, which is how the
// service rescues cold seeds. A popWeight of zero, or an index without stats,
// is identical to TopK.
func (x *Index) TopKBlended(query []float32, k int, exclude map[int]bool, popWeight float64) []Scored {
	return x.topK(query, k, exclude, popWeight)
}

func (x *Index) topK(query []float32, k int, exclude map[int]bool, popWeight float64) []Scored {
	if x == nil || len(query) != x.dim || k <= 0 {
		return nil
	}
	if k > len(x.ids) {
		k = len(x.ids)
	}
	blend := popWeight != 0 && x.pop != nil
	best := make(topKHeap, 0, k)
	for row := 0; row < len(x.ids); row++ {
		id := int(x.ids[row])
		if exclude != nil && exclude[id] {
			continue
		}
		base := row * x.dim
		var dot float32
		for d := 0; d < x.dim; d++ {
			dot += query[d] * x.vecs[base+d]
		}
		score := float64(dot)
		if blend {
			score += popWeight * float64(x.pop[row])
		}
		candidate := Scored{ID: id, Score: score}
		if len(best) < k {
			heap.Push(&best, candidate)
			continue
		}
		if better(candidate, best[0]) {
			best[0] = candidate
			heap.Fix(&best, 0)
		}
	}
	sort.Slice(best, func(i, j int) bool { return better(best[i], best[j]) })
	return best
}

func better(a, b Scored) bool {
	if a.Score == b.Score {
		return a.ID < b.ID
	}
	return a.Score > b.Score
}

// topKHeap keeps the worst retained result at index zero.
type topKHeap []Scored

func (h topKHeap) Len() int { return len(h) }
func (h topKHeap) Less(i, j int) bool {
	if h[i].Score == h[j].Score {
		return h[i].ID > h[j].ID
	}
	return h[i].Score < h[j].Score
}
func (h topKHeap) Swap(i, j int)   { h[i], h[j] = h[j], h[i] }
func (h *topKHeap) Push(value any) { *h = append(*h, value.(Scored)) }
func (h *topKHeap) Pop() any {
	old := *h
	last := old[len(old)-1]
	*h = old[:len(old)-1]
	return last
}

// float16to32 widens an IEEE 754 half-precision value.
func float16to32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h) & 0x3ff

	var bits uint32
	switch {
	case exp == 0 && frac == 0: // signed zero
		bits = sign << 31
	case exp == 0: // subnormal: renormalize
		e := uint32(127 - 15 + 1)
		for frac&0x400 == 0 {
			frac <<= 1
			e--
		}
		frac &= 0x3ff
		bits = sign<<31 | e<<23 | frac<<13
	case exp == 0x1f: // inf / NaN
		bits = sign<<31 | 0xff<<23 | frac<<13
	default:
		bits = sign<<31 | (exp-15+127)<<23 | frac<<13
	}
	return math.Float32frombits(bits)
}
