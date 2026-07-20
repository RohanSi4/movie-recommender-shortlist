package retrieval

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

// Item-support stats written by ml/scripts/export_embeddings.py:
// magic "STA1", uint32 count, count int32 ids, count uint32 counts, all
// little-endian. Counts are per-item positive interactions in the training
// window, the single source of truth for both the popularity blend and the
// per-seed warmth signal.
const statsMagic = "STA1"

// loadItemStats reads the per-item training-support counts.
func loadItemStats(path string) (ids []int32, counts []uint32, err error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	if len(raw) < 8 || string(raw[:4]) != statsMagic {
		return nil, nil, fmt.Errorf("%s: not a STA1 stats file", path)
	}
	count := int(binary.LittleEndian.Uint32(raw[4:8]))
	if count <= 0 {
		return nil, nil, fmt.Errorf("%s: invalid count %d", path, count)
	}
	want := 8 + count*4 + count*4
	if len(raw) != want {
		return nil, nil, fmt.Errorf("%s: size %d does not match header (want %d)", path, len(raw), want)
	}
	ids = make([]int32, count)
	counts = make([]uint32, count)
	off := 8
	for i := 0; i < count; i++ {
		ids[i] = int32(binary.LittleEndian.Uint32(raw[off : off+4]))
		off += 4
	}
	for i := 0; i < count; i++ {
		counts[i] = binary.LittleEndian.Uint32(raw[off : off+4])
		off += 4
	}
	return ids, counts, nil
}

// AttachPopularity aligns per-item training-support counts to the index rows
// and precomputes a normalized log-count popularity score (mean 0, unit
// variance) matching ml/scripts/evaluate_taste_retrieval.py. Index rows with
// no matching stat default to zero support. Stats for ids absent from the
// index are ignored.
func (x *Index) AttachPopularity(ids []int32, counts []uint32) {
	if x == nil {
		return
	}
	byID := make(map[int]uint32, len(ids))
	for i, id := range ids {
		byID[int(id)] = counts[i]
	}
	support := make([]int32, len(x.ids))
	logs := make([]float64, len(x.ids))
	var sum float64
	for row := range x.ids {
		c := byID[int(x.ids[row])]
		support[row] = int32(c)
		l := math.Log1p(float64(c))
		logs[row] = l
		sum += l
	}
	n := float64(len(logs))
	mean := sum / n
	var sumSq float64
	for _, l := range logs {
		d := l - mean
		sumSq += d * d
	}
	std := math.Sqrt(sumSq/n) + 1e-6
	pop := make([]float32, len(logs))
	for i, l := range logs {
		pop[i] = float32((l - mean) / std)
	}
	x.support = support
	x.pop = pop
}

// HasPopularity reports whether popularity stats are attached.
func (x *Index) HasPopularity() bool {
	return x != nil && x.pop != nil
}

// Support returns the training-support count for id, and false when stats are
// missing or the id is unknown.
func (x *Index) Support(id int) (int, bool) {
	if x == nil || x.support == nil {
		return 0, false
	}
	row, ok := x.rows[id]
	if !ok {
		return 0, false
	}
	return int(x.support[row]), true
}

// Warmth maps a training-support count to a 0..1 confidence that an item's
// embedding carries real collaborative signal. Zero support is fully cold;
// warmRef positives and above is fully warm, smoothed in log space so the
// climb from cold to warm is gradual rather than a cliff.
func Warmth(support int, warmRef float64) float64 {
	if support <= 0 {
		return 0
	}
	if warmRef <= 1 {
		warmRef = 2
	}
	w := math.Log1p(float64(support)) / math.Log1p(warmRef)
	if w > 1 {
		return 1
	}
	return w
}
