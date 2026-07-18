package retrieval

import (
	"encoding/binary"
	"fmt"
	"os"
)

const historyMagic = "HST1"

// History stores delta-varint encoded movie ids for each user. It keeps the
// artifact compressed in memory and only decodes one user's history per query.
type History struct {
	userIDs []int32
	rows    map[int]int
	offsets []uint64
	payload []byte
}

// LoadHistory reads the history artifact written by export_embeddings.py.
func LoadHistory(path string) (*History, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) < 8 || string(raw[:4]) != historyMagic {
		return nil, fmt.Errorf("%s: not an HST1 history file", path)
	}
	count := int(binary.LittleEndian.Uint32(raw[4:8]))
	if count <= 0 {
		return nil, fmt.Errorf("%s: invalid user count %d", path, count)
	}
	headerSize := uint64(8) + uint64(count)*4 + uint64(count+1)*8
	if headerSize > uint64(len(raw)) {
		return nil, fmt.Errorf("%s: truncated history header", path)
	}

	history := &History{
		userIDs: make([]int32, count),
		rows:    make(map[int]int, count),
		offsets: make([]uint64, count+1),
	}
	off := 8
	for row := 0; row < count; row++ {
		id := int32(binary.LittleEndian.Uint32(raw[off : off+4]))
		if _, exists := history.rows[int(id)]; exists {
			return nil, fmt.Errorf("%s: duplicate user id %d", path, id)
		}
		history.userIDs[row] = id
		history.rows[int(id)] = row
		off += 4
	}
	for i := range history.offsets {
		history.offsets[i] = binary.LittleEndian.Uint64(raw[off : off+8])
		if i > 0 && history.offsets[i] < history.offsets[i-1] {
			return nil, fmt.Errorf("%s: history offsets are not ordered", path)
		}
		off += 8
	}
	history.payload = raw[off:]
	if history.offsets[0] != 0 || history.offsets[len(history.offsets)-1] != uint64(len(history.payload)) {
		return nil, fmt.Errorf("%s: history offsets do not match payload", path)
	}

	for row := 0; row < count; row++ {
		if _, err := history.decodeRow(row); err != nil {
			return nil, fmt.Errorf("%s: user %d: %w", path, history.userIDs[row], err)
		}
	}
	return history, nil
}

// Len reports the number of users with stored history.
func (h *History) Len() int {
	if h == nil {
		return 0
	}
	return len(h.userIDs)
}

// ExcludeSet returns the movies already rated by user. Unknown users return
// nil so callers naturally fall through to cold-start behavior.
func (h *History) ExcludeSet(userID int) map[int]bool {
	if h == nil {
		return nil
	}
	row, ok := h.rows[userID]
	if !ok {
		return nil
	}
	ids, err := h.decodeRow(row)
	if err != nil || len(ids) == 0 {
		return nil
	}
	exclude := make(map[int]bool, len(ids))
	for _, id := range ids {
		exclude[id] = true
	}
	return exclude
}

func (h *History) decodeRow(row int) ([]int, error) {
	start := h.offsets[row]
	end := h.offsets[row+1]
	if start > end || end > uint64(len(h.payload)) {
		return nil, fmt.Errorf("invalid payload range %d:%d", start, end)
	}
	encoded := h.payload[start:end]
	ids := make([]int, 0, 64)
	previous := uint64(0)
	for len(encoded) > 0 {
		delta, n := binary.Uvarint(encoded)
		if n <= 0 || delta == 0 {
			return nil, fmt.Errorf("invalid movie id delta")
		}
		previous += delta
		if previous > uint64(^uint(0)>>1) {
			return nil, fmt.Errorf("movie id overflows int")
		}
		ids = append(ids, int(previous))
		encoded = encoded[n:]
	}
	return ids, nil
}
