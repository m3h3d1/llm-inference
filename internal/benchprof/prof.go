package benchprof

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"time"
)

var enabled bool
var once sync.Once

type stat struct {
	mu        sync.Mutex
	calls     int
	totalNS   int64
	maxNS     int64
	lastShape string
}

func (s *stat) record(d time.Duration, shape string) {
	s.mu.Lock()
	s.calls++
	ns := d.Nanoseconds()
	s.totalNS += ns
	if ns > s.maxNS {
		s.maxNS = ns
	}
	s.lastShape = shape
	s.mu.Unlock()
}

func (s *stat) snapshot() (calls int, avg, max time.Duration, shape string) {
	s.mu.Lock()
	calls = s.calls
	if calls > 0 {
		avg = time.Duration(s.totalNS / int64(calls))
	}
	max = time.Duration(s.maxNS)
	shape = s.lastShape
	s.mu.Unlock()
	return
}

func (s *stat) reset() {
	s.mu.Lock()
	s.calls = 0
	s.totalNS = 0
	s.maxNS = 0
	s.lastShape = ""
	s.mu.Unlock()
}

var (
	linearForward               stat
	outputLogits                stat
	attentionForward            stat
	attentionForwardWithCache   stat
)

func Enabled() bool {
	once.Do(func() {
		enabled = os.Getenv("LLM_BENCH_DEBUG") != ""
	})
	return enabled
}

func ResetAll() {
	linearForward.reset()
	outputLogits.reset()
	attentionForward.reset()
	attentionForwardWithCache.reset()
}

func RecordLinearForward(d time.Duration, batch, seq, inFeatures, outFeatures int) {
	if !Enabled() {
		return
	}
	shape := fmt.Sprintf("b=%d seq=%d in=%d out=%d", batch, seq, inFeatures, outFeatures)
	linearForward.record(d, shape)
}

func RecordOutputLogits(d time.Duration, batch, seq, embDim, vocabSize int) {
	if !Enabled() {
		return
	}
	shape := fmt.Sprintf("b=%d seq=%d emb=%d vocab=%d", batch, seq, embDim, vocabSize)
	outputLogits.record(d, shape)
}

func RecordAttentionForward(d time.Duration, seq, nHeads, nKVHeads, headDim int) {
	if !Enabled() {
		return
	}
	shape := fmt.Sprintf("seq=%d heads=%d kvHeads=%d dim=%d", seq, nHeads, nKVHeads, headDim)
	attentionForward.record(d, shape)
}

func RecordAttentionForwardWithCache(d time.Duration, qSeq, kvSeq, nHeads, nKVHeads, headDim int) {
	if !Enabled() {
		return
	}
	shape := fmt.Sprintf("q=%d kv=%d heads=%d kvHeads=%d dim=%d", qSeq, kvSeq, nHeads, nKVHeads, headDim)
	attentionForwardWithCache.record(d, shape)
}

func Summary() string {
	if !Enabled() {
		return ""
	}
	var b strings.Builder
	writeStat(&b, "linear_forward", &linearForward)
	writeStat(&b, "output_logits", &outputLogits)
	writeStat(&b, "attention_forward", &attentionForward)
	writeStat(&b, "attention_forward_with_cache", &attentionForwardWithCache)
	return b.String()
}

func writeStat(b *strings.Builder, name string, s *stat) {
	calls, avg, max, shape := s.snapshot()
	if calls == 0 {
		return
	}
	b.WriteString(fmt.Sprintf("  %s: calls=%d total=%v avg=%v max=%v\n", name, calls,
		time.Duration(int64(calls))*avg, avg, max))
	if shape != "" {
		b.WriteString(fmt.Sprintf("    last_shape=%s\n", shape))
	}
}
