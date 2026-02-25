//go:build !windows && cgo

package benchmarks

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
)

const INTENT_GOLD_DATASET_PATH = "INTENT_GOLD_DATASET_PATH"

type intentGoldExample struct {
	Text       string
	GoldIntent string
}

type intentGoldRow struct {
	Text       string `json:"text"`
	GoldIntent string `json:"gold_intent"`
	Label      string `json:"label"`
	Intent     string `json:"intent"`
}

func resolveIntentGoldDatasetPath() string {
	if path := os.Getenv(INTENT_GOLD_DATASET_PATH); path != "" {
		return path
	}

	// Prefer MMLU-derived dataset (exported via perf/scripts/export_mmlu_intent_gold.py)
	mmluPath := filepath.Join("..", "testdata", "mmlu_intent_gold.jsonl")
	if _, err := os.Stat(mmluPath); err == nil {
		return mmluPath
	}

	// Fallback starter dataset
	return filepath.Join("..", "testdata", "intent_gold.jsonl")
}

func normalizeIntentLabel(label string) string {
	return strings.ToLower(strings.TrimSpace(label))
}

func loadIntentGoldExamples(path string) ([]intentGoldExample, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	examples := make([]intentGoldExample, 0)
	scanner := bufio.NewScanner(file)
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		var row intentGoldRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return nil, fmt.Errorf("invalid JSONL at line %d: %w", lineNum, err)
		}

		gold := row.GoldIntent
		if gold == "" {
			gold = row.Label
		}
		if gold == "" {
			gold = row.Intent
		}
		if strings.TrimSpace(row.Text) == "" || strings.TrimSpace(gold) == "" {
			return nil, fmt.Errorf("missing text or gold label at line %d", lineNum)
		}

		examples = append(examples, intentGoldExample{
			Text:       row.Text,
			GoldIntent: gold,
		})
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return examples, nil
}

func BenchmarkClassifyIntent_GoldAccuracy(b *testing.B) {
	initClassifier(b)
	classifier := classification.GetGlobalUnifiedClassifier()

	datasetPath := resolveIntentGoldDatasetPath()
	examples, err := loadIntentGoldExamples(datasetPath)
	if err != nil {
		b.Skipf("intent gold dataset not ready (%v); set %s or provide %s", err, INTENT_GOLD_DATASET_PATH, datasetPath)
	}
	if len(examples) == 0 {
		b.Skipf("intent gold dataset is empty: %s", datasetPath)
	}

	normalizedGold := make([]string, len(examples))
	for i := range examples {
		normalizedGold[i] = normalizeIntentLabel(examples[i].GoldIntent)
	}

	b.ReportAllocs()
	b.ResetTimer()

	var accuracy float64
	for i := 0; i < b.N; i++ {
		correct := 0
		for idx, ex := range examples {
			results, classifyErr := classifier.ClassifyBatch([]string{ex.Text})
			if classifyErr != nil {
				b.Fatalf("classification failed: %v", classifyErr)
			}
			if len(results.IntentResults) == 0 {
				continue
			}

			predicted := normalizeIntentLabel(results.IntentResults[0].Category)
			if predicted == normalizedGold[idx] {
				correct++
			}
		}
		accuracy = float64(correct) / float64(len(examples))
	}

	b.StopTimer()
	b.ReportMetric(accuracy*100, "intent_acc_pct")
	b.Logf("dataset=%s samples=%d intent_accuracy=%.2f%%", datasetPath, len(examples), accuracy*100)
}
