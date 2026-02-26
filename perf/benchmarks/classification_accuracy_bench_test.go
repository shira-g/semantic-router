//go:build !windows && cgo

package benchmarks

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
)

const INTENT_GOLD_DATASET_PATH = "INTENT_GOLD_DATASET_PATH"
const INTENT_COMPARE_DEBUG_ENV = "VSR_DEBUG_INTENT_COMPARE"
const INTENT_CATEGORY_THRESHOLD_ENV = "INTENT_CATEGORY_CONFIDENCE_THRESHOLD"
const INTENT_UNKNOWN_LABEL = "unknown"

type intentGoldExample struct {
	Text       string
	Question   string
	Options    []string
	RouterPrompt string
	GoldIntent string
}

type intentGoldRow struct {
	Text       string `json:"text"`
	Question   string `json:"question"`
	Options    []string `json:"options"`
	RouterPrompt string `json:"router_prompt"`
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

func resolveIntentCategoryThreshold() (float64, error) {
	raw := strings.TrimSpace(os.Getenv(INTENT_CATEGORY_THRESHOLD_ENV))
	if raw == "" {
		return 0.0, nil
	}

	threshold, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return 0.0, fmt.Errorf("invalid %s=%q: %w", INTENT_CATEGORY_THRESHOLD_ENV, raw, err)
	}
	if threshold < 0.0 || threshold > 1.0 {
		return 0.0, fmt.Errorf("invalid %s=%q: must be in [0.0, 1.0]", INTENT_CATEGORY_THRESHOLD_ENV, raw)
	}

	return threshold, nil
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
			Question:   row.Question,
			Options:    row.Options,
			RouterPrompt: row.RouterPrompt,
			GoldIntent: gold,
		})
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return examples, nil
}

func runIntentAccuracyBenchmark(b *testing.B, useRouterPrompt bool) {
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

	threshold, thresholdErr := resolveIntentCategoryThreshold()
	if thresholdErr != nil {
		b.Fatalf("%v", thresholdErr)
	}

	normalizedGold := make([]string, len(examples))
	for i := range examples {
		normalizedGold[i] = normalizeIntentLabel(examples[i].GoldIntent)
	}

	b.ReportAllocs()
	b.ResetTimer()

	if useRouterPrompt {
		hasRouterPrompt := false
		for _, ex := range examples {
			if strings.TrimSpace(ex.RouterPrompt) != "" {
				hasRouterPrompt = true
				break
			}
		}
		if !hasRouterPrompt {
			b.Skipf("dataset=%s has no router_prompt field; regenerate via perf/scripts/export_mmlu_intent_gold.py", datasetPath)
		}
	}

	var accuracy float64
	for i := 0; i < b.N; i++ {
		correct := 0
		unknownCount := 0
		for idx, ex := range examples {
			inputText := ex.Text
			if useRouterPrompt {
				inputText = ex.RouterPrompt
			}

			results, classifyErr := classifier.ClassifyBatch([]string{inputText})
			if classifyErr != nil {
				b.Fatalf("classification failed: %v", classifyErr)
			}
			if len(results.IntentResults) == 0 {
				continue
			}

			confidence := float64(results.IntentResults[0].Confidence)
			predicted := normalizeIntentLabel(results.IntentResults[0].Category)
			if confidence < threshold {
				predicted = INTENT_UNKNOWN_LABEL
				unknownCount++
			}
			if os.Getenv(INTENT_COMPARE_DEBUG_ENV) == "1" {
				flow := "perf"
				if useRouterPrompt {
					flow = "perf-router-prompt"
				}
				fmt.Printf(
					"[IntentCompare][%s] text=%q predicted=%q confidence=%.4f threshold=%.4f gold=%q\n",
					flow,
					inputText,
					predicted,
					confidence,
					threshold,
					normalizedGold[idx],
				)
			}
			if predicted == normalizedGold[idx] {
				correct++
			}
		}
		accuracy = float64(correct) / float64(len(examples))
		if i == b.N-1 {
			b.ReportMetric(float64(unknownCount), "intent_unknown_count")
			b.ReportMetric((float64(unknownCount)/float64(len(examples)))*100.0, "intent_unknown_pct")
		}
	}

	b.StopTimer()
	b.ReportMetric(accuracy*100, "intent_acc_pct")
	mode := "raw_text"
	if useRouterPrompt {
		mode = "router_prompt"
	}
	b.Logf("dataset=%s mode=%s samples=%d threshold=%.4f intent_accuracy=%.2f%%", datasetPath, mode, len(examples), threshold, accuracy*100)
}

func BenchmarkClassifyIntent_GoldAccuracy(b *testing.B) {
	runIntentAccuracyBenchmark(b, false)
}

func BenchmarkClassifyIntent_GoldAccuracy_RouterPrompt(b *testing.B) {
	runIntentAccuracyBenchmark(b, true)
}
