//go:build !windows && cgo

package benchmarks

import (
	"bufio"
	"encoding/csv"
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
	Text         string   `json:"text"`
	Question     string   `json:"question"`
	Options      []string `json:"options"`
	RouterPrompt string `json:"router_prompt"`
	GoldIntent   string   `json:"gold_intent"`
	Label        string   `json:"label"`
	Intent       string   `json:"intent"`
}

type intentPredictionLogRow struct {
	Index             int
	Mode              string
	InputText         string
	GoldIntent        string
	RawPredictedCategory string
	PredictedCategory string
	PredictedCategoryReason string
	Confidence        float64
	Threshold         float64
	IsCorrect         bool
}

func writeIntentPredictionLog(path string, rows []intentPredictionLogRow) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create log directory: %w", err)
	}

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create log file: %w", err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	if err := w.Write([]string{
		"index",
		"mode",
		"input_text",
		"gold_intent",
		"raw_predicted_category",
		"predicted_category",
		"predicted_category_reason",
		"confidence",
		"threshold",
		"is_correct",
	}); err != nil {
		return fmt.Errorf("write csv header: %w", err)
	}

	for _, row := range rows {
		if err := w.Write([]string{
			strconv.Itoa(row.Index),
			row.Mode,
			row.InputText,
			row.GoldIntent,
			row.RawPredictedCategory,
			row.PredictedCategory,
			row.PredictedCategoryReason,
			fmt.Sprintf("%.6f", row.Confidence),
			fmt.Sprintf("%.6f", row.Threshold),
			strconv.FormatBool(row.IsCorrect),
		}); err != nil {
			return fmt.Errorf("write csv row: %w", err)
		}
	}

	if err := w.Error(); err != nil {
		return fmt.Errorf("flush csv writer: %w", err)
	}

	return nil
}

func resolveIntentPredictionLogPath(mode string) string {
	fileName := fmt.Sprintf("intent_predicted_category_log_%s.csv", mode)

	// When invoked via `make` from repo root, cwd is typically `perf` and
	// ../reports is correct. If go test executes with cwd at perf/benchmarks,
	// ../../reports is the repo-level reports directory.
	candidates := []string{
		filepath.Join("..", "reports", fileName),
		filepath.Join("..", "..", "reports", fileName),
	}

	for _, p := range candidates {
		dir := filepath.Dir(p)
		if st, err := os.Stat(dir); err == nil && st.IsDir() {
			return p
		}
	}

	// Fallback to repo-level reports path relative to perf/benchmarks.
	return candidates[1]
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
	var finalLogRows []intentPredictionLogRow
	mode := "raw_text"
	if useRouterPrompt {
		mode = "router_prompt"
	}
	for i := 0; i < b.N; i++ {
		correct := 0
		unknownCount := 0
		iterLogRows := make([]intentPredictionLogRow, 0, len(examples))
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

			for resultIdx, intentResult := range results.IntentResults {
				if len(intentResult.Probabilities) > 0 {
					fmt.Printf(
						"[IntentResults][%d] category=%q confidence=%.6f probabilities=%v\n",
						resultIdx,
						intentResult.Category,
						intentResult.Confidence,
						intentResult.Probabilities,
					)
				} else {
					fmt.Printf(
						"[IntentResults][%d] category=%q confidence=%.6f probabilities=<unavailable>\n",
						resultIdx,
						intentResult.Category,
						intentResult.Confidence,
					)
				}
			}
			confidence := float64(results.IntentResults[0].Confidence)
			rawPredicted := normalizeIntentLabel(results.IntentResults[0].Category)
			predicted := rawPredicted
			reason := fmt.Sprintf(
				"classifier returned %q with confidence=%.6f (threshold=%.6f)",
				rawPredicted,
				confidence,
				threshold,
			)
			if confidence < threshold {
				predicted = INTENT_UNKNOWN_LABEL
				unknownCount++
				reason = fmt.Sprintf(
					"confidence %.6f below threshold %.6f; downgraded to %q (raw=%q)",
					confidence,
					threshold,
					INTENT_UNKNOWN_LABEL,
					rawPredicted,
				)
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

			iterLogRows = append(iterLogRows, intentPredictionLogRow{
				Index:                   idx,
				Mode:                    mode,
				InputText:               inputText,
				GoldIntent:              normalizedGold[idx],
				RawPredictedCategory:    rawPredicted,
				PredictedCategory:       predicted,
				PredictedCategoryReason: reason,
				Confidence:              confidence,
				Threshold:               threshold,
				IsCorrect:               predicted == normalizedGold[idx],
			})
		}
		accuracy = float64(correct) / float64(len(examples))
		if i == b.N-1 {
			b.ReportMetric(float64(unknownCount), "intent_unknown_count")
			b.ReportMetric((float64(unknownCount)/float64(len(examples)))*100.0, "intent_unknown_pct")
			finalLogRows = iterLogRows
		}
	}

	b.StopTimer()
	b.ReportMetric(accuracy*100, "intent_acc_pct")
	logFilePath := resolveIntentPredictionLogPath(mode)
	if err := writeIntentPredictionLog(logFilePath, finalLogRows); err != nil {
		b.Logf("failed to write predicted category log: %v", err)
	} else {
		b.Logf("predicted category log: %s", logFilePath)
	}

	b.Logf("dataset=%s mode=%s samples=%d threshold=%.4f intent_accuracy=%.2f%%", datasetPath, mode, len(examples), threshold, accuracy*100)
}

func BenchmarkClassifyIntent_GoldAccuracy(b *testing.B) {
	runIntentAccuracyBenchmark(b, false)
}

func BenchmarkClassifyIntent_GoldAccuracy_RouterPrompt(b *testing.B) {
	runIntentAccuracyBenchmark(b, true)
}
