import csv
import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer

torch.set_printoptions(precision=4, sci_mode=False)

cat_num = 14

phi3_category_performance = [0.6, 0.1, 0.2, 0.0, 0.6, 0.2, 0.5, 0.4, 0.6, 0.2, 0.5, 0.0, 0.1, 0.6]
gpt_oss_category_performance = [0.8, 0.6, 0.6, 0.7, 0.7, 0.7, 0.4, 0.6, 0.5, 1.0, 0.8, 0.2, 0.9, 0.4]
gpt_oss_reasoning_category_performance = [0.8, 0.5, 0.6, 0.5, 1.0, 0.7, 0.3, 0.6, 0.4, 0.9, 0.7, 0.4, 0.7, 0.5]

print(f"size of category performance: {len(phi3_category_performance)}")
print(f"size of category performance: {len(gpt_oss_category_performance)}")
print(f"size of category performance: {len(gpt_oss_reasoning_category_performance)}")

def get_model_score(model_vector, question_vector):
    model_tensor = torch.as_tensor(
        model_vector,
        dtype=question_vector.dtype,
        device=question_vector.device,
    )
    return (model_tensor * question_vector).sum()


SCRIPT_DIR = Path(__file__).resolve().parent
TESTDATA_DIR = SCRIPT_DIR.parent / "testdata"

phi3_results_file = TESTDATA_DIR / "phi3.csv"
gpt_oss_results_file = TESTDATA_DIR / "oss_low.csv"
gpt_oss_reasoning_results_file = TESTDATA_DIR / "oss_high.csv"
intent_classifier_model_dir = (
    SCRIPT_DIR.parent.parent / "models" / "lora_intent_classifier_bert-base-uncased_model"
)

intent_tokenizer = AutoTokenizer.from_pretrained(
    str(intent_classifier_model_dir),
    local_files_only=True,
)
intent_classifier_model = AutoModelForSequenceClassification.from_pretrained(
    str(intent_classifier_model_dir),
    local_files_only=True,
)
intent_classifier_model.eval()


# def load_questions_from_csv(file_path, question_column="question"):
#     questions = []
#     with open(file_path, newline="", encoding="utf-8") as csv_file:
#         reader = csv.DictReader(csv_file)
#         if reader.fieldnames is None or question_column not in reader.fieldnames:
#             raise ValueError(f"missing '{question_column}' column in {file_path}")

#         for row in reader:
#             question = (row.get(question_column) or "").strip()
#             if question:
#                 questions.append(question)

#     return questions


def load_column_from_csv(file_path, column_name):
    values = []
    with open(file_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or column_name not in reader.fieldnames:
            raise ValueError(f"missing '{column_name}' column in {file_path}")

        for row in reader:
            values.append((row.get(column_name) or "").strip())

    return values


def uniformity_stats(probs: torch.Tensor):
    k = probs.numel()
    eps = 1e-12
    p = probs.clamp_min(eps)
    u = torch.full_like(p, 1.0 / k)

    entropy = -(p * p.log()).sum()
    norm_entropy = entropy / torch.log(torch.tensor(float(k), device=p.device, dtype=p.dtype))
    kl_to_uniform = (p * (p.log() - u.log())).sum()
    max_prob = p.max()
    l1_to_uniform = torch.abs(p - u).sum()

    return {
        "max_prob": float(max_prob),
        "uniform_max_prob": 1.0 / k,
        "norm_entropy": float(norm_entropy),      # 1.0 => uniform
        "kl_to_uniform": float(kl_to_uniform),    # 0.0 => uniform
        "l1_to_uniform": float(l1_to_uniform),    # 0.0 => uniform
    }


# Use one of the results files above as the question source.
questions = load_column_from_csv(phi3_results_file, "question")

# Load correctness labels from each results file.
phi3_is_correct = load_column_from_csv(phi3_results_file, "is_correct")
gpt_oss_is_correct = load_column_from_csv(gpt_oss_results_file, "is_correct")
gpt_oss_reasoning_is_correct = load_column_from_csv(gpt_oss_reasoning_results_file, "is_correct")

phi3_runtimes = load_column_from_csv(phi3_results_file, "response_time")
gpt_oss_runtimes = load_column_from_csv(gpt_oss_results_file, "response_time")
gpt_oss_reasoning_runtimes = load_column_from_csv(gpt_oss_reasoning_results_file, "response_time")


print(f"size of correctness labels: {len(phi3_is_correct)}")
print(f"size of correctness labels: {len(gpt_oss_is_correct)}")
print(f"size of correctness labels: {len(gpt_oss_reasoning_is_correct)}")
print(f"is correct: {phi3_is_correct[:5]}")


def classify_question_relevance(question):
    encoded = intent_tokenizer(question, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = intent_classifier_model(**encoded).logits
        # print(f"logits: {logits}")
    return torch.softmax(logits[0], dim=-1)

performance_per_thres = []
for thres in range(0, 10):
    thres_use = thres / 10
    print(f"\n\nUsing threshold: {thres_use}")
    for q_index, q in zip(range(len(questions)), questions):
        
        question_vs_category_relevance = classify_question_relevance(q)
        print(f"question vs category relevance: {question_vs_category_relevance}")
        max_val = question_vs_category_relevance.max().item()

        for i, v in enumerate(question_vs_category_relevance):
            bar = "█" * int((v / max_val) * 40)
            print(f"{i:02d} | {v:.4f} | {bar}")

        stats = uniformity_stats(question_vs_category_relevance)
        print(f"uniformity stats: {stats}")

        print(f"question: {q}")
        print(f"question vs category relevance: {question_vs_category_relevance}")

        phi3_score = get_model_score(phi3_category_performance, question_vs_category_relevance)
        gpt_oss_score = get_model_score(gpt_oss_category_performance, question_vs_category_relevance)
        
        models_scores = torch.stack([phi3_score, gpt_oss_score])
        
        models_scores_norm = models_scores / models_scores.sum()
        
        phi3_norm_score = models_scores_norm[0]

        if phi3_norm_score > thres_use:
            print("Take phi3")
            is_correct = phi3_is_correct[q_index].lower() == "true"
            runtime = float(phi3_runtimes[q_index])
            model_used = "phi3"
        else:
            print("Take gpt oss")
            is_correct = gpt_oss_is_correct[q_index].lower() == "true"
            runtime = float(gpt_oss_runtimes[q_index])
            model_used = "gpt-oss"

        # correctness = check_if_answer_is_correct(answer_from_model, expected_answer)
        performance_per_thres.append({
            "question": q_index,
            "correctness": is_correct,
            "model_used": model_used,
            "thres_use": thres_use,
            "runtime_model": runtime
        })


# Plot average correctness per threshold.
thresholds = sorted({row["thres_use"] for row in performance_per_thres})
avg_correctness = []
avg_runtime = []
for threshold in thresholds:
    threshold_rows = [row for row in performance_per_thres if row["thres_use"] == threshold]
    if not threshold_rows:
        avg_correctness.append(0.0)
        avg_runtime.append(0.0)
        continue

    mean_correctness = sum(1.0 if row["correctness"] else 0.0 for row in threshold_rows) / len(threshold_rows)
    mean_runtime = sum(float(row["runtime_model"]) for row in threshold_rows) / len(threshold_rows)
    avg_correctness.append(mean_correctness)
    avg_runtime.append(mean_runtime)

print("here")
plt.figure(figsize=(8, 4.5))
plt.plot(thresholds, avg_correctness, marker="o")
plt.title("Average Correctness by Threshold")
plt.xlabel("Threshold")
plt.ylabel("Average Correctness")
plt.ylim(0.0, 1.0)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_path = SCRIPT_DIR / "avg_correctness_by_threshold.png"
plt.savefig(plot_path, dpi=150)
print(f"Saved plot to: {plot_path}")

if os.environ.get("DISPLAY"):
    plt.show()
else:
    plt.close()


# Plot average correctness vs average runtime.
plt.figure(figsize=(8, 4.5))
plt.plot(avg_runtime, avg_correctness, marker="o", color="tab:green")
for threshold, runtime, correctness in zip(thresholds, avg_runtime, avg_correctness):
    plt.annotate(f"t={threshold:.1f}", (runtime, correctness), textcoords="offset points", xytext=(4, 4), fontsize=8)
plt.title("Average Correctness vs Runtime")
plt.xlabel("Average Runtime")
plt.ylabel("Average Correctness")
plt.ylim(0.0, 1.0)
plt.grid(True, alpha=0.3)
plt.tight_layout()

tradeoff_plot_path = SCRIPT_DIR / "avg_correctness_vs_runtime.png"
plt.savefig(tradeoff_plot_path, dpi=150)
print(f"Saved plot to: {tradeoff_plot_path}")

if os.environ.get("DISPLAY"):
    plt.show()
else:
    plt.close()


# Plot average runtime per threshold.
plt.figure(figsize=(8, 4.5))
plt.plot(thresholds, avg_runtime, marker="o", color="tab:orange")
plt.title("Average Runtime by Threshold")
plt.xlabel("Threshold")
plt.ylabel("Average Runtime")
plt.grid(True, alpha=0.3)
plt.tight_layout()

runtime_plot_path = SCRIPT_DIR / "avg_runtime_by_threshold.png"
plt.savefig(runtime_plot_path, dpi=150)
print(f"Saved plot to: {runtime_plot_path}")

if os.environ.get("DISPLAY"):
    plt.show()
else:
    plt.close()

