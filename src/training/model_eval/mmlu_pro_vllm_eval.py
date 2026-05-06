# MMLU-Pro evaluation script for vLLM OpenAI API endpoint
# Based on https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_api.py
# Sample usage:
# python mmlu_pro_vllm_eval.py --endpoint http://127.0.0.1/v1 --models gemma3:27b,phi4:latest,mistral-small3.1:latest

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from openai import OpenAI
from tqdm import tqdm

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[3]))
from bench.reasoning.router_reason_bench_multi_dataset import extract_answer, build_extra_body_for_model
from bench.reasoning.dataset_factory import DatasetFactory

# Constants
TIMEOUT_SECONDS = 120
MAX_RETRIES = 1  # No retries


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MMLU-Pro benchmark against a vLLM OpenAI endpoint"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM OpenAI API endpoint URL",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="List of model names to evaluate. If not provided, will be fetched from the API.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="List of categories to evaluate. If not provided, all available categories will be used.",
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=5,
        help="Number of questions to sample per category. If not provided, all questions will be used.",
    )
    parser.add_argument(
        "--api-key", type=str, default="", help="API key for vLLM endpoint"
    )
    parser.add_argument(
        "--use-cot", action="store_true", help="Use Chain-of-Thought prompting"
    )
    parser.add_argument(
        "--reasoning-mode",
        type=str,
        choices=["on", "off", "none"],
        default=None,
        help="Reasoning control mode for model-specific extra_body: on/off/none. If omitted, falls back to --use-cot behavior.",
    )
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=1,
        help="Number of concurrent requests to make",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,  # Make it sufficient for the model to answer the question
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for text generation"
    )
    parser.add_argument(
        "--measure-ttft-with-streaming",
        action="store_true",
        help="Use streaming requests to measure TTFT from the first emitted token.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index (0-based) for dataset slicing",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Number of questions to process from start-idx. If not specified, processes all remaining questions.",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="Ending index (exclusive) for dataset slicing. Alternative to --num-questions.",
    )
    return parser.parse_args()


def get_available_models(endpoint: str, api_key: str = "") -> List[str]:
    """Get the list of available models from the vLLM OpenAI API endpoint."""
    client = OpenAI(
        base_url=endpoint,
        api_key=api_key,
    )
    try:
        models = client.models.list()
        return [model.id for model in models.data]
    except Exception as e:
        print(f"Error communicating with vLLM endpoint: {e}")
        # Try direct HTTP request as fallback
        try:
            response = requests.get(f"{endpoint}/models")
            if response.status_code == 200:
                models_data = response.json()
                return [model["id"] for model in models_data.get("data", [])]
            else:
                print(f"Failed to get models list. Status code: {response.status_code}")
        except Exception as e:
            print(f"Failed to get models list via HTTP: {e}")
        return []


def format_cot_prompt(question: str, options: List[str], use_cot: bool = False) -> str:
    """Format the prompt for the model with or without Chain-of-Thought."""
    letter_mapping = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "H",
        8: "I",
        9: "J",
    }
    formatted_options = ""

    for i, option in enumerate(options):
        if option.lower() != "n/a":
            formatted_options += f"{letter_mapping[i]}) {option}\n"

    if use_cot:
        prompt = f"Question: {question}\n\nOptions:\n{formatted_options}\n\nPlease solve this step-by-step, then provide your final answer in the format 'Answer: [letter]'."
    else:
        prompt = f"Question: {question}\n\nOptions:\n{formatted_options}\n\nPlease choose the correct answer from the options above. Provide your answer in the format 'Answer: [letter]'."

    return prompt


def _extract_ttft_seconds(response: Any) -> Optional[float]:
    """Best-effort extraction of TTFT from provider-specific response metadata.

    Returns TTFT in seconds if present, otherwise None.
    """
    try:
        data = response.model_dump() if hasattr(response, "model_dump") else {}
    except Exception:
        data = {}

    def _get_path(obj: Dict[str, Any], path: List[str]) -> Any:
        cur: Any = obj
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        return cur

    candidates = [
        (["ttft"], 1.0),
        (["ttft_seconds"], 1.0),
        (["ttft_ms"], 1e-3),
        (["metrics", "ttft"], 1.0),
        (["metrics", "ttft_seconds"], 1.0),
        (["metrics", "ttft_ms"], 1e-3),
        (["metrics", "time_to_first_token"], 1.0),
        (["metrics", "time_to_first_token_seconds"], 1.0),
        (["metrics", "time_to_first_token_ms"], 1e-3),
        (["usage", "ttft"], 1.0),
        (["usage", "ttft_seconds"], 1.0),
        (["usage", "ttft_ms"], 1e-3),
        (["usage", "time_to_first_token"], 1.0),
        (["usage", "time_to_first_token_seconds"], 1.0),
        (["usage", "time_to_first_token_ms"], 1e-3),
    ]

    for path, scale in candidates:
        value = _get_path(data, path)
        if isinstance(value, (int, float)) and value >= 0:
            return float(value) * scale

    return None


def call_model_with_retry(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    reasoning: Optional[bool] = False,
    measure_ttft_with_streaming: bool = False,
) -> Tuple[str, bool, Optional[int], Optional[int], Optional[int], Optional[float]]:
    """Call the model with retry logic for handling timeouts and errors."""
    extra_body = build_extra_body_for_model(model, reasoning=reasoning) or {}
    print(f"extra_body for model {model}: {extra_body}")
    # Deterministic decoding controls for reproducible evaluation runs.
    extra_body.update({"top_k": -1})

    for attempt in range(MAX_RETRIES):
        try:
            if measure_ttft_with_streaming:
                request_start_time = time.time()
                stream = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=1,
                    presence_penalty=0,
                    frequency_penalty=0,
                    seed=42,
                    extra_body=extra_body,
                    stream=True,
                    stream_options={"include_usage": True},
                )

                first_token_time = None
                prompt_tokens = None
                completion_tokens = None
                total_tokens = None
                content_parts = []
                reasoning_parts = []

                for chunk in stream:
                    usage = getattr(chunk, "usage", None)
                    if usage is not None:
                        prompt_tokens = getattr(usage, "prompt_tokens", prompt_tokens)
                        completion_tokens = getattr(
                            usage, "completion_tokens", completion_tokens
                        )
                        total_tokens = getattr(usage, "total_tokens", total_tokens)

                    choices = getattr(chunk, "choices", None) or []
                    if not choices:
                        continue

                    delta = getattr(choices[0], "delta", None)
                    if delta is None:
                        continue

                    content_delta = getattr(delta, "content", None)
                    if content_delta:
                        if first_token_time is None:
                            first_token_time = time.time()
                        content_parts.append(content_delta)

                    reasoning_delta = getattr(delta, "reasoning_content", None)
                    if not reasoning_delta:
                        reasoning_delta = getattr(delta, "reasoning", None)
                    if reasoning_delta:
                        if first_token_time is None:
                            first_token_time = time.time()
                        reasoning_parts.append(reasoning_delta)

                text = "".join(content_parts) or "".join(reasoning_parts)
                ttft_seconds = (
                    first_token_time - request_start_time
                    if first_token_time is not None
                    else None
                )
                return (
                    text,
                    True,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    ttft_seconds,
                )

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1,
                presence_penalty=0,
                frequency_penalty=0,
                seed=42,
                extra_body=extra_body,
            )
            message = response.choices[0].message
            text = (
                message.content
                or getattr(message, "reasoning_content", None)
                or getattr(message, "reasoning", None)
                or ""
            )
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            completion_tokens = (
                getattr(usage, "completion_tokens", None) if usage else None
            )
            total_tokens = getattr(usage, "total_tokens", None) if usage else None
            ttft_seconds = _extract_ttft_seconds(response)
            return (
                text,
                True,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                ttft_seconds,
            )
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = 2**attempt  # Exponential backoff
                print(
                    f"Error calling model (attempt {attempt+1}/{MAX_RETRIES}), retrying in {delay}s: {e}"
                )
                time.sleep(delay)
            else:
                print(f"Failed to call model after {MAX_RETRIES} attempts: {e}")
                return "ERROR", False, None, None, None, None


def process_question(
    client: OpenAI,
    model: str,
    question_data,
    reasoning: Optional[bool],
    max_tokens: int,
    temperature: float,
    measure_ttft_with_streaming: bool = False,
    dataset=None,
) -> Dict[str, Any]:
    """Process a single question and return the results."""
    question = question_data.question
    options = question_data.options
    correct_answer = question_data.correct_answer

    # Keep prompt style plain; reasoning toggles reasoning flags in extra_body.
    prompt = dataset.format_prompt(question_data, "plain")

    start_time = time.time()
    response_text, success, prompt_tokens, completion_tokens, total_tokens, ttft_seconds = call_model_with_retry(
        client,
        model,
        prompt,
        max_tokens,
        temperature,
        reasoning,
        measure_ttft_with_streaming,
    )
    end_time = time.time()
    response_time = end_time - start_time

    # TPOT (seconds per output token): completion latency / completion tokens.
    # When TTFT is available, completion latency = full response time - TTFT.
    # Otherwise we fall back to an end-to-end latency-per-token approximation.
    completion_latency_seconds = (
        max(response_time - ttft_seconds, 0.0)
        if ttft_seconds is not None
        else None
    )
    tpot_seconds = (
        completion_latency_seconds / completion_tokens
        if completion_latency_seconds is not None
        and completion_tokens is not None
        and completion_tokens > 0
        else response_time / completion_tokens
        if completion_tokens is not None and completion_tokens > 0
        else None
    )
    tpot_is_approximate = ttft_seconds is None and tpot_seconds is not None

    predicted_answer = extract_answer(response_text, question_data) if success else None
    if predicted_answer is not None:
        predicted_answer = str(predicted_answer).strip().upper()
        valid_letters = {
            chr(ord("A") + i)
            for i, option in enumerate(options)
            if str(option).lower() != "n/a"
        }
        if predicted_answer not in valid_letters:
            print(
                f"Discarding invalid extracted answer '{predicted_answer}' (valid: {sorted(valid_letters)})"
            )
            predicted_answer = None
    is_correct = (predicted_answer == correct_answer) if predicted_answer else False
    print(f"Predicted answer: {predicted_answer}, Correct answer: {correct_answer}")

    # Append full per-question details to final log, including token usage.
    with open("mmlu_pro_vllm_eval.txt", "a") as f:
        f.write(f"Category: {question_data.category}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Correct answer: {correct_answer}\n")
        f.write(f"Model response: {response_text}\n")
        f.write(f"Predicted answer: {predicted_answer}\n")
        f.write(f"Prompt tokens: {prompt_tokens}\n")
        f.write(f"Completion tokens: {completion_tokens}\n")
        f.write(f"Total tokens: {total_tokens}\n")
        f.write(f"TTFT (s): {ttft_seconds}\n")
        f.write(f"Completion latency (s): {completion_latency_seconds}\n")
        f.write(f"TPOT (s/token): {tpot_seconds}\n\n")
        f.write(f"TPOT approximate: {tpot_is_approximate}\n\n")

    return {
        "question_id": question_data.question_id,
        "question": question,
        "options": options,
        "correct_answer": correct_answer,
        "model_response": response_text,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "category": question_data.category,
        "response_time": response_time,
        "success": success,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "ttft_seconds": ttft_seconds,
        "completion_latency_seconds": completion_latency_seconds,
        "tpot_seconds": tpot_seconds,
        "tpot_is_approximate": tpot_is_approximate,
    }


def evaluate_model(
    df: list,
    model: str,
    endpoint: str,
    api_key: str,
    reasoning: Optional[bool],
    concurrent_requests: int,
    max_tokens: int,
    temperature: float,
    measure_ttft_with_streaming: bool = False,
    dataset=None,
) -> pd.DataFrame:
    """Evaluate a model on the MMLU-Pro dataset."""
    client = OpenAI(base_url=endpoint, api_key=api_key if api_key else "dummy")
    print(f"Using model: {model}, endpoint: {endpoint}, api_key: {api_key}")
    results = []

    questions_data = df

    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = []
        for i, question_data in enumerate(questions_data):
            future = executor.submit(
                process_question,
                client,
                model,
                question_data,
                reasoning,
                max_tokens,
                temperature,
                measure_ttft_with_streaming,
                dataset,
            )
            futures.append(future)

        for future in tqdm(futures, total=len(futures), desc=f"Evaluating {model}"):
            result = future.result()
            results.append(result)

    results_df = pd.DataFrame(results)
    return results_df


def analyze_results(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze the results and compute statistics."""
    # Skip failed requests in the analysis
    valid_results = results_df[results_df["success"]]

    # Overall accuracy
    overall_accuracy = (
        valid_results["is_correct"].mean() if not valid_results.empty else 0.0
    )

    # Accuracy per category
    category_accuracy = {}
    for category in valid_results["category"].unique():
        category_df = valid_results[valid_results["category"] == category]
        category_accuracy[category] = category_df["is_correct"].mean()

    # Compute average response time
    avg_response_time = (
        valid_results["response_time"].mean() if not valid_results.empty else 0.0
    )

    # TTFT/TPOT summaries. TTFT may be unavailable depending on backend response fields.
    ttft_series = valid_results["ttft_seconds"].dropna()
    exact_tpot_series = valid_results[
        (valid_results["tpot_is_approximate"] == False) & valid_results["tpot_seconds"].notna()
    ]["tpot_seconds"]
    approx_tpot_series = valid_results[
        (valid_results["tpot_is_approximate"] == True) & valid_results["tpot_seconds"].notna()
    ]["tpot_seconds"]

    avg_ttft = float(ttft_series.mean()) if not ttft_series.empty else None
    p95_ttft = float(ttft_series.quantile(0.95)) if not ttft_series.empty else None
    avg_tpot = float(exact_tpot_series.mean()) if not exact_tpot_series.empty else None
    p95_tpot = (
        float(exact_tpot_series.quantile(0.95)) if not exact_tpot_series.empty else None
    )
    avg_tpot_approx = (
        float(approx_tpot_series.mean()) if not approx_tpot_series.empty else None
    )
    p95_tpot_approx = (
        float(approx_tpot_series.quantile(0.95))
        if not approx_tpot_series.empty
        else None
    )

    return {
        "overall_accuracy": overall_accuracy,
        "category_accuracy": category_accuracy,
        "avg_response_time": avg_response_time,
        "avg_ttft": avg_ttft,
        "p95_ttft": p95_ttft,
        "avg_tpot": avg_tpot,
        "p95_tpot": p95_tpot,
        "avg_tpot_approx": avg_tpot_approx,
        "p95_tpot_approx": p95_tpot_approx,
        "ttft_samples": int(ttft_series.shape[0]),
        "tpot_samples": int(exact_tpot_series.shape[0]),
        "tpot_approx_samples": int(approx_tpot_series.shape[0]),
        "total_questions": len(results_df),
        "successful_queries": len(valid_results),
        "failed_queries": len(results_df) - len(valid_results),
    }


def save_results(
    results_df: pd.DataFrame,
    analysis: Dict[str, Any],
    model: str,
    output_dir: str,
    reasoning: Optional[bool],
):
    """Save the results and analysis to files."""
    model_name = model.replace("/", "_")
    reasoning_label = (
        "Reasoning-On"
        if reasoning is True
        else "Reasoning-Off"
        if reasoning is False
        else "Reasoning-None"
    )
    cot_suffix = (
        "cot"
        if reasoning is True
        else "direct"
        if reasoning is False
        else "none"
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, f"{model_name}_{cot_suffix}")
    os.makedirs(model_dir, exist_ok=True)

    # Save detailed results
    results_df.to_csv(os.path.join(model_dir, "detailed_results.csv"), index=False)

    # Save analysis
    with open(os.path.join(model_dir, "analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    # Save summary
    summary = {
        "model": model,
        "approach": reasoning_label,
        "overall_accuracy": analysis["overall_accuracy"],
        "total_questions": analysis["total_questions"],
        "successful_queries": analysis["successful_queries"],
        "failed_queries": analysis["failed_queries"],
        "avg_response_time": analysis["avg_response_time"],
        "avg_ttft": analysis["avg_ttft"],
        "p95_ttft": analysis["p95_ttft"],
        "avg_tpot": analysis["avg_tpot"],
        "p95_tpot": analysis["p95_tpot"],
        "avg_tpot_approx": analysis["avg_tpot_approx"],
        "p95_tpot_approx": analysis["p95_tpot_approx"],
        "ttft_samples": analysis["ttft_samples"],
        "tpot_samples": analysis["tpot_samples"],
        "tpot_approx_samples": analysis["tpot_approx_samples"],
    }

    with open(os.path.join(model_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print(f"Model: {model}")
    print(f"Approach: {reasoning_label}")
    print(f"Overall Accuracy: {analysis['overall_accuracy']:.4f}")
    print(f"Total Questions: {analysis['total_questions']}")
    print(f"Successful Queries: {analysis['successful_queries']}")
    print(f"Failed Queries: {analysis['failed_queries']}")
    print(f"Average Response Time: {analysis['avg_response_time']:.2f}s")
    if analysis["avg_ttft"] is not None:
        print(
            f"TTFT: avg={analysis['avg_ttft']:.4f}s, p95={analysis['p95_ttft']:.4f}s (n={analysis['ttft_samples']})"
        )
    else:
        print("TTFT: not available from response metadata")
    if analysis["avg_tpot"] is not None:
        print(
            f"TPOT: avg={analysis['avg_tpot']:.6f}s/token, p95={analysis['p95_tpot']:.6f}s/token (n={analysis['tpot_samples']})"
        )
    elif analysis["avg_tpot_approx"] is not None:
        print(
            f"TPOT (approx): avg={analysis['avg_tpot_approx']:.6f}s/token, p95={analysis['p95_tpot_approx']:.6f}s/token (n={analysis['tpot_approx_samples']})"
        )
    else:
        print("TPOT: not available (missing completion token counts)")
    print("=" * 50 + "\n")

    # Print category accuracy
    print("Category Accuracy:")
    for category, accuracy in sorted(
        analysis["category_accuracy"].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {category}: {accuracy:.4f}")
    print()


def main():
    args = parse_args()

    if args.reasoning_mode is not None:
        reasoning = (
            True if args.reasoning_mode == "on" else False if args.reasoning_mode == "off" else None
        )
    else:
        # Backward compatible behavior: --use-cot => reasoning on, otherwise off.
        reasoning = True if args.use_cot else False

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Get available models if not specified
    if not args.models:
        print("Fetching available models from vLLM endpoint...")
        models = get_available_models(args.endpoint, args.api_key)
        if not models:
            print(
                "Could not retrieve models from the endpoint. Please specify models using --models."
            )
            return
        args.models = models

    if args.models and len(args.models) == 1 and "," in args.models[0]:
        args.models = args.models[0].split(",")

    print(f"Models to evaluate: {args.models}")

    # Load dataset
    print("Loading MMLU-Pro dataset...")
    dataset = DatasetFactory.create_dataset("mmlu-pro")
    questions, dataset_info = dataset.load_dataset(
        categories=args.categories,
        samples_per_category=args.samples_per_category,
        seed=args.seed,
    )
    df = questions

    print(f"Available categories: {', '.join(dataset_info.categories)}")
    print(
        f"Dataset loaded: {len(questions)} questions across {len(dataset_info.categories)} categories"
    )

    # Apply dataset slicing if specified
    if args.start_idx > 0 or args.num_questions or args.end_idx:
        start_idx = args.start_idx
        if args.end_idx is not None:
            end_idx = args.end_idx
        elif args.num_questions is not None:
            end_idx = start_idx + args.num_questions
        else:
            end_idx = len(questions)
        
        # Validate indices
        end_idx = min(end_idx, len(questions))
        start_idx = max(0, start_idx)
        
        if start_idx >= end_idx:
            print(f"Error: start_idx ({start_idx}) must be less than end_idx ({end_idx})")
            return
        
        df = df[start_idx:end_idx]
        print(f"Dataset sliced: using questions [{start_idx}:{end_idx}] ({len(df)} questions)")

    # Evaluate each model
    for model in args.models:
        print(f"\nEvaluating model: {model}")
        results_df = evaluate_model(
            df=df,
            model=model,
            endpoint=args.endpoint,
            api_key=args.api_key,
            reasoning=reasoning,
            concurrent_requests=args.concurrent_requests,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            measure_ttft_with_streaming=args.measure_ttft_with_streaming,
            dataset=dataset,
        )

        # Analyze and save results
        analysis = analyze_results(results_df)
        save_results(
            results_df=results_df,
            analysis=analysis,
            model=model,
            output_dir=args.output_dir,
            reasoning=reasoning,
        )


if __name__ == "__main__":
    main()
