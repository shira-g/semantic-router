"""
Multi-Dataset Reasoning Benchmark

A comprehensive evaluation framework for comparing semantic router performance
against direct vLLM inference across various reasoning datasets.

Features:
- Dataset-agnostic architecture supporting MMLU, ARC, GPQA, TruthfulQA, CommonsenseQA, HellaSwag
- Optimized token limits per dataset complexity
- Multiple reasoning modes (NR, XC, NR_REASONING)
- Structured response parsing with robust answer extraction
- Comprehensive metrics and visualization
"""

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from openai import OpenAI
from tqdm import tqdm

from .dataset_factory import DatasetFactory, list_available_datasets
from .dataset_interface import DatasetInfo, Question, questions_to_dataframe

# Robust answer extraction patterns for structured response parsing
ANSWER_PATTERN_PRIMARY = re.compile(
    r"(?:answer\s*:?\s+)([A-Z])(?:\s|[.!?)]|$)", re.IGNORECASE
)
ANSWER_PATTERN_FINAL = re.compile(
    r"(?:final\s*answer\s*:?\s+)([A-Z])(?:\s|[.!?)]|$)", re.IGNORECASE
)
ANSWER_PATTERN_CONCLUSION = re.compile(
    r"(?:therefore|thus|so).*?(?:answer\s+is\s+|is\s+)([A-Z])(?:\s|[.!?)]|$)",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Dataset Reasoning Benchmark: Comprehensive evaluation framework for semantic router vs direct vLLM"
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        default="mmlu",
        help="Dataset to evaluate on. Use --list-datasets to see available options.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets and exit",
    )

    # Semantic router configuration
    parser.add_argument(
        "--router-endpoint",
        type=str,
        default=os.environ.get("ROUTER_ENDPOINT", "http://127.0.0.1:8801/v1"),
        help="Semantic router endpoint URL",
    )
    parser.add_argument(
        "--router-api-key",
        type=str,
        default=os.environ.get(
            "ROUTER_API_KEY", os.environ.get("OPENAI_API_KEY", "1234")
        ),
        help="API key for router endpoint",
    )
    parser.add_argument(
        "--router-models",
        type=str,
        nargs="+",
        default=["auto"],
        help="Router models to evaluate (default: auto).",
    )

    # Direct vLLM configuration
    parser.add_argument(
        "--vllm-endpoint",
        type=str,
        default=os.environ.get("VLLM_ENDPOINT", ""),
        help="Direct vLLM endpoint URL",
    )
    parser.add_argument(
        "--vllm-api-key",
        type=str,
        default=os.environ.get("VLLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        help="API key for vLLM endpoint",
    )
    parser.add_argument(
        "--vllm-models",
        type=str,
        nargs="+",
        default=[],
        help="Direct vLLM models to evaluate (leave empty to fetch from endpoint).",
    )

    # vLLM reasoning modes
    parser.add_argument(
        "--vllm-exec-modes",
        type=str,
        nargs="+",
        default=["NR", "XC"],
        help="vLLM reasoning modes: NR (neutral), XC (chain-of-thought), NR_REASONING (reasoning-enabled)",
    )
    parser.add_argument(
        "--run-router",
        action="store_true",
        help="Evaluate semantic router performance",
    )
    parser.add_argument(
        "--run-vllm",
        action="store_true",
        help="Evaluate direct vLLM performance across multiple reasoning modes",
    )

    # Dataset filtering options
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

    # Execution options
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=1,
        help="Number of concurrent requests to make",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/reasonbench",
        help="Directory to save results",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate (default: dataset-optimal)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for text generation",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling parameter (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling parameter (default: -1, disabled)",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="Presence penalty (default: 0.0)",
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.0,
        help="Frequency penalty (default: 0.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility and model request seed",
    )
    parser.add_argument(
        "--ar-extra-body",
        type=str,
        default="",
        help=(
            'JSON string passed as extra_body for AR mode (e.g., \'{"reasoning":{"effort":"medium"}}\'). '
            "If empty, AR modes are disabled."
        ),
    )
    parser.add_argument(
        "--debug-log-request-response",
        action="store_true",
        help="Include full request and response fields in output CSV logs (debug only).",
    )
    parser.add_argument(
        "--debug-print-request-response",
        action="store_true",
        help="Print full request and response per example to stdout (debug only).",
    )
    return parser.parse_args()


def get_dataset_optimal_tokens(dataset_info, model_name=None):
    """
    Determine optimal token limit based on dataset complexity, reasoning requirements, and model capabilities.

    Token limits are optimized for structured response generation while maintaining
    efficiency across different reasoning complexity levels and model architectures.

    Args:
        dataset_info: Dataset information object
        model_name: Model identifier (e.g., "openai/gpt-oss-20b", "Qwen/Qwen3-30B-A3B")
    """
    dataset_name = dataset_info.name.lower()
    difficulty = dataset_info.difficulty_level.lower()

    # Determine model type and capabilities
    model_multiplier = 1.5
    if model_name:
        model_lower = model_name.lower()
        print(f"  🔍 Model detection: '{model_name}' -> '{model_lower}'")
        if "qwen" in model_lower:
            # Qwen models are more efficient and can handle longer contexts
            model_multiplier = 1.5
            print(f"  ✅ Qwen model detected, using multiplier: {model_multiplier}")
        elif "deepseek" in model_lower:
            # DeepSeek models (e.g., V3.1) are capable and can handle longer contexts
            model_multiplier = 1.5
            print(f"  ✅ DeepSeek model detected, using multiplier: {model_multiplier}")
        elif "gpt-oss" in model_lower:
            # GPT-OSS models use baseline token limits
            model_multiplier = 1.5
            print(f"  ✅ GPT-OSS model detected, using multiplier: {model_multiplier}")
        else:
            print(
                f"  ⚠️  Unknown model type, using baseline multiplier: {model_multiplier}"
            )
        # Default to baseline for unknown models

    # Base token limits per dataset (optimized for reasoning tasks with generous headroom)
    base_dataset_tokens = {
        # Proven optimal datasets
        "gpqa": 4000,  # Graduate-level scientific reasoning (proven optimal from results)
        "mmlu": 4000,  # Academic knowledge (proven optimal from results)
        "truthfulqa": 2500,  # Misconception analysis (proven adequate from results)
        # Mathematical reasoning datasets
        # "math": 6000,  # Competition mathematics - DISABLED: dataset not available
        "gsm8k": 2500,  # Elementary math word problems - simpler than competition math
        "aqua-rat": 3000,  # Algebraic word problems with rationales
        # Multi-step reasoning datasets
        "drop": 4000,  # Reading comprehension with discrete reasoning - complex passages
        "strategyqa": 3500,  # Multi-step implicit reasoning - requires detailed thinking
        # Scientific reasoning datasets
        "sciq": 2000,  # Science questions - moderate complexity
        "openbookqa": 2500,  # Elementary science with fact reasoning
        # Other datasets
        "hellaswag": 2000,  # Natural continuation reasoning
        "arc": 2000,  # Elementary/middle school science
        "arc-challenge": 3000,  # Harder ARC questions
        "commonsenseqa": 2500,  # Common sense reasoning
    }

    # Find matching dataset and apply model multiplier
    base_tokens = None
    for dataset_key, tokens in base_dataset_tokens.items():
        if dataset_key in dataset_name:
            base_tokens = tokens
            break

    # Fallback to difficulty-based tokens if dataset not found
    if base_tokens is None:
        difficulty_tokens = {"graduate": 300, "hard": 300, "moderate": 200, "easy": 150}
        base_tokens = difficulty_tokens.get(difficulty, 200)

    # Special case: Qwen3 models need higher tokens for complex reasoning datasets
    if model_name and "qwen" in model_name.lower():
        if "mmlu" in dataset_name or "gpqa" in dataset_name:
            final_tokens = 10240
            dataset_type = "MMLU" if "mmlu" in dataset_name else "GPQA"
            print(
                f"  🎯 Special case: Qwen3 + {dataset_type} = {final_tokens} tokens (fixed requirement)"
            )
            return final_tokens
        # elif "math" in dataset_name:  # DISABLED: dataset not available
        #     final_tokens = 8000  # Competition math needs extensive proofs
        #     print(f"  🎯 Special case: Qwen3 + MATH = {final_tokens} tokens (competition math requirement)")
        #     return final_tokens

    # Apply model-specific multiplier and round to nearest 50
    final_tokens = int(base_tokens * model_multiplier)
    final_tokens = ((final_tokens + 25) // 50) * 50  # Round to nearest 50

    print(
        f"  🧮 Token calculation: {base_tokens} × {model_multiplier} = {int(base_tokens * model_multiplier)} → {final_tokens} (rounded)"
    )

    return final_tokens


def get_available_models(endpoint: str, api_key: str = "") -> List[str]:
    """Get available models from an endpoint."""
    client = OpenAI(base_url=endpoint, api_key=api_key or None, timeout=300.0)
    try:
        models = client.models.list()
        return [m.id for m in models.data]
    except Exception as e:
        print(f"Error communicating with endpoint to list models: {e}")
        return []


def extract_answer(response: Any, question: Optional[Question] = None) -> Optional[str]:
    """Extract answer from model response based on question format."""
    # Normalize non-string responses into a string to be robust to providers
    # that return structured content (e.g., lists of parts or dicts).
    if response is None:
        return None

    if not isinstance(response, str):
        try:
            # Handle list-of-parts shapes
            if isinstance(response, list):
                parts: List[str] = []
                for part in response:
                    if isinstance(part, dict):
                        if "text" in part and isinstance(part["text"], str):
                            parts.append(part["text"])
                        elif "content" in part and isinstance(part["content"], str):
                            parts.append(part["content"])
                        else:
                            parts.append(str(part))
                    else:
                        parts.append(str(part))
                response = "\n".join(parts)
            # Handle dict shapes
            elif isinstance(response, dict):
                for key in ("content", "text", "reasoning_content"):
                    val = response.get(key) if isinstance(response, dict) else None
                    if isinstance(val, str) and val:
                        response = val
                        break
                else:
                    # Fallback to JSON stringification
                    response = json.dumps(response, ensure_ascii=False)
            else:
                response = str(response)
        except Exception:
            response = str(response)

    # First, try to extract structured answer format "ANSWER: [value]"
    structured_answer = extract_structured_answer(response)
    if structured_answer:
        return structured_answer

    # Determine answer format based on question type
    if question and hasattr(question, "options") and question.options:
        if len(question.options) == 2 and set(question.options) == {"Yes", "No"}:
            # Binary Yes/No questions (StrategyQA)
            return extract_binary_answer(response)
        else:
            # Multiple choice questions (GPQA, MMLU, etc.)
            return extract_multiple_choice_answer(response)
    else:
        # Free-form questions (GSM8K, DROP, etc.)
        return extract_free_form_answer(response)


def extract_structured_answer(response: str) -> Optional[str]:
    """Extract answer from structured 'ANSWER: [value]' format."""
    # Look for "ANSWER: [value]" pattern (case insensitive)
    pattern = re.compile(r"ANSWER:\s*(.+?)(?:\n|$)", re.IGNORECASE)
    match = pattern.search(response)
    if match:
        answer = match.group(1).strip()
        # Clean up common trailing punctuation
        answer = re.sub(r"[.!?]+$", "", answer)
        return answer
    return None


def extract_multiple_choice_answer(response: str) -> Optional[str]:
    """Extract multiple choice answer (A, B, C, D, etc.)."""
    # Try multiple extraction patterns in order of preference
    patterns = [ANSWER_PATTERN_PRIMARY, ANSWER_PATTERN_FINAL, ANSWER_PATTERN_CONCLUSION]

    for pattern in patterns:
        match = pattern.search(response)
        if match:
            return match.group(1).upper()

    # Additional patterns for common answer formats
    additional_patterns = [
        r"(?:correct\s+answer\s+is\s+)([A-Z])",  # "correct answer is E"
        r"(?:option\s+)([A-Z])",  # "option E"
        r"(?:choice\s+)([A-Z])",  # "choice E"
        r"([A-Z])\)",  # "E)" format
        r"([A-Z])\s*[.!]?\s*$",  # Letter at end of line
    ]

    for pattern in additional_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Fallback 1: Look for standalone letters at end of response
    lines = response.strip().split("\n")
    for line in reversed(lines[-3:]):  # Check last 3 lines
        line = line.strip()
        if len(line) == 1 and line.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return line.upper()

    # Fallback 2: Look for letters in specific contexts (more targeted)
    # Check for patterns like "is E" or "answer E" in last few lines
    for line in reversed(lines[-3:]):
        line = line.strip()
        # Look for letter after common words
        context_match = re.search(
            r"(?:is|answer|option|choice)\s+([A-Z])(?:\s|[.!?]|$)", line, re.IGNORECASE
        )
        if context_match:
            return context_match.group(1).upper()

    # Final fallback: Find last letter that appears to be an answer (not in middle of words)
    # Only consider letters that are standalone or followed by punctuation
    for match in re.finditer(r"\b([A-Z])(?:\s|[.!?)]|$)", response):
        letter = match.group(1).upper()
        if letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return letter  # Return the last match found

    return None


def extract_binary_answer(response: str) -> Optional[str]:
    """Extract Yes/No answer from response."""
    response_lower = response.lower()

    # Look for explicit yes/no patterns
    yes_patterns = [r"\byes\b", r"\btrue\b", r"\bcorrect\b", r"\baffirmative\b"]
    no_patterns = [r"\bno\b", r"\bfalse\b", r"\bincorrect\b", r"\bnegative\b"]

    # Check last few lines first (most likely to contain final answer)
    lines = response.strip().split("\n")
    for line in reversed(lines[-3:]):
        line_lower = line.lower().strip()

        for pattern in yes_patterns:
            if re.search(pattern, line_lower):
                return "Yes"

        for pattern in no_patterns:
            if re.search(pattern, line_lower):
                return "No"

    # Fallback: check entire response
    for pattern in yes_patterns:
        if re.search(pattern, response_lower):
            return "Yes"

    for pattern in no_patterns:
        if re.search(pattern, response_lower):
            return "No"

    return None


def extract_free_form_answer(response: str) -> Optional[str]:
    """Extract free-form answer (numbers, text, etc.)."""
    # For numerical answers, look for numbers with improved patterns
    number_patterns = [
        r"(?:answer\s*:?\s*)([0-9,.-]+)",  # "Answer: 42" or "Answer 42"
        r"####\s*([0-9,.-]+)",  # GSM8K format "#### 42"
        r"\$([0-9,.-]+)",  # Money format "$42"
        r"([0-9,.-]+)\s*(?:dollars?|cents?|%|percent)",  # "42 dollars"
        r"(?:is\s+)([0-9,.-]+)",  # "is 42" or "is 68.5"
        r"(?:was\s+)([0-9,.-]+)",  # "was 42"
        r"(?:were\s+)([0-9,.-]+)",  # "were 42"
        r"([0-9,.-]+)(?:\s+(?:people|units|items|years|days|months|miles|kilometers|percent|%|dollars?|cents?))",  # "68.5 people"
    ]

    # Check last few lines first (most likely to contain final answer)
    lines = response.strip().split("\n")
    for line in reversed(lines[-3:]):
        line = line.strip()

        for pattern in number_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1).replace(",", "")  # Remove commas from numbers

    # Fallback: check entire response for numbers
    for pattern in number_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).replace(",", "")

    # For non-numerical free-form answers (like "germans", "Centenary Medal")
    # Look for explicit answer patterns first
    text_patterns = [
        r"(?:answer\s*:?\s*)([a-zA-Z][a-zA-Z0-9\s-]+?)(?:\s*[.!?]|$)",  # "Answer: germans" or "Answer: Centenary Medal"
        r"(?:is\s+)([a-zA-Z][a-zA-Z0-9\s-]+?)(?:\s*[.!?]|$)",  # "is germans"
        r"(?:was\s+)([a-zA-Z][a-zA-Z0-9\s-]+?)(?:\s*[.!?]|$)",  # "was Centenary Medal"
        r"(?:were\s+)([a-zA-Z][a-zA-Z0-9\s-]+?)(?:\s*[.!?]|$)",  # "were germans"
        r"(?:awarded\s+(?:him\s+)?(?:the\s+)?)([A-Z][a-zA-Z0-9\s-]+?)(?:\s*[.!?]|$)",  # "awarded the Centenary Medal"
        r"(?:received\s+(?:the\s+)?)([A-Z][a-zA-Z0-9\s-]+?)(?:\s*[.!?]|$)",  # "received the Centenary Medal"
        r"(?:called\s+)([a-zA-Z][a-zA-Z0-9\s-]+?)(?:\s*[.!?]|$)",  # "called germans"
        r"(?:named\s+)([a-zA-Z][a-zA-Z0-9\s-]+?)(?:\s*[.!?]|$)",  # "named Centenary Medal"
    ]

    # Check last few lines for text answers
    for line in reversed(lines[-3:]):
        line = line.strip()

        for pattern in text_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Clean up common suffixes but preserve important words
                answer = re.sub(
                    r"\s+(?:in\s+\d+|for\s+service).*$", "", answer, flags=re.IGNORECASE
                )
                # Limit to reasonable length (1-4 words for most DROP answers)
                words = answer.split()
                if len(words) <= 4:
                    return answer
                else:
                    return " ".join(words[:2])  # Take first 2 words for long matches

    # Final fallback: extract last meaningful line
    for line in reversed(lines[-3:]):
        line = line.strip()
        if line and not line.startswith(
            (
                "Question:",
                "Answer:",
                "Therefore",
                "So",
                "Thus",
                "Based on",
                "Looking at",
            )
        ):
            # Remove common prefixes and return clean answer
            line = re.sub(
                r"^(?:the\s+)?(?:answer\s+is\s+)?", "", line, flags=re.IGNORECASE
            )
            # Take first few words if it's a long sentence
            words = line.split()
            if len(words) > 5:
                return " ".join(words[:3])  # Take first 3 words
            return line.strip()

    return None


def compare_free_form_answers(predicted: str, correct: str) -> bool:
    """Compare free-form answers with normalization."""
    if not predicted or not correct:
        return False

    # Normalize both answers
    predicted_norm = normalize_answer(predicted)
    correct_norm = normalize_answer(correct)

    # Direct match
    if predicted_norm == correct_norm:
        return True

    # For numerical answers, try parsing as numbers
    try:
        pred_num = float(predicted_norm.replace(",", ""))
        correct_num = float(correct_norm.replace(",", ""))
        # Allow small floating point differences
        return abs(pred_num - correct_num) < 1e-6
    except (ValueError, AttributeError):
        pass

    # For text answers, check if predicted contains correct or vice versa
    if len(predicted_norm) > 3 and len(correct_norm) > 3:
        return predicted_norm in correct_norm or correct_norm in predicted_norm

    return False


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if not isinstance(answer, str):
        answer = str(answer)

    # Convert to lowercase and strip
    answer = answer.lower().strip()

    # Remove common punctuation and extra spaces
    answer = re.sub(r"[^\w\s.-]", "", answer)
    answer = re.sub(r"\s+", " ", answer).strip()

    # Remove common prefixes
    prefixes = [
        "the answer is",
        "answer:",
        "the answer:",
        "answer is",
        "final answer:",
        "therefore",
    ]
    for prefix in prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix) :].strip()

    return answer


def call_model(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
    frequency_penalty: float,
    seed: int,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Tuple[
    str,
    bool,
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[Dict[str, Any]],
]:
    """Call model with given parameters."""

    def _normalize_header_value(v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    def _first_header(headers_obj: Any, keys: List[str]) -> Optional[str]:
        """Read first non-empty header value from a case-insensitive header object."""
        if headers_obj is None:
            return None

        # Try direct lookups first (works for httpx Headers / dict-like objects).
        for key in keys:
            try:
                val = headers_obj.get(key)
            except Exception:
                val = None
            norm = _normalize_header_value(val)
            if norm:
                return norm

        # Fallback: iterate all headers and compare lowercase keys.
        items: List[Tuple[str, Any]] = []
        try:
            items = list(headers_obj.items())
        except Exception:
            items = []

        if items:
            lowered = {str(k).lower(): v for k, v in items}
            for key in keys:
                norm = _normalize_header_value(lowered.get(key.lower()))
                if norm:
                    return norm

        return None

    def _extract_router_headers(raw_obj: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract category/decision/model from any known response wrapper shape."""
        if raw_obj is None:
            return None, None, None

        header_candidates = [
            getattr(raw_obj, "headers", None),
            getattr(getattr(raw_obj, "http_response", None), "headers", None),
            getattr(getattr(raw_obj, "response", None), "headers", None),
            getattr(getattr(raw_obj, "_response", None), "headers", None),
        ]

        selected_category_local = None
        selected_decision_local = None
        selected_model_local = None

        for headers_obj in header_candidates:
            if headers_obj is None:
                continue

            if not selected_category_local:
                selected_category_local = _first_header(
                    headers_obj,
                    [
                        "x-vsr-selected-category",
                        "x-selected-category",
                        "x-vsr-matched-domains",
                        "x-vsr-domain",
                    ],
                )
            if not selected_decision_local:
                selected_decision_local = _first_header(
                    headers_obj,
                    ["x-vsr-selected-decision", "x-selected-decision"],
                )
            if not selected_model_local:
                selected_model_local = _first_header(
                    headers_obj,
                    ["x-vsr-selected-model", "x-selected-model"],
                )

            if selected_category_local and selected_decision_local and selected_model_local:
                break

        # If matched domains are present, normalize to a single domain label.
        if selected_category_local and ("," in selected_category_local or "|" in selected_category_local):
            selected_category_local = re.split(r"[,|]", selected_category_local)[0].strip()

        return selected_category_local, selected_decision_local, selected_model_local

    def _extract_http_response_details(raw_obj: Any) -> Optional[Dict[str, Any]]:
        """Extract HTTP response details (status, headers, body) for debugging."""
        if raw_obj is None:
            return None

        details = {}

        # Try to get status code from various wrapper shapes
        status_code = None
        for attr in ["status_code", "http_response.status_code", "response.status_code", "_response.status_code"]:
            if "." in attr:
                parts = attr.split(".")
                obj = raw_obj
                for part in parts:
                    obj = getattr(obj, part, None)
                    if obj is None:
                        break
                if obj is not None:
                    status_code = obj
                    break
            else:
                status_code = getattr(raw_obj, attr, None)
                if status_code is not None:
                    break

        if status_code is not None:
            details["status_code"] = status_code

        # Extract headers from various wrapper shapes
        headers_dict = {}
        header_candidates = [
            getattr(raw_obj, "headers", None),
            getattr(getattr(raw_obj, "http_response", None), "headers", None),
            getattr(getattr(raw_obj, "response", None), "headers", None),
            getattr(getattr(raw_obj, "_response", None), "headers", None),
        ]

        for headers_obj in header_candidates:
            if headers_obj is None:
                continue
            try:
                if hasattr(headers_obj, "items"):
                    headers_dict = dict(headers_obj.items())
                else:
                    headers_dict = dict(headers_obj)
                if headers_dict:
                    break
            except Exception:
                pass

        if headers_dict:
            details["headers"] = headers_dict

        # Try to get response body/content
        content = None
        for attr in ["content", "text", "body", "_content"]:
            content = getattr(raw_obj, attr, None)
            if content is not None:
                break

        if content is not None:
            # Truncate very large responses for logging
            if isinstance(content, str) and len(content) > 5000:
                details["body"] = content[:5000] + f"\n... (truncated, total {len(content)} chars)"
            else:
                details["body"] = content

        return details if details else None


    selected_category = None
    selected_decision = None
    selected_model_header = None
    raw_response = None
    request_extra_body = dict(extra_body) if extra_body else {}
    if top_k >= 0:
        request_extra_body["top_k"] = top_k

    try:
        # Prefer raw response mode to read router decision headers; fallback to
        # regular completion API if raw response is unavailable.
        try:
            raw_response = client.chat.completions.with_raw_response.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                seed=seed,
                extra_body=request_extra_body if request_extra_body else None,
            )
            response = raw_response.parse()
            (
                selected_category,
                selected_decision,
                selected_model_header,
            ) = _extract_router_headers(raw_response)

            # response = client.chat.completions.create(
            #     model=model,
            #     messages=[{"role": "user", "content": prompt}],
            #     max_tokens=max_tokens,
            #     temperature=temperature,
            #     extra_body=extra_body if extra_body else None,
            # )
            # (
            #     selected_category,
            #     selected_decision,
            #     selected_model_header,
            # ) = _extract_router_headers(response)

        except AttributeError:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                seed=seed,
                extra_body=request_extra_body if request_extra_body else None,
            )
            (
                selected_category,
                selected_decision,
                selected_model_header,
            ) = _extract_router_headers(response)

        # Normalize empty-string headers to None for cleaner downstream fallback.
        if selected_category == "":
            selected_category = None
        if selected_decision == "":
            selected_decision = None
        if selected_model_header == "":
            selected_model_header = None

        # For reasoning models, content might be in reasoning_content instead of content
        responser = selected_model_header or response.model
        message = response.choices[0].message
        text = message.content or getattr(message, "reasoning_content", None) or ""
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None
        http_response_details = _extract_http_response_details(raw_response)
        return (
            text,
            True,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            responser,
            selected_category,
            selected_decision,
            http_response_details,
        )
    except Exception as e:
        # Keep any router decision headers if the provider exposes an HTTP response
        # object on exceptions (useful for debugging routing on failed calls).
        response_obj = getattr(e, "response", None)
        (
            selected_category,
            selected_decision,
            selected_model_header,
        ) = _extract_router_headers(response_obj)

        # Normalize empty headers after exception handling too.
        if selected_category == "":
            selected_category = None
        if selected_decision == "":
            selected_decision = None
        if selected_model_header == "":
            selected_model_header = None

        status_code = None
        if response_obj is not None:
            status_code = getattr(response_obj, "status_code", None)

        error_text = f"ERROR: {type(e).__name__}: {e}"
        if status_code is not None:
            error_text += f" [status={status_code}]"

        print(f"Model call failed: {error_text}")
        http_response_details = _extract_http_response_details(response_obj)
        return (
            error_text,
            False,
            None,
            None,
            None,
            selected_model_header,
            selected_category,
            selected_decision,
            http_response_details,
        )


def build_extra_body_for_model(
    model_name: str, reasoning: Optional[bool]
) -> Optional[Dict[str, Any]]:
    """Return an extra_body dict to toggle reasoning for a given model.

    This function matches the exact pattern from reasoning_eval_consolidated.py
    to ensure compatibility and consistent behavior.

    - DeepSeek v3.1: {"chat_template_kwargs": {"thinking": true/false}}
    - Qwen3: {"chat_template_kwargs": {"enable_thinking": true/false}}
    - GPT-OSS: {"reasoning_effort": "low|high"} based on reasoning flag
    """
    # reasoning: True -> ON, False -> OFF, None -> no reasoning parameters
    if reasoning is None:
        return None

    lower = model_name.lower()

    # DeepSeek v3.1 family (matches reasoning_eval_consolidated.py pattern)
    if (("ds" in lower) or ("deepseek" in lower)) and (
        "v31" in lower or "v3.1" in lower or "v3" in lower
    ):
        return {"chat_template_kwargs": {"thinking": reasoning}}

    # Qwen3 family (matches reasoning_eval_consolidated.py pattern)
    if "qwen3" in lower:
        return {"chat_template_kwargs": {"enable_thinking": reasoning}}

    # GPT-OSS family (matches reasoning_eval_consolidated.py pattern)
    if "gpt-oss" in lower or "openai/gpt-oss" in lower or "gpt_oss" in lower:
        effort = "high" if reasoning else "low"
        # Put reasoning_effort inside chat_template_kwargs (vLLM requirement)
        return {"chat_template_kwargs": {"reasoning_effort": effort}}

    # OpenAI models with reasoning parameter
    if "gpt" in lower or "o1" in lower:
        return {"reasoning": reasoning}

    # Model does not support reasoning parameters
    return None


def process_question_single(
    client: OpenAI,
    model: str,
    question: Question,
    dataset: Any,  # DatasetInterface
    prompt_mode: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
    frequency_penalty: float,
    seed: int,
    ar_extra_body: Optional[Dict[str, Any]] = None,
    mode_label: Optional[str] = None,
    debug_log_request_response: bool = False,
    debug_print_request_response: bool = False,
) -> Dict[str, Any]:
    """Process a single question with the model."""
    # Format prompt based on mode
    if prompt_mode == "XC":
        prompt = dataset.format_prompt(question, "explicit_cot")
        extra_body = (
            None  # XC mode never uses reasoning parameters (CoT prompt instead)
        )
    elif prompt_mode == "AR":
        prompt = dataset.format_prompt(question, "plain")
        extra_body = ar_extra_body
    else:  # NR mode (could be Router-Transparent or direct vLLM)
        prompt = dataset.format_prompt(question, "plain")
        # For Router-Transparent: ar_extra_body=None (router decides reasoning)
        # For direct vLLM: ar_extra_body contains reasoning parameters
        extra_body = ar_extra_body

    request_payload = None
    if debug_log_request_response or debug_print_request_response:
        # Build a reproducible request payload when debug logging/printing is enabled.
        request_payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "seed": seed,
            "extra_body": extra_body,
            "prompt_mode": prompt_mode,
            "mode_label": mode_label or prompt_mode,
        }

    start_time = time.time()
    (
        response_text,
        success,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        responser,
        predicted_category,
        selected_decision,
        http_response_details,
    ) = call_model(
        client,
        model,
        prompt,
        max_tokens,
        temperature,
        top_p,
        top_k,
        presence_penalty,
        frequency_penalty,
        seed,
        extra_body=extra_body,
    )
    end_time = time.time()

    predicted_answer = extract_answer(response_text, question) if success else None

    if debug_print_request_response:
        print("\n" + "=" * 30 + " DEBUG REQUEST/RESPONSE " + "=" * 30)
        print(
            json.dumps(
                request_payload,
                ensure_ascii=False,
                indent=2,
            )
        )
        print("--- HTTP RESPONSE ---")
        if http_response_details:
            print(
                json.dumps(
                    http_response_details,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
        else:
            print("(No HTTP response details captured)")
        print("--- RESPONSE TEXT ---")
        print(response_text)
        print("=" * 84 + "\n")

    # Compare predicted answer with correct answer (handle multiple formats)
    is_correct = False
    if predicted_answer:
        if hasattr(question, "options") and question.options:
            if len(question.options) == 2 and set(question.options) == {"Yes", "No"}:
                # Binary Yes/No questions (StrategyQA)
                is_correct = predicted_answer == question.correct_answer
            elif predicted_answer in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                # Multiple choice questions (GPQA, MMLU, etc.)
                if isinstance(question.correct_answer, str):
                    # Dataset stores answer as letter (e.g., MMLU: "F")
                    is_correct = predicted_answer == question.correct_answer
                elif isinstance(question.correct_answer, int):
                    # Dataset stores answer as index (e.g., CommonsenseQA: 1, ARC: 0)
                    predicted_idx = ord(predicted_answer) - ord("A")
                    is_correct = predicted_idx == question.correct_answer
        else:
            # Free-form questions (GSM8K, DROP, etc.)
            is_correct = compare_free_form_answers(
                predicted_answer, question.correct_answer
            )

    result = {
        "mode": prompt_mode,
        "mode_label": mode_label or prompt_mode,
        "requested_model": model,
        "question_id": question.question_id,
        "category": question.category,
        "predicted_category": predicted_category,
        "selected_decision": selected_decision,
        "question": question.question,
        "options": question.options,
        "correct_answer": question.correct_answer,
        "model_response": response_text,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "response_time": end_time - start_time,
        "success": success,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "responser": responser,
    }

    if debug_log_request_response:
        result["request_prompt"] = prompt
        result["request_payload"] = json.dumps(request_payload, ensure_ascii=False)
        result["http_response_details"] = json.dumps(
            http_response_details, ensure_ascii=False, default=str
        ) if http_response_details else None

    return result


def evaluate_model_router_transparent(
    questions: List[Question],
    dataset: Any,  # DatasetInterface
    model: str,
    endpoint: str,
    api_key: str,
    concurrent_requests: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
    frequency_penalty: float,
    seed: int,
    debug_log_request_response: bool = False,
    debug_print_request_response: bool = False,
) -> pd.DataFrame:
    """Evaluate model in router-transparent mode."""
    client = OpenAI(base_url=endpoint, api_key=api_key or None, timeout=300.0)
    print(f"Using model: {model}, endpoint: {endpoint}")

    results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = []
        for i, question in enumerate(questions):
            if i < 0 or i >= 1:
                continue  # Only run the x question for router evaluation to save time
            futures.append(
                executor.submit(
                    process_question_single,
                    client,
                    model,
                    question,
                    dataset,
                    "NR",
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    presence_penalty,
                    frequency_penalty,
                    seed,
                    None,
                    mode_label="Router_NR",
                    debug_log_request_response=debug_log_request_response,
                    debug_print_request_response=debug_print_request_response,
                )
            )

        try:
            for future in tqdm(
                futures,
                total=len(futures),
                desc=f"Evaluating {model} (Router-Transparent)",
            ):
                results.append(future.result())
        except KeyboardInterrupt:
            print(
                "\n⚠️  Router evaluation interrupted by user. Saving partial results..."
            )
            # Cancel remaining futures
            for future in futures:
                future.cancel()
            # Collect results from completed futures
            for future in futures:
                if future.done() and not future.cancelled():
                    try:
                        results.append(future.result())
                    except Exception:
                        pass  # Skip failed results
            if not results:
                print("❌ No router results to save.")
                raise
            print(f"✅ Saved {len(results)} partial router results.")

    return pd.DataFrame(results)


def evaluate_model_vllm_multimode(
    questions: List[Question],
    dataset: Any,  # DatasetInterface
    model: str,
    endpoint: str,
    api_key: str,
    concurrent_requests: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
    frequency_penalty: float,
    seed: int,
    exec_modes: List[str],
    debug_log_request_response: bool = False,
    debug_print_request_response: bool = False,
) -> pd.DataFrame:
    """Run vLLM with 2-3 realistic reasoning scenarios.

    The scenarios represent real-world router decision patterns:
    1. NR - Plain prompt, no reasoning toggle (fast baseline) - ALWAYS included
    2. XC - CoT prompt, no reasoning toggle (prompt-based reasoning) - ONLY if dataset has CoT
    3. NR_REASONING - Plain prompt, reasoning toggle ON (model-based reasoning) - ALWAYS included
    """
    client = OpenAI(base_url=endpoint, api_key=api_key or "dummy-key", timeout=300.0)
    print(f"Using vLLM model: {model}, endpoint: {endpoint}")

    # Check if dataset has actual CoT content by examining sample questions
    has_cot_content = any(
        q.cot_content is not None and q.cot_content.strip() for q in questions[:10]
    )

    # Debug: Show CoT content status for first few questions
    print(f"  CoT Debug - Checking first 10 questions:")
    for i, q in enumerate(questions[:10]):
        cot_status = (
            "None"
            if q.cot_content is None
            else (
                f"'{q.cot_content[:50]}...'"
                if len(q.cot_content) > 50
                else f"'{q.cot_content}'"
            )
        )
        print(f"    Q{i+1}: CoT = {cot_status}")

    if has_cot_content:
        print(f"  Dataset has CoT content - using 3 modes: NR, XC, NR_REASONING")
    else:
        print(
            f"  Dataset lacks CoT content - using 2 modes: NR, NR_REASONING (skipping XC)"
        )

    results: List[Dict[str, Any]] = []

    # Define mode variants based on model type and CoT availability
    model_lower = model.lower()
    is_deepseek_or_qwen = (
        (("ds" in model_lower) or ("deepseek" in model_lower))
        and ("v31" in model_lower or "v3.1" in model_lower or "v3" in model_lower)
    ) or ("qwen3" in model_lower)

    # Base modes (always included)
    # Always use explicit True/False for reasoning-capable models to ensure consistent behavior
    mode_variants: List[Tuple[str, str, Optional[bool]]] = [
        ("VLLM_NR", "NR", False),  # Plain prompt, reasoning OFF (baseline)
        # (
        #     "VLLM_NR_REASONING",
        #     "NR",
        #     True,
        # ),  # Plain prompt, reasoning ON (model reasoning)
    ]

    # Add XC mode only if dataset has CoT content
    if has_cot_content:
        # Always use explicit False for XC mode (CoT prompt with reasoning OFF)
        mode_variants.insert(
            1, ("VLLM_XC", "XC", False)
        )  # Insert between NR and NR_REASONING

    def run_variants(q: Question) -> List[Dict[str, Any]]:
        local_records: List[Dict[str, Any]] = []
        for label, prompt_mode, reasoning_flag in mode_variants:
            extra_body = build_extra_body_for_model(model, reasoning_flag)
            # Debug: print extra_body for first question to verify configuration
            if q == questions[0]:
                print(
                    f"  {label}: reasoning_flag={reasoning_flag}, extra_body={extra_body}"
                )
            rec = process_question_single(
                client,
                model,
                q,
                dataset,
                prompt_mode,
                max_tokens,
                temperature,
                top_p,
                top_k,
                presence_penalty,
                frequency_penalty,
                seed,
                ar_extra_body=extra_body,
                mode_label=label,
                debug_log_request_response=debug_log_request_response,
                debug_print_request_response=debug_print_request_response,
            )
            local_records.append(rec)
        return local_records

    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = [executor.submit(run_variants, q) for q in questions]
        try:
            for future in tqdm(
                futures, total=len(futures), desc=f"Evaluating {model} (vLLM modes)"
            ):
                results.extend(future.result())
        except KeyboardInterrupt:
            print("\n⚠️  Benchmark interrupted by user. Saving partial results...")
            # Cancel remaining futures
            for future in futures:
                future.cancel()
            # Collect results from completed futures
            for future in futures:
                if future.done() and not future.cancelled():
                    try:
                        results.extend(future.result())
                    except Exception:
                        pass  # Skip failed results
            if not results:
                print("❌ No results to save.")
                raise
            print(f"✅ Saved {len(results)} partial results.")

    return pd.DataFrame(results)


def analyze_results(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze results and compute metrics."""
    valid = results_df[results_df["success"]]
    overall_acc = valid["is_correct"].mean() if not valid.empty else 0.0

    category_metrics: Dict[str, Dict[str, Any]] = {}
    for category in valid["category"].unique():
        sub = valid[valid["category"] == category]
        category_metrics[category] = {
            "accuracy": float(sub["is_correct"].mean()) if not sub.empty else 0.0,
            "avg_response_time": (
                float(sub["response_time"].mean()) if not sub.empty else 0.0
            ),
            "avg_prompt_tokens": (
                float(sub["prompt_tokens"].dropna().mean())
                if not sub["prompt_tokens"].dropna().empty
                else None
            ),
            "avg_completion_tokens": (
                float(sub["completion_tokens"].dropna().mean())
                if not sub["completion_tokens"].dropna().empty
                else None
            ),
            "avg_total_tokens": (
                float(sub["total_tokens"].dropna().mean())
                if not sub["total_tokens"].dropna().empty
                else None
            ),
        }

    avg_latency = valid["response_time"].mean() if not valid.empty else 0.0
    avg_prompt_tokens = (
        valid["prompt_tokens"].dropna().mean() if not valid.empty else None
    )
    avg_completion_tokens = (
        valid["completion_tokens"].dropna().mean() if not valid.empty else None
    )
    avg_total_tokens = (
        valid["total_tokens"].dropna().mean() if not valid.empty else None
    )

    # Optional: metrics by mode_label
    by_mode: Dict[str, Dict[str, Any]] = {}
    if "mode_label" in valid.columns:
        for label in valid["mode_label"].unique():
            sub = valid[valid["mode_label"] == label]
            by_mode[label] = {
                "accuracy": float(sub["is_correct"].mean()) if not sub.empty else 0.0,
                "avg_response_time": (
                    float(sub["response_time"].mean()) if not sub.empty else 0.0
                ),
                "avg_prompt_tokens": (
                    float(sub["prompt_tokens"].dropna().mean())
                    if not sub["prompt_tokens"].dropna().empty
                    else None
                ),
                "avg_completion_tokens": (
                    float(sub["completion_tokens"].dropna().mean())
                    if not sub["completion_tokens"].dropna().empty
                    else None
                ),
                "avg_total_tokens": (
                    float(sub["total_tokens"].dropna().mean())
                    if not sub["total_tokens"].dropna().empty
                    else None
                ),
            }

    return {
        "overall_accuracy": float(overall_acc),
        "category_metrics": category_metrics,
        "avg_response_time": float(avg_latency) if avg_latency is not None else 0.0,
        "avg_prompt_tokens": (
            float(avg_prompt_tokens) if avg_prompt_tokens is not None else None
        ),
        "avg_completion_tokens": (
            float(avg_completion_tokens) if avg_completion_tokens is not None else None
        ),
        "avg_total_tokens": (
            float(avg_total_tokens) if avg_total_tokens is not None else None
        ),
        "total_questions": int(len(results_df)),
        "successful_queries": int(len(valid)),
        "failed_queries": int(len(results_df) - len(valid)),
        "by_mode": by_mode,
    }


def save_results(
    results_df: pd.DataFrame,
    analysis: Dict[str, Any],
    model: str,
    dataset_name: str,
    output_dir: str,
    debug_log_request_response: bool = False,
):
    """Save results to files."""
    model_name = model.replace("/", "_")
    model_dir = os.path.join(output_dir, f"{dataset_name}_{model_name}")
    os.makedirs(model_dir, exist_ok=True)

    detailed_results_df = results_df.copy()
    if not debug_log_request_response:
        detailed_results_df = detailed_results_df.drop(
            columns=["request_prompt", "request_payload", "http_response_details", "model_response"],
            errors="ignore",
        )
    detailed_results_df.to_csv(
        os.path.join(model_dir, "detailed_results.csv"), index=False
    )

    # Per-example prediction log focused on the routing/category prediction signal.
    # Add explicit source/reason fields so fallback values (for example "MoM") are
    # easy to interpret when x-vsr-selected-category is absent.
    prediction_log_df = results_df.copy()
    if "predicted_category" not in prediction_log_df.columns:
        prediction_log_df["predicted_category"] = None

    def _has_value(v: Any) -> bool:
        if pd.isna(v):
            return False
        return str(v).strip() != ""

    def _extract_category_from_text(v: Any) -> Optional[str]:
        """Best-effort extraction of routed category from debug/decision text."""
        if not _has_value(v):
            return None

        text = str(v)

        # Common router log style: domain=[computer science]
        match = re.search(r"(?:domain|category)\s*=\s*\[\s*([^\]]+?)\s*\]", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # JSON-like style: "category": "computer science"
        match = re.search(r'"category"\s*:\s*"([^"]+?)"', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return None

    def _resolve_predicted_category(row: pd.Series) -> Tuple[str, str, str]:
        header_category = row.get("predicted_category")
        decision = row.get("selected_decision")
        responser = row.get("responser")
        requested_model = row.get("requested_model")
        success = bool(row.get("success")) if "success" in row else True
        model_response = row.get("model_response")

        if _has_value(header_category):
            category = str(header_category)
            if _has_value(decision):
                reason = (
                    "Router selected category from x-vsr-selected-category "
                    f"(x-vsr-selected-decision={decision})"
                )
            else:
                reason = "Router selected category from x-vsr-selected-category"
            return category, "x_vsr_selected_category", reason

        decision_category = _extract_category_from_text(decision)
        if decision_category:
            reason = (
                "x-vsr-selected-category missing; inferred category from "
                f"x-vsr-selected-decision (decision={decision})"
            )
            return decision_category, "x_vsr_selected_decision_inferred", reason

        response_category = _extract_category_from_text(model_response)
        if response_category:
            reason = (
                "x-vsr-selected-category missing; inferred category from "
                "model_response text"
            )
            return response_category, "model_response_inferred", reason

        if not success:
            reason = (
                "x-vsr-selected-category missing on failed request; category is "
                "unknown (request did not return router headers)"
            )
            return "unknown", "missing_on_failed_request", reason

        if _has_value(requested_model):
            requested_model_str = str(requested_model)
            reason = (
                "x-vsr-selected-category missing; used requested_model fallback "
                f"(requested_model={requested_model_str})"
            )
            return requested_model_str, "requested_model_fallback", reason

        if _has_value(responser):
            responder_str = str(responser)
            reason = (
                "x-vsr-selected-category missing; used responder fallback "
                f"(responser={responder_str})"
            )
            return responder_str, "responser_fallback", reason

        reason = (
            "x-vsr-selected-category missing; used run model fallback "
            f"(model={model})"
        )
        return model, "run_model_fallback", reason

    prediction_log_df[
        [
            "predicted_category",
            "predicted_category_source",
            "predicted_category_reason",
        ]
    ] = prediction_log_df.apply(_resolve_predicted_category, axis=1, result_type="expand")

    preferred_cols = [
        "question_id",
        "mode_label",
        "requested_model",
        "category",
        "predicted_category",
        "predicted_category_source",
        "predicted_category_reason",
        "selected_decision",
        "predicted_answer",
        "correct_answer",
        "is_correct",
        "success",
        "response_time",
    ]
    if debug_log_request_response:
        preferred_cols = [
            "question_id",
            "mode_label",
            "requested_model",
            "request_prompt",
            "request_payload",
            "http_response_details",
            "model_response",
            "category",
            "predicted_category",
            "predicted_category_source",
            "predicted_category_reason",
            "selected_decision",
            "predicted_answer",
            "correct_answer",
            "is_correct",
            "success",
            "response_time",
        ]

    if not debug_log_request_response:
        prediction_log_df = prediction_log_df.drop(
            columns=["request_prompt", "request_payload", "http_response_details", "model_response"],
            errors="ignore",
        )
    existing_cols = [c for c in preferred_cols if c in prediction_log_df.columns]
    trailing_cols = [c for c in prediction_log_df.columns if c not in existing_cols]
    prediction_log_df = prediction_log_df[existing_cols + trailing_cols]
    prediction_log_df.to_csv(
        os.path.join(model_dir, "predicted_category_log.csv"), index=False
    )

    with open(os.path.join(model_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "model": model,
                "dataset": dataset_name,
                **analysis,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 50)
    print(f"Model: {model} | Dataset: {dataset_name}")
    print(f"Overall Accuracy: {analysis['overall_accuracy']:.4f}")
    print(f"Total Questions: {analysis['total_questions']}")
    print(f"Successful Queries: {analysis['successful_queries']}")
    print(f"Failed Queries: {analysis['failed_queries']}")
    print(
        f"Avg Latency: {analysis['avg_response_time']:.2f}s | Avg Total Tokens: {analysis['avg_total_tokens']}"
    )
    print("=" * 50 + "\n")

    if "category_metrics" in analysis:
        print("Category Metrics (acc | latency | total_tokens):")
        printable = []
        for category, met in analysis["category_metrics"].items():
            printable.append((category, met.get("accuracy", 0.0)))
        for category, acc in sorted(printable, key=lambda x: x[1], reverse=True):
            m = analysis["category_metrics"][category]
            print(
                f"  {category}: acc={m['accuracy']:.4f}, latency={m['avg_response_time']:.2f}s, tokens={m['avg_total_tokens']}"
            )
        print()


def main():
    args = parse_args()

    # Handle dataset listing
    if args.list_datasets:
        list_available_datasets()
        return

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    try:
        dataset = DatasetFactory.create_dataset(args.dataset)
        questions, dataset_info = dataset.load_dataset(
            categories=args.categories,
            samples_per_category=args.samples_per_category,
            seed=args.seed,
        )
        print(
            f"Dataset loaded: {len(questions)} questions across {len(dataset_info.categories)} categories"
        )
        print(f"Categories: {', '.join(dataset_info.categories)}")

        # Check for empty dataset
        if len(questions) == 0:
            print(f"❌ No questions loaded from dataset '{args.dataset}'")
            print("This could be due to:")
            print("  - Dataset requiring authentication (gated dataset)")
            print("  - Network connectivity issues")
            print("  - Invalid dataset name or configuration")
            print("\nTry a different dataset:")
            list_available_datasets()
            return

    except Exception as e:
        print(f"Error loading dataset '{args.dataset}': {e}")
        print("\nAvailable datasets:")
        list_available_datasets()
        return

    # Resolve endpoints and models
    router_endpoint = (
        args.router_endpoint
        or os.environ.get("ROUTER_ENDPOINT")
        or "http://127.0.0.1:8801/v1"
    )
    router_api_key = (
        args.router_api_key
        or os.environ.get("ROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "1234"
    )

    vllm_endpoint = args.vllm_endpoint or os.environ.get("VLLM_ENDPOINT", "")
    vllm_api_key = (
        args.vllm_api_key
        or os.environ.get("VLLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )

    router_models = args.router_models
    if router_models and len(router_models) == 1 and "," in router_models[0]:
        router_models = router_models[0].split(",")
    if not router_models or (len(router_models) == 1 and router_models[0] == "auto"):
        print("Fetching available models from router endpoint...")
        fetched_models = get_available_models(router_endpoint, router_api_key)
        if fetched_models:
            router_models = fetched_models
        else:
            print("No models returned from endpoint, using 'auto' as fallback")
            router_models = ["auto"]

    vllm_models = args.vllm_models
    if vllm_models and len(vllm_models) == 1 and "," in vllm_models[0]:
        vllm_models = vllm_models[0].split(",")
    if not vllm_models and vllm_endpoint:
        print("Fetching available models from vLLM endpoint...")
        vllm_models = get_available_models(vllm_endpoint, vllm_api_key)

    print(f"Router models: {router_models}")
    print(f"vLLM models: {vllm_models}")

    # Function to get optimal tokens for a specific model
    # Use model-aware token allocation for optimal performance
    def get_model_optimal_tokens(model_name):
        if args.max_tokens:
            return args.max_tokens
        else:
            # For router evaluation, use the first vLLM model for token calculation if available
            # This ensures consistent token allocation between router and vLLM evaluations
            reference_model = None
            if vllm_models and len(vllm_models) > 0:
                reference_model = vllm_models[0]
                print(
                    f"  🔗 Using vLLM model '{reference_model}' for router token calculation"
                )
            elif model_name and model_name != "auto":
                reference_model = model_name

            return get_dataset_optimal_tokens(dataset_info, model_name=reference_model)

    # Router evaluation (NR-only)
    if args.run_router and router_endpoint and router_models:
        for model in router_models:
            model_tokens = get_model_optimal_tokens(model)
            print(f"\nEvaluating router model: {model}")
            print(
                f"Using max_tokens: {model_tokens} (dataset-optimized for fair comparison)"
            )
            rt_df = evaluate_model_router_transparent(
                questions=questions,
                dataset=dataset,
                model=model,
                endpoint=router_endpoint,
                api_key=router_api_key,
                concurrent_requests=args.concurrent_requests,
                max_tokens=model_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                presence_penalty=args.presence_penalty,
                frequency_penalty=args.frequency_penalty,
                seed=args.seed,
                debug_log_request_response=args.debug_log_request_response,
                debug_print_request_response=args.debug_print_request_response,
            )
            analysis = analyze_results(rt_df)
            save_results(
                results_df=rt_df,
                analysis=analysis,
                model=f"router::{model}",
                dataset_name=dataset_info.name,
                output_dir=args.output_dir,
                debug_log_request_response=args.debug_log_request_response,
            )

    # Direct vLLM evaluation (NR/XC with reasoning ON/OFF)
    if args.run_vllm and vllm_endpoint and vllm_models:
        for model in vllm_models:
            model_tokens = get_model_optimal_tokens(model)
            print(f"\nEvaluating vLLM model: {model}")
            print(
                f"Using max_tokens: {model_tokens} (dataset-optimized for fair comparison)"
            )
            vdf = evaluate_model_vllm_multimode(
                questions=questions,
                dataset=dataset,
                model=model,
                endpoint=vllm_endpoint,
                api_key=vllm_api_key,
                concurrent_requests=args.concurrent_requests,
                max_tokens=model_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                presence_penalty=args.presence_penalty,
                frequency_penalty=args.frequency_penalty,
                seed=args.seed,
                exec_modes=args.vllm_exec_modes,
                debug_log_request_response=args.debug_log_request_response,
                debug_print_request_response=args.debug_print_request_response,
            )
            analysis = analyze_results(vdf)
            save_results(
                results_df=vdf,
                analysis=analysis,
                model=f"vllm::{model}",
                dataset_name=dataset_info.name,
                output_dir=args.output_dir,
                debug_log_request_response=args.debug_log_request_response,
            )


if __name__ == "__main__":
    main()
