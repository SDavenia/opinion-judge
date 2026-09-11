"""
OpenRouter (API) equivalent of extract_model_embedded_opinion.py: repeatedly prompts a
closed/API-only judge (an OpenRouter model slug, e.g. "anthropic/claude-haiku-4.5") to
state its own stance on each situation in a dataset, and tallies the label distribution
-- grounding the model's own embedded opinion on each situation.

Same --dataset_id / --alignment_prompt_version / output-path conventions as
extract_model_embedded_opinion.py (model_alignment/{dataset_id}/{judge_model_id}/...),
so local (REGISTRY) and OpenRouter judges land in the same tree and are both picked up
by analyse_results.py's --run_on open/closed split.

Usage:
    python extract_model_embedded_opinion_openrouter.py \
        --dataset_id valueprism --judge_model_id anthropic/claude-haiku-4.5

    python extract_model_embedded_opinion_openrouter.py \
        --dataset_id habermas --judge_model_id openai/gpt-5-nano
"""
import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

from utils.prompts_utils import ALIGNMENT_PROMPTS, ALIGNMENT_PROMPTS_BY_DATASET, LABEL_MAPPING

# ---------------------------------------------------------------------------
# .env / client setup
# ---------------------------------------------------------------------------
load_dotenv()  # reads .env in the current working directory (or a parent)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY not found. Add a line like "
        "OPENROUTER_API_KEY=sk-or-... to a .env file in your project root."
    )

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    # Optional but recommended by OpenRouter for routing/rate-limit headers.
    default_headers={
        "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", ""),
        "X-Title": os.environ.get("OPENROUTER_SITE_NAME", "opinion-judge"),
    },
)


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_id", type=str, required=True, choices=["habermas", "valueprism"],
        help="Which prepared dataset's situations to probe.",
    )
    parser.add_argument(
        "--judge_model_id", type=str, required=True,
        help="OpenRouter model slug for the judge whose embedded opinion to extract, "
             "e.g. 'anthropic/claude-haiku-4.5' or 'openai/gpt-5-nano'.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=10, help="Max tokens to generate per opinion")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation")
    parser.add_argument("--n_samples", type=int, default=30, help="Number of times to repeat the generation for each situation")

    parser.add_argument("--path_dataset", type=Path, default=None,
                        help="Path of dataset to use. Defaults to data/{dataset_id}_sample.csv.")
    parser.add_argument("--output_dir", type=Path, default="model_alignment", help="Directory to save the output CSV and JSON files")
    parser.add_argument(
        "--alignment_prompt_version", type=str, default=None,
        help="Version of the prompt to use for parsing generations. Must be valid for "
             f"--dataset_id (see ALIGNMENT_PROMPTS_BY_DATASET). Defaults to "
             "'impartial_evaluator' for valueprism, 'hb_impartial_evaluator' for habermas.",
    )
    parser.add_argument("--max_concurrency", type=int, default=8,
                         help="Number of concurrent OpenRouter requests in flight.")
    parser.add_argument("--max_retries", type=int, default=5,
                         help="Retries per request on transient API errors.")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                         help="Save progress to disk every N completed requests.")
    args = parser.parse_args()
    if args.path_dataset is None:
        args.path_dataset = Path(f"data/{args.dataset_id}_sample.csv")
    if args.alignment_prompt_version is None:
        args.alignment_prompt_version = ALIGNMENT_PROMPTS_BY_DATASET[args.dataset_id][0]
    elif args.alignment_prompt_version not in ALIGNMENT_PROMPTS_BY_DATASET[args.dataset_id]:
        parser.error(
            f"--alignment_prompt_version={args.alignment_prompt_version!r} is not valid for "
            f"--dataset_id={args.dataset_id!r}. Valid options: "
            f"{ALIGNMENT_PROMPTS_BY_DATASET[args.dataset_id]}"
        )
    return args


def prepare_alignment_prompt(situation, prompt_version, n_samples):
    prompt = ALIGNMENT_PROMPTS[prompt_version].format(
        situation=situation
    )
    return [prompt] * n_samples


def build_label_regex(labels):
    """
    Build a single regex that matches any of the given labels, longest first,
    so that e.g. "I WOULD NOT" is preferred over a partial match on "I WOULD".
    Labels may contain spaces/underscores; both are treated as literal text,
    matched case-insensitively, bounded by word boundaries.
    """
    ordered = sorted(labels, key=len, reverse=True)
    escaped = [re.escape(label) for label in ordered]
    pattern = r'\b(' + '|'.join(escaped) + r')\b'
    return re.compile(pattern, re.IGNORECASE)


def parse_generations(generations, prompt_version):
    if prompt_version not in LABEL_MAPPING:
        raise NotImplementedError(f"Prompt version {prompt_version} not implemented for parsing generations.")

    labels = LABEL_MAPPING[prompt_version]
    label_regex = build_label_regex(labels)
    # Map the uppercased, matched text back to the canonical label spelling
    # (regex match is case-insensitive, so we normalize the result).
    canonical_by_upper = {label.upper(): label for label in labels}

    parsed_results = []
    for gen in generations:
        # gen is None for requests not yet completed (checkpointing runs mid-flight).
        if not gen:
            parsed_results.append(None)
            continue
        match = label_regex.search(gen)
        if match:
            matched_upper = match.group(1).upper()
            evaluation = canonical_by_upper.get(matched_upper, matched_upper)
            parsed_results.append(evaluation)
        else:
            parsed_results.append(None)
    return parsed_results


def get_situations(args):
    df = pd.read_csv(args.path_dataset)
    # One row per unique situation, keeping its situation_id from the dataset.
    # If the same situation string appears with multiple situation_ids, the
    # first occurrence's id is kept.
    situation_lookup = df[["situation_id", "situation"]].drop_duplicates(subset="situation")
    return situation_lookup["situation"].tolist(), dict(zip(situation_lookup["situation"], situation_lookup["situation_id"]))


def call_openrouter(model: str, prompt: str, max_tokens: int, temperature: float, max_retries: int) -> str:
    """Single chat-completion call with exponential backoff on transient errors."""
    messages = [{"role": "user", "content": prompt}]
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = (resp.choices[0].message.content or "").strip()
            if not content and attempt == 0:
                print(f"[warn] model returned empty content")
                print(f"  Full response: {resp}")
                print(f"  Finish reason: {resp.choices[0].finish_reason}")
            return content
        except (RateLimitError, APITimeoutError, APIError) as e:
            if attempt == max_retries - 1:
                print(f"[error] giving up on prompt after {max_retries} attempts: {e}")
                return ""
            sleep_for = delay + random.uniform(0, 0.5)
            time.sleep(sleep_for)
            delay = min(delay * 2, 30)
    return ""


def main():
    args = parse_command_line_args()
    out_dir = args.output_dir / args.dataset_id / args.judge_model_id
    os.makedirs(out_dir, exist_ok=True)

    situations, situation_id_lookup = get_situations(args)
    # Keep only first 100 situations for debugging 
    situations = situations[:100]

    print(f"Judge (OpenRouter): {args.judge_model_id} | situations: {len(situations)} | "
          f"n_samples: {args.n_samples} | max_concurrency: {args.max_concurrency}")

    situations, prompts = [[situation] * args.n_samples for situation in situations], [prepare_alignment_prompt(situation, args.alignment_prompt_version, args.n_samples) for situation in situations]

    # Flatten the lists of situations and prompts
    situations = [item for sublist in situations for item in sublist]
    prompts = [item for sublist in prompts for item in sublist]
    generations = [None] * len(prompts)

    path_generations_csv = out_dir / f"{args.alignment_prompt_version}_generations.csv"

    def save_progress():
        parsed = parse_generations(generations, args.alignment_prompt_version)
        df = pd.DataFrame({
            "situation": situations,
            "generation": generations,
            "parsed_evaluations": parsed,
        })
        df.to_csv(path_generations_csv, encoding="utf-8", index=False)
        return parsed

    with ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
        future_to_idx = {
            pool.submit(
                call_openrouter,
                args.judge_model_id,
                prompt,
                args.max_new_tokens,
                args.temperature,
                args.max_retries,
            ): i
            for i, prompt in enumerate(prompts)
        }

        completed = 0
        sample_shown = 0
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx),
                            desc=f"Probing ({args.judge_model_id})"):
            idx = future_to_idx[future]
            try:
                result = future.result()
                generations[idx] = result
                # Show first few responses for debugging
                if sample_shown < 3:
                    print(f"[sample {idx}] raw: {repr(result)[:100]}")
                    sample_shown += 1
            except Exception as e:
                print(f"[error] prompt {idx} failed permanently: {e}")
                generations[idx] = ""

            completed += 1
            if completed % args.checkpoint_every == 0:
                save_progress()

    parsed_generations = save_progress()
    print(f"Saving generations and parsed evaluations to {path_generations_csv}")

    # Debug: show response statistics
    empty_count = sum(1 for g in generations if g == "")
    none_count = sum(1 for g in generations if g is None)
    print(f"Response stats: {len(generations)} total, {empty_count} empty, {none_count} None")
    print(f"Sample raw responses: {[repr(g)[:50] for g in generations[:5]]}")

    # Compute the distribution over labels for each situation. This supports
    # an arbitrary number of labels (as defined in LABEL_MAPPING), not just a
    # binary pair, so prompts like "likert_5" work without any special-casing.
    values = LABEL_MAPPING[args.alignment_prompt_version]

    unique_situations = list(dict.fromkeys(situations))  # preserve first-seen order, dedup

    output_rows = []
    for situation in unique_situations:
        # Get all evaluations for this situation
        evaluations = [parsed_generations[i] for i in range(len(situations)) if situations[i] == situation]

        counts = {label: evaluations.count(label) for label in values}
        total_count = sum(counts.values())

        if total_count > 0:
            probs = {f"p({label})": counts[label] / total_count for label in values}
        else:
            probs = {f"p({label})": 0.0 for label in values}

        alignment_distribution = {
            **probs,
            "total_evaluations": total_count
        }

        output_rows.append({
            "situation_id": situation_id_lookup.get(situation),
            "situation": situation,
            "alignment_distribution": alignment_distribution
        })

    path_distribution_csv = out_dir / f"{args.alignment_prompt_version}.csv"
    print(f"Saving alignment distributions to {path_distribution_csv}")
    distribution_df = pd.DataFrame(output_rows)
    # alignment_distribution is written out via json.dumps so the cell is a
    # valid JSON string (parse back with json.loads(...) rather than eval).
    distribution_df["alignment_distribution"] = distribution_df["alignment_distribution"].apply(json.dumps)
    distribution_df.to_csv(path_distribution_csv, encoding="utf-8", index=False)


if __name__ == "__main__":
    main()
