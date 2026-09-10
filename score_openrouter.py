import argparse
import os
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

from utils.prompts_utils import SCORING_PROMPTS, GENERATION_PROMPTS
from utils.scoring_utils import (
    resolve_output_path,
    load_generation_df,
    build_pairs,
    build_direction_prompts,
    parse_generation_scoring,
    expand_pairs_for_variations,
)

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
        "--judge_model_id", type=str, required=True,
        help="OpenRouter model slug for the judge, e.g. "
             "'anthropic/claude-3.5-sonnet' or 'meta-llama/llama-3.3-70b-instruct'.",
    )
    parser.add_argument(
        "--generation_model_id", type=str, default=None,
        help="Identifier for the model whose generated opinions to load and score "
             "(used only for locating/labeling the generation file, not for API calls).",
    )
    parser.add_argument(
        "--generation_prompt_version", type=str, default="base", choices=list(GENERATION_PROMPTS.keys()),
        help="Which generation-prompt style's opinions to load and score.",
    )
    parser.add_argument(
        "--scoring_prompt_version", type=str, default="0_1", choices=list(SCORING_PROMPTS.keys()),
        help="Version of the judge prompt to use.",
    )
    parser.add_argument("--generation_csv_path", type=str, default=None)
    parser.add_argument("--max_concurrency", type=int, default=8,
                         help="Number of concurrent OpenRouter requests in flight.")
    parser.add_argument("--output_dir", type=str, default="scoring")
    parser.add_argument("--num_examples", type=int, default=None)
    parser.add_argument("--option_setting", type=str, default="0_1", choices=["four", "0_1"])
    parser.add_argument("--extract_ids_from", type=Path, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=5,
                         help="Retries per request on transient API errors.")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                         help="Save progress to disk every N completed requests.")
    return parser.parse_args()


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

    gen_df = load_generation_df(args)
    pairs_df = build_pairs(gen_df)

    if args.num_examples is not None:
        pairs_df = pairs_df.head(args.num_examples)

    print(f"Scoring {len(pairs_df)} pairs of opinions from {len(gen_df)} total generations.")

    if args.extract_ids_from is not None:
        reference_df_ids = pd.read_csv(args.extract_ids_from)
        pairs_df = pairs_df.merge(
            reference_df_ids[["id_1", "id_2"]],
            on=["id_1", "id_2"],
            how="inner",
        )

    output_path = resolve_output_path(args)
    print(f"Judge (OpenRouter): {args.judge_model_id} | max_concurrency: {args.max_concurrency}")

    entries = build_direction_prompts(pairs_df, args.scoring_prompt_version)  # (variation, direction, id1, id2, prompt)
    prompts = [e[4] for e in entries]

    pairs_df = expand_pairs_for_variations(pairs_df, args.scoring_prompt_version)

    generations = [None] * len(prompts)

    def save_progress():
        pairs_df["generated_score_1to2"] = pd.Series(generations[0::2])
        pairs_df["generated_score_2to1"] = pd.Series(generations[1::2])
        pairs_df["judge_model"] = args.judge_model_id
        pairs_df["generation_model"] = args.generation_model_id
        pairs_df["generation_prompt_version"] = args.generation_prompt_version
        pairs_df["parsed_score_1to2"] = pairs_df["generated_score_1to2"].apply(
            parse_generation_scoring, option_setting=args.option_setting
        )
        pairs_df["parsed_score_2to1"] = pairs_df["generated_score_2to1"].apply(
            parse_generation_scoring, option_setting=args.option_setting
        )
        pairs_df.to_csv(output_path, index=False)

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
                            desc=f"Scoring ({args.judge_model_id})"):
            idx = future_to_idx[future]
            try:
                result = future.result()
                generations[idx] = result
                # Show first few responses for debugging
                if sample_shown < 3:
                    parsed = parse_generation_scoring(result, option_setting=args.option_setting)
                    print(f"[sample {idx}] raw: {repr(result)[:100]} -> parsed: {parsed}")
                    sample_shown += 1
            except Exception as e:
                print(f"[error] prompt {idx} failed permanently: {e}")
                generations[idx] = ""

            completed += 1
            if completed % args.checkpoint_every == 0:
                save_progress()

    save_progress()
    print(f"Saved {len(pairs_df)} pairs (with 1->2 and 2->1 scores) to {output_path}")

    # Debug: show response statistics
    empty_count = sum(1 for g in generations if g == "")
    none_count = sum(1 for g in generations if g is None)
    print(f"Response stats: {len(generations)} total, {empty_count} empty, {none_count} None")
    print(f"Sample raw responses: {[repr(g)[:50] for g in generations[:5]]}")


if __name__ == "__main__":
    main()