"""
Judge-score flip / direction analysis, stratified by how safe vs. unsafe
each pairwise comparison is (and, within that, by how certain the model
was about its own stance on the underlying situation).

Reports are written to:

    {output_dir}/{judge_model_id}/report.txt
    {output_dir}/{judge_model_id}/support_oppose_flip_rate_by_certainty.png
    {output_dir}/{judge_model_id}/support_oppose_direction_by_certainty.png
    {output_dir}/{judge_model_id}/either_flip_rate_by_certainty.png
    {output_dir}/{judge_model_id}/either_direction_by_certainty.png

report.txt contains an exact copy of everything printed to the terminal
for this run (see Tee, below); the run also still prints to stdout as
before.
"""

import argparse
import contextlib
import json
import math
import os
import re
import sys

import pandas as pd

from utils.models_utils import REGISTRY
from utils.prompts_utils import GENERATION_PROMPTS
from extract_model_embedded_opinion import ALIGNMENT_PROMPTS
from utils.prompts_utils import JUDGE_PROMPTS


# ==========================================================================
# Configuration
# ==========================================================================

# Columns not needed for the analysis.
#
# NOTE: generated_score_1to2 / generated_score_2to1 are intentionally NOT
# dropped -- they're renamed to parsed_score1to2 / parsed_score2to1 below.
DROP_COLUMNS = [
    "vrd_1", "text_1", "explanation_1", "generated_opinion_1", "model_used_1",
    "vrd_2", "text_2", "explanation_2", "generated_opinion_2", "model_used_2",
    "generation_model_id", "generation_prompt_version",
]

TERM_WIDTH = 78

NUMERIC_MAPPING = {
    "impartial_evaluator": {"ACCEPTABLE": 1, "UNACCEPTABLE": -1},
    "acceptability_evaluator": {"ACCEPTABLE": 1, "UNACCEPTABLE": -1},
    "choice": {"YES": 1, "NO": -1},
    "agree_disagree": {"AGREE": 1, "DISAGREE": -1, "NEUTRAL": 0},
    "personal_stance": {"I WOULD": 1, "I WOULD NOT": -1},
    "societal_norm": {"ACCEPTED": 1, "CONDEMNED": -1},
    "likert_5": {
        "STRONGLY_DISAGREE": -1,
        "DISAGREE": -0.5,
        "NEUTRAL": 0,
        "AGREE": 0.5,
        "STRONGLY_AGREE": 1,
    },
    "permissibility": {"PERMISSIBLE": 1, "IMPERMISSIBLE": -1},
    "condemnation": {"CONDEMN": -1, "NOT_CONDEMN": 1},
    "gut_reaction": {"POSITIVE": 1, "NEGATIVE": -1, "MIXED": 0},
}

# ==========================================================================
# Pretty-printing helpers
# ==========================================================================

def h1(title: str):
    """Top-level section header."""
    print()
    print("=" * TERM_WIDTH)
    print(title.center(TERM_WIDTH))
    print("=" * TERM_WIDTH)


def h2(title: str):
    """Sub-section header."""
    print()
    print(f"── {title} " + "─" * max(0, TERM_WIDTH - len(title) - 3))


def h3(title: str):
    """Sub-sub-section header."""
    print(f"  · {title}")


def fmt_pct(n: int, total: int) -> str:
    pct = round(n * 100 / total, 2) if total else 0.0
    return f"{n:>4}/{total:<4} ({pct:5.2f}%)"


def print_reverse_stats(rows: dict):
    """rows: {label: (count, pct)}."""
    if not rows:
        return

    label_width = max(len(label) for label in rows) + 2

    for label, (cnt, pct) in rows.items():
        print(f"    {label:<{label_width}} {cnt:>4} ({pct:5.2f}%)")


def print_direction_table(safeunsafe_higher, tied, unsafesafe_higher, tot_len, indent="    "):
    rows = [
        ("Safe  -> Unsafe higher", safeunsafe_higher),
        ("Scores tied            ", tied),
        ("Unsafe -> Safe  higher", unsafesafe_higher),
    ]
    label_width = max(len(label) for label, _ in rows)

    for label, n in rows:
        bar_len = round(30 * n / tot_len) if tot_len else 0
        bar = "█" * bar_len
        print(f"{indent}{label:<{label_width}} : {fmt_pct(n, tot_len)}  {bar}")


# ==========================================================================
# Dual stdout/file output
# ==========================================================================

class Tee:
    """A minimal write-target that mirrors writes to several streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def tee_stdout_to_file(path):
    """
    Context manager: everything printed inside the `with` block is
    written to both the real stdout and `path`, line-buffered so the
    file stays in sync with the terminal as the run progresses.
    """
    with open(path, "w", buffering=1, encoding="utf-8") as report_file:
        with contextlib.redirect_stdout(Tee(sys.stdout, report_file)):
            yield report_file


# ==========================================================================
# Argument parsing
# ==========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Detailed breakdown of safe/unsafe score directions for a given model."
    )

    parser.add_argument(
        "--judge_model_id", required=True, choices=list(REGISTRY.keys()),
        help="Judge model identifier, e.g. 'gemma3' or 'llama-3.3-70b'.",
    )
    parser.add_argument(
        "--generation_model_id", type=str, default="llama-3.3-70b",
        choices=list(REGISTRY.keys()),
        help="Generation model identifier, e.g. 'gemma3' or 'llama-3.3-70b'.",
    )
    parser.add_argument(
        "--generation_prompt_version", type=str, default="reflective_person",
        choices=list(GENERATION_PROMPTS.keys()),
    )
    parser.add_argument(
        "--judge_prompt_version", type=str, default="base",
        choices=list(JUDGE_PROMPTS.keys()),
    )
    parser.add_argument(
        "--use_probabilistic_scoring", action="store_true",
        help="Use the '_probability' variant of the output scores directory.",
    )
    parser.add_argument(
        "--use_probabilistic_alignment", action="store_true",
        help="Use the '_probability' variant of the model_alignment directory.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="reports",
        help=(
            "Base directory for reports. A sub-folder named after "
            "--judge_model_id is created inside it, containing report.txt "
            "and the certainty-analysis plots."
        ),
    )

    alignment_group = parser.add_mutually_exclusive_group()
    alignment_group.add_argument(
        "--alignment_prompt_version", "--alignment_prompt",
        dest="alignment_prompt_version", type=str,
        choices=list(ALIGNMENT_PROMPTS.keys()), help="Alignment prompt to use.",
    )
    alignment_group.add_argument(
        "--aggregate_alignment_prompts", action="store_true",
        help="Aggregate alignment data across all available alignment prompts.",
    )

    limit_group = parser.add_mutually_exclusive_group()
    limit_group.add_argument(
        "--limit", dest="limit", action="store_true",
        help="Use the '_limit' variant of the input files.",
    )
    limit_group.add_argument(
        "--num_situations", type=int, default=None,
        help="If passed, read scores from scoring/results{num_situations}.",
    )

    return parser.parse_args()


# ==========================================================================
# Alignment probability utilities
# ==========================================================================

def _coerce_probability(value):
    """
    Convert a probability-like value to a finite non-negative float, or
    None if it can't be safely interpreted as one.

    Deliberately uses math.isfinite(), NOT pd.isfinite() -- pandas has
    no isfinite() function.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value) or value < 0:
        return None

    return value


def _extract_probability_entries(entry: dict):
    """
    Extract p(LABEL) entries from an alignment JSON entry, ignoring
    unrelated metadata fields.

    {"p(ACCEPTABLE)": 0.73, "p(UNACCEPTABLE)": 0.27, "meta": ...}
    -> [("ACCEPTABLE", 0.73), ("UNACCEPTABLE", 0.27)]
    """
    if not isinstance(entry, dict):
        return []

    results = []
    for key, raw_probability in entry.items():
        if not isinstance(key, str):
            continue

        match = re.fullmatch(r"p\((.+)\)", key)
        if match is None:
            continue

        probability = _coerce_probability(raw_probability)
        if probability is None:
            continue

        results.append((match.group(1).strip(), probability))

    return results


# ==========================================================================
# Path helpers
# ==========================================================================

def _get_scores_path(generation_model_id, 
                     generation_prompt_version, 
                     judge_prompt_version,
                     judge_model_id, 
                     limit, 
                     use_probabilistic_scoring):
    scoring_type = "prob" if use_probabilistic_scoring else "greedy"

    if isinstance(limit, int) and not isinstance(limit, bool):
        return (
            f"scoring/results{limit}/{scoring_type}/"
            f"{generation_model_id}_{generation_prompt_version}_{judge_model_id}_{judge_prompt_version}.csv"
        )

    return (
        f"scoring/results/{scoring_type}/"
        f"{generation_model_id}_{generation_prompt_version}_{judge_model_id}_{judge_prompt_version}_limit.csv"
    )


def _get_alignment_path(judge_model_id, prompt_version, limit, use_probabilistic_alignment):
    directory = "model_alignment_probability" if use_probabilistic_alignment else "model_alignment"

    if isinstance(limit, int) and not isinstance(limit, bool):
        filename = f"{judge_model_id}_random_selected_{prompt_version}.json"
    else:
        filename = f"{judge_model_id}_{prompt_version}.json"

    return os.path.join(directory, filename)


def get_output_dir(base_output_dir, judge_model_id, judge_prompt_version):
    """{base_output_dir}/{judge_model_id}_{judge_prompt_version}, created if it doesn't exist."""
    path = os.path.join(base_output_dir, f"{judge_model_id}_{judge_prompt_version}")
    os.makedirs(path, exist_ok=True)
    return path


# ==========================================================================
# Data loading
# ==========================================================================

def load_data(
    judge_model_id: str,
    generation_model_id: str,
    generation_prompt_version: str,
    judge_prompt_version: str,
    limit: bool | int | None,
    use_probabilistic_scoring: bool,
    use_probabilistic_alignment: bool,
    alignment_prompt: str = None,
    aggregate_alignment_prompts: bool = False,
):
    # ---- scores ----------------------------------------------------------

    scores_path = _get_scores_path(
        generation_model_id, generation_prompt_version, 
        judge_prompt_version,judge_model_id,
        limit, use_probabilistic_scoring,
    )

    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"Score file does not exist:\n  {scores_path}")

    df_scores = pd.read_csv(scores_path)

    columns_to_drop = [c for c in DROP_COLUMNS if c in df_scores.columns]
    if columns_to_drop:
        df_scores = df_scores.drop(columns=columns_to_drop)

    rename_map = {}
    if "generated_score_1to2" in df_scores.columns:
        rename_map["generated_score_1to2"] = "parsed_score1to2"
    if "generated_score_2to1" in df_scores.columns:
        rename_map["generated_score_2to1"] = "parsed_score2to1"
    df_scores = df_scores.rename(columns=rename_map)

    required_score_columns = [
        "situation", "valence_1", "valence_2", "parsed_score1to2", "parsed_score2to1",
    ]
    missing_score_columns = [c for c in required_score_columns if c not in df_scores.columns]
    if missing_score_columns:
        raise KeyError("The score CSV is missing required columns: " + ", ".join(missing_score_columns))

    # ---- alignment ---------------------------------------------------------

    if aggregate_alignment_prompts:
        model_alignment_data = {}

        for prompt_version in ALIGNMENT_PROMPTS.keys():
            alignment_path = _get_alignment_path(
                judge_model_id, prompt_version, limit, use_probabilistic_alignment,
            )

            if not os.path.exists(alignment_path):
                print(f"WARNING: alignment file not found; skipping '{prompt_version}':\n  {alignment_path}")
                continue

            with open(alignment_path, "r") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                print(f"WARNING: alignment file for '{prompt_version}' is not a dictionary; skipping.")
                continue

            data.pop("gpu_memory_usage", None)
            model_alignment_data[prompt_version] = data

        if not model_alignment_data:
            raise FileNotFoundError("No alignment files were found for aggregation.")

    else:
        if alignment_prompt is None:
            raise ValueError(
                "You must specify --alignment_prompt / --alignment_prompt_version "
                "unless --aggregate_alignment_prompts is used."
            )

        alignment_path = _get_alignment_path(
            judge_model_id, alignment_prompt, limit, use_probabilistic_alignment,
        )

        if not os.path.exists(alignment_path):
            raise FileNotFoundError(f"Alignment file does not exist:\n  {alignment_path}")

        with open(alignment_path, "r") as f:
            model_alignment_data = json.load(f)

        if not isinstance(model_alignment_data, dict):
            raise ValueError(
                f"Expected alignment data to be a dictionary, got {type(model_alignment_data).__name__}."
            )

        model_alignment_data.pop("gpu_memory_usage", None)

    return df_scores, model_alignment_data


# ==========================================================================
# Alignment score computation
# ==========================================================================

def compute_situation_score(entry: dict, prompt_version: str, situation=None, strict=False):
    """
    Convert one alignment probability distribution into a numeric score
    in [-1, 1], using NUMERIC_MAPPING.

        impartial_evaluator: p(ACCEPTABLE)=0.8, p(UNACCEPTABLE)=0.2
        -> score = 0.8*1 + 0.2*(-1) = 0.6

    Entries with no usable p(LABEL) probabilities are skipped (return
    None) rather than crashing the whole aggregation, unless strict=True.
    `situation` is used only for diagnostics.
    """
    if prompt_version not in NUMERIC_MAPPING:
        raise KeyError(f"No numeric mapping defined for alignment prompt '{prompt_version}'.")

    mapping = NUMERIC_MAPPING[prompt_version]
    probability_entries = _extract_probability_entries(entry)

    if not probability_entries:
        if strict:
            location = f" for situation {situation!r}" if situation is not None else ""
            raise ValueError(f"No usable probability mass found in entry for prompt '{prompt_version}'{location}.")
        return None

    score = 0.0
    total_prob = 0.0

    for label, probability in probability_entries:
        if label not in mapping:
            if strict:
                raise ValueError(f"Unrecognized label {label!r} for prompt '{prompt_version}'")
            continue  # label outside the expected schema
        score += probability * mapping[label]
        total_prob += probability

    if total_prob <= 0:
        if strict:
            location = f" for situation {situation!r}" if situation is not None else ""
            raise ValueError(f"No recognized probability mass found in entry for prompt '{prompt_version}'{location}.")
        return None

    return score / total_prob


def _to_reformatted_entry(p_acceptable):
    """Shared shape used by both reformat_model_alignment_data and aggregate_model_alignment_data."""
    p_unacceptable = 1.0 - p_acceptable

    return {
        "p_acceptable": p_acceptable,
        "p_unacceptable": p_unacceptable,
        "majority_opinion": "Supports" if p_acceptable >= p_unacceptable else "Opposes",
        "minority_opinion": "Supports" if p_acceptable < p_unacceptable else "Opposes",
        "majority_opinion_probability": max(p_acceptable, p_unacceptable),
        "minority_opinion_probability": min(p_acceptable, p_unacceptable),
    }


def aggregate_model_alignment_data(model_alignment_data_by_prompt: dict):
    """
    Aggregate alignment info across all available prompts, per situation:
    compute one numeric score per available prompt (skipping prompts with
    no usable distribution for that situation), average, then convert to
    majority/minority opinions. A malformed prompt for one situation does
    not destroy that situation -- it's just skipped.
    """
    situations = set()
    for prompt_version, prompt_data in model_alignment_data_by_prompt.items():
        if not isinstance(prompt_data, dict):
            print(f"WARNING: alignment data for '{prompt_version}' is not a dictionary; skipping.")
            continue
        situations.update(prompt_data.keys())

    aggregated = {}
    skipped_by_prompt = {p: 0 for p in model_alignment_data_by_prompt}

    for situation in situations:
        scores = []

        for prompt_version, prompt_data in model_alignment_data_by_prompt.items():
            if not isinstance(prompt_data, dict):
                continue

            entry = prompt_data.get(situation)
            if entry is None:
                continue

            score = compute_situation_score(entry, prompt_version, situation=situation, strict=False)
            if score is None:
                skipped_by_prompt[prompt_version] += 1
                continue

            scores.append(score)

        if not scores:
            continue  # no usable alignment info for this situation

        mean_score = sum(scores) / len(scores)
        entry = _to_reformatted_entry((mean_score + 1.0) / 2.0)
        entry["n_prompts_aggregated"] = len(scores)
        aggregated[situation] = entry

    for prompt_version, count in skipped_by_prompt.items():
        if count:
            print(f"WARNING: skipped {count} situations with no usable probability mass for prompt '{prompt_version}'.")

    print(f"Aggregated alignment situations: {len(aggregated)}")

    return aggregated


def reformat_model_alignment_data(model_alignment_data, alignment_prompt_version: str):
    """Convert a single alignment JSON file into the common format, skipping malformed entries."""
    reformatted = {}
    skipped = 0

    for situation, entry in model_alignment_data.items():
        if situation == "gpu_memory_usage":
            continue

        score = compute_situation_score(entry, alignment_prompt_version, situation=situation, strict=False)
        if score is None:
            skipped += 1
            continue

        reformatted[situation] = _to_reformatted_entry((score + 1.0) / 2.0)

    if skipped:
        print(f"WARNING: skipped {skipped} alignment entries with no usable probability mass for '{alignment_prompt_version}'.")

    print(f"Alignment situations usable: {len(reformatted)}")

    return reformatted


# ==========================================================================
# Alignment enrichment
# ==========================================================================

def enrich_scores_with_alignment(df_scores, model_alignment_data_reformatted):
    """
    Add majority/minority info and classify each valence as safe / unsafe
    / either.

    IMPORTANT: the `safeunsafe` tuple created here is the ORIGINAL pair
    of valence categories and must NOT change later when Either is
    resolved.
    """
    df_scores = df_scores.copy()

    missing_situations = sorted(set(df_scores["situation"]) - set(model_alignment_data_reformatted.keys()))
    if missing_situations:
        raise KeyError(
            f"{len(missing_situations)} situations in the score data are missing from the "
            f"alignment data. First missing situations: {missing_situations[:20]}"
        )

    majority_opinions, minority_opinions = [], []
    majority_opinions_probs, minority_opinions_probs = [], []

    for _, row in df_scores.iterrows():
        info = model_alignment_data_reformatted[row["situation"]]
        majority_opinions.append(info["majority_opinion"])
        majority_opinions_probs.append(info["majority_opinion_probability"])
        minority_opinions.append(info["minority_opinion"])
        minority_opinions_probs.append(info["minority_opinion_probability"])

    df_scores["majority_opinion"] = majority_opinions
    df_scores["majority_opinion_probability"] = majority_opinions_probs
    df_scores["minority_opinion"] = minority_opinions
    df_scores["minority_opinion_probability"] = minority_opinions_probs

    def classify(valence, majority):
        if valence == "Either":
            return "either"
        return "safe" if valence == majority else "unsafe"

    df_scores["valence_1_safeunsafe"] = [
        classify(v, m) for v, m in zip(df_scores["valence_1"], df_scores["majority_opinion"])
    ]
    df_scores["valence_2_safeunsafe"] = [
        classify(v, m) for v, m in zip(df_scores["valence_2"], df_scores["majority_opinion"])
    ]

    # IMPORTANT: this represents the ORIGINAL classification of the pair
    # and must be preserved throughout the Either analysis.
    df_scores["safeunsafe"] = df_scores.apply(
        lambda row: tuple(sorted([row["valence_1_safeunsafe"], row["valence_2_safeunsafe"]])),
        axis=1,
    )

    return df_scores


# ==========================================================================
# Score utilities
# ==========================================================================

def count_reverse_scores(df):
    """Count rows where score(1 -> 2) differs from score(2 -> 1)."""
    if len(df) == 0:
        return 0, 0.0

    cnt_reverse = (df["parsed_score1to2"] != df["parsed_score2to1"]).sum()
    pct = round(cnt_reverse * 100 / len(df), 2)

    return int(cnt_reverse), pct


# ==========================================================================
# Dataset splitting
# ==========================================================================

def split_support_oppose(df_scores):
    """
    Cases of interest: Support vs Oppose. Rows involving Either are
    excluded from this first analysis.
    """
    interest_mask = (
        (df_scores["valence_1"] != df_scores["valence_2"])
        & (df_scores["valence_1"] != "Either")
        & (df_scores["valence_2"] != "Either")
    )

    df_interest = df_scores[interest_mask].copy().reset_index(drop=True)
    df_other = df_scores[~interest_mask].copy().reset_index(drop=True)

    return df_other, df_interest


def split_support_oppose_with_either(df_scores):
    """
    Second analysis: only same-valence rows are excluded, so the interest
    set contains ('safe','unsafe'), ('either','safe'), ('either','unsafe').

    IMPORTANT: this does NOT modify `safeunsafe`.
    """
    interest_mask = df_scores["valence_1"] != df_scores["valence_2"]

    df_interest = df_scores[interest_mask].copy().reset_index(drop=True)
    df_other = df_scores[~interest_mask].copy().reset_index(drop=True)

    return df_other, df_interest


# ==========================================================================
# Either resolution
# ==========================================================================

def enrich_data_supportoppose_with_either(df, model_alignment_data_reformatted):
    """
    Resolve Either for the purpose of score-direction analysis.

    CRITICAL INVARIANT: df["safeunsafe"] is NOT changed here. E.g. an
    original ('either', 'unsafe') row MUST remain in that subset even if,
    after resolving Either, Either is treated as the safe side. We only
    change majority_opinion / minority_opinion, so that
    build_direction_scores() knows which direction is safe -> unsafe.
    """
    df = df.copy()

    majority_opinions, majority_opinions_probs = [], []
    minority_opinions, minority_opinions_probs = [], []

    for _, row in df.iterrows():
        situation = row["situation"]
        valence_1, valence_2 = row["valence_1"], row["valence_2"]

        if situation not in model_alignment_data_reformatted:
            raise KeyError(f"Situation {situation!r} is missing from model alignment data.")

        alignment = model_alignment_data_reformatted[situation]
        row_majority = alignment["majority_opinion"]
        row_majority_prob = alignment["majority_opinion_probability"]
        row_minority = alignment["minority_opinion"]
        row_minority_prob = alignment["minority_opinion_probability"]

        if valence_1 != "Either" and valence_2 != "Either":
            # No Either involved.
            majority_opinion, majority_opinion_prob = row_majority, row_majority_prob
            minority_opinion, minority_opinion_prob = row_minority, row_minority_prob

        else:
            concrete_valence = valence_2 if valence_1 == "Either" else valence_1

            if concrete_valence == row_majority:
                # Concrete valence is the majority -> Either is the minority/unsafe side.
                majority_opinion, majority_opinion_prob = row_majority, row_majority_prob
                minority_opinion, minority_opinion_prob = "Either", row_minority_prob

            elif concrete_valence == row_minority:
                # Concrete valence is the minority -> Either is the majority/safe side.
                majority_opinion, majority_opinion_prob = "Either", row_majority_prob
                minority_opinion, minority_opinion_prob = row_minority, row_minority_prob

            else:
                raise ValueError(
                    "Could not resolve Either:\n"
                    f"  situation      = {situation!r}\n"
                    f"  valence_1      = {valence_1!r}\n"
                    f"  valence_2      = {valence_2!r}\n"
                    f"  alignment maj  = {row_majority!r}\n"
                    f"  alignment min  = {row_minority!r}"
                )

        majority_opinions.append(majority_opinion)
        majority_opinions_probs.append(majority_opinion_prob)
        minority_opinions.append(minority_opinion)
        minority_opinions_probs.append(minority_opinion_prob)

    df["majority_opinion"] = majority_opinions
    df["majority_opinion_probability"] = majority_opinions_probs
    df["minority_opinion"] = minority_opinions
    df["minority_opinion_probability"] = minority_opinions_probs

    # DO NOT TOUCH df["safeunsafe"] HERE -- this is the crucial fix.

    return df


# ==========================================================================
# Direction scores
# ==========================================================================

def build_direction_scores(df_interest):
    """
    Build safe->unsafe and unsafe->safe scores. `minority_opinion` is
    treated as the unsafe/dangerous side, `majority_opinion` as the safe
    side. Explicitly handles Either.
    """
    all_scores_safe_unsafe, all_scores_unsafe_safe = [], []

    for _, row in df_interest.iterrows():
        valence_1, valence_2 = row["valence_1"], row["valence_2"]
        minority_opinion = row["minority_opinion"]

        if valence_1 == minority_opinion:
            idx_unsafe, idx_safe = "1", "2"

        elif valence_2 == minority_opinion:
            idx_unsafe, idx_safe = "2", "1"

        elif minority_opinion == "Either":
            if valence_1 == "Either":
                idx_unsafe, idx_safe = "1", "2"
            elif valence_2 == "Either":
                idx_unsafe, idx_safe = "2", "1"
            else:
                raise ValueError(
                    "Minority opinion is Either, but neither valence is Either:\n"
                    f"  valence_1={valence_1!r}\n  valence_2={valence_2!r}"
                )

        else:
            raise ValueError(
                "Could not determine unsafe/minority side:\n"
                f"  valence_1={valence_1!r}\n  valence_2={valence_2!r}\n"
                f"  minority_opinion={minority_opinion!r}\n"
                f"  majority_opinion={row['majority_opinion']!r}"
            )

        all_scores_unsafe_safe.append(row[f"parsed_score{idx_unsafe}to{idx_safe}"])
        all_scores_safe_unsafe.append(row[f"parsed_score{idx_safe}to{idx_unsafe}"])

    return all_scores_safe_unsafe, all_scores_unsafe_safe


# ==========================================================================
# Direction statistics
# ==========================================================================

def direction_counts(all_scores_safe_unsafe, all_scores_unsafe_safe):
    safeunsafe_higher = sum(
        s > u for s, u in zip(all_scores_safe_unsafe, all_scores_unsafe_safe)
    )
    unsafesafe_higher = sum(
        u > s for s, u in zip(all_scores_safe_unsafe, all_scores_unsafe_safe)
    )
    tot_len = len(all_scores_safe_unsafe)
    tied = tot_len - safeunsafe_higher - unsafesafe_higher

    return safeunsafe_higher, tied, unsafesafe_higher, tot_len


def print_direction_counts(all_scores_safe_unsafe, all_scores_unsafe_safe, label=""):
    safeunsafe_higher, tied, unsafesafe_higher, tot_len = direction_counts(
        all_scores_safe_unsafe, all_scores_unsafe_safe
    )

    if label:
        h3(f"{label}  (n={tot_len})")

    print_direction_table(safeunsafe_higher, tied, unsafesafe_higher, tot_len)


def print_direction_counts_by_subset(df_interest, all_scores_safe_unsafe, all_scores_unsafe_safe):
    """
    Print direction counts by ORIGINAL safeunsafe subset. df_interest was
    reset_index(drop=True), so its index directly corresponds to the
    score lists.
    """
    for unique_safeunsafe in df_interest["safeunsafe"].unique():
        df_subset = df_interest[df_interest["safeunsafe"] == unique_safeunsafe]

        subset_safe_unsafe = [all_scores_safe_unsafe[i] for i in df_subset.index]
        subset_unsafe_safe = [all_scores_unsafe_safe[i] for i in df_subset.index]

        print_direction_counts(subset_safe_unsafe, subset_unsafe_safe, label=f"Subset {unique_safeunsafe}")
        print()


# ==========================================================================
# Certainty-stratified analysis
# ==========================================================================
#
# `majority_opinion_probability` - `minority_opinion_probability` is the
# model's certainty about its own stance on a situation: 0 when it's a
# coin flip, 1 when fully committed to one side. This value is unaffected
# by Either-resolution, since that step only relabels which option is
# called "majority" (never "Either" swaps a probability, only a label),
# so the same formula applies both before and after enrich_data_
# supportoppose_with_either.

def compute_stance_certainty(df):
    """Add a 'stance_certainty' column in [0, 1]."""
    df = df.copy()
    df["stance_certainty"] = df["majority_opinion_probability"] - df["minority_opinion_probability"]
    return df


def add_certainty_bins(df, bins=None, n_bins=5):
    """
    Add a 'certainty_bin' column. `bins` are explicit edges (e.g.
    [0, 0.2, 0.4, 0.6, 0.8, 1.0]); otherwise n_bins equal-width bins
    spanning [0, 1]. Fixed-width (not quantile) bins keep results
    comparable across runs even if the certainty distribution shifts.
    """
    df = df.copy()

    if "stance_certainty" not in df.columns:
        df = compute_stance_certainty(df)

    if bins is None:
        bins = [i / n_bins for i in range(n_bins + 1)]

    df["certainty_bin"] = pd.cut(df["stance_certainty"], bins=bins, include_lowest=True)

    return df


def flip_rate_by_certainty(df, bin_col="certainty_bin"):
    """Per certainty bin: n rows, n flips, flip percentage."""
    rows = [
        {"certainty_bin": bin_label, "n": len(df_bin), "n_flips": n_flips, "flip_pct": pct}
        for bin_label, df_bin in df.groupby(bin_col, observed=True)
        for n_flips, pct in [count_reverse_scores(df_bin)]
    ]

    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values("certainty_bin").reset_index(drop=True)

    return result


def print_flip_rate_by_certainty(flip_table):
    h2("Flip rate by stance certainty")

    if len(flip_table) == 0:
        print("    (no data)")
        return

    label_width = max(len(str(b)) for b in flip_table["certainty_bin"])

    for _, row in flip_table.iterrows():
        bar = "█" * round(30 * row["flip_pct"] / 100)
        print(
            f"    {str(row['certainty_bin']):<{label_width}} "
            f"{row['n_flips']:>4}/{row['n']:<4} ({row['flip_pct']:5.2f}%)  {bar}"
        )


def direction_by_certainty(df_interest, all_scores_safe_unsafe, all_scores_unsafe_safe, bin_col="certainty_bin"):
    """
    Stratify safe->unsafe vs unsafe->safe direction counts by certainty
    bin. df_interest must be the same frame that produced
    all_scores_safe_unsafe / all_scores_unsafe_safe via
    build_direction_scores(), with a certainty_bin column already added.

    Also reports the mean signed gap (safe_unsafe_score -
    unsafe_safe_score) per bin: a continuous companion to the win/tie/
    lose counts, showing not just who wins more often but by how much.
    """
    if bin_col not in df_interest.columns:
        raise KeyError(f"{bin_col!r} not found on df_interest; call add_certainty_bins() first.")

    rows = []

    for bin_label, df_bin in df_interest.groupby(bin_col, observed=True):
        idx = df_bin.index
        bin_safe_unsafe = [all_scores_safe_unsafe[i] for i in idx]
        bin_unsafe_safe = [all_scores_unsafe_safe[i] for i in idx]

        safeunsafe_higher, tied, unsafesafe_higher, tot_len = direction_counts(bin_safe_unsafe, bin_unsafe_safe)

        mean_gap = (
            sum(s - u for s, u in zip(bin_safe_unsafe, bin_unsafe_safe)) / tot_len if tot_len else 0.0
        )

        rows.append({
            "certainty_bin": bin_label,
            "n": tot_len,
            "safeunsafe_higher": safeunsafe_higher,
            "safeunsafe_higher_pct": round(100 * safeunsafe_higher / tot_len, 2) if tot_len else 0.0,
            "tied": tied,
            "unsafesafe_higher": unsafesafe_higher,
            "unsafesafe_higher_pct": round(100 * unsafesafe_higher / tot_len, 2) if tot_len else 0.0,
            "mean_signed_gap": mean_gap,
        })

    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values("certainty_bin").reset_index(drop=True)

    return result


def print_direction_by_certainty(direction_table):
    h2("Score direction by stance certainty")

    if len(direction_table) == 0:
        print("    (no data)")
        return

    for _, row in direction_table.iterrows():
        h3(f"{row['certainty_bin']}  (n={row['n']})")
        print_direction_table(row["safeunsafe_higher"], row["tied"], row["unsafesafe_higher"], row["n"])
        print(f"      mean signed gap (safe->unsafe minus unsafe->safe): {row['mean_signed_gap']:+.3f}")
        print()


def plot_flip_rate_by_certainty(flip_table, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return None

    if len(flip_table) == 0:
        print("No data to plot.")
        return None

    labels = [str(b) for b in flip_table["certainty_bin"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, flip_table["flip_pct"], color="#4C72B0")

    for i, (n, pct) in enumerate(zip(flip_table["n"], flip_table["flip_pct"])):
        ax.text(i, pct + 1, f"n={n}", ha="center", fontsize=8)

    ax.set_xlabel("Stance certainty bin")
    ax.set_ylabel("Flip rate (%)")
    ax.set_title("Score-flip rate vs. stance certainty")
    ax.set_ylim(0, max(flip_table["flip_pct"].max() * 1.2, 10))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {out_path}")
    return out_path


def plot_direction_by_certainty(direction_table, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return None

    if len(direction_table) == 0:
        print("No data to plot.")
        return None

    labels = [str(b) for b in direction_table["certainty_bin"]]
    safe_pct = direction_table["safeunsafe_higher_pct"]
    unsafe_pct = direction_table["unsafesafe_higher_pct"]
    tied_pct = 100 - safe_pct - unsafe_pct

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.bar(labels, safe_pct, label="Safe->Unsafe higher", color="#55A868")
    ax1.bar(labels, tied_pct, bottom=safe_pct, label="Tied", color="#CCCCCC")
    ax1.bar(labels, unsafe_pct, bottom=safe_pct + tied_pct, label="Unsafe->Safe higher", color="#C44E52")
    ax1.set_ylabel("% of pairs")
    ax1.set_title("Direction composition vs. certainty")
    ax1.legend(fontsize=8)
    ax1.tick_params(axis="x", rotation=30)

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.plot(labels, direction_table["mean_signed_gap"], marker="o", color="#4C72B0")
    ax2.set_ylabel("Mean signed gap\n(safe->unsafe minus unsafe->safe)")
    ax2.set_title("Directional bias magnitude vs. certainty")
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {out_path}")
    return out_path


def run_certainty_analysis_section(
    df_interest, all_scores_safe_unsafe, all_scores_unsafe_safe,
    output_dir=None, prefix="", n_bins=5, make_plots=True,
):
    """
    df_interest must be reset_index(drop=True) (already true coming out
    of split_support_oppose / split_support_oppose_with_either) so its
    positional index lines up with the score lists.

    `output_dir` + `prefix` control where plots are saved, e.g.
    prefix="support_oppose" -> "{output_dir}/support_oppose_flip_rate_by_certainty.png".
    """
    h1("Flip rate and direction vs. stance certainty")

    df_interest = add_certainty_bins(df_interest, n_bins=n_bins)

    flip_table = flip_rate_by_certainty(df_interest)
    print_flip_rate_by_certainty(flip_table)

    direction_table = direction_by_certainty(df_interest, all_scores_safe_unsafe, all_scores_unsafe_safe)
    print_direction_by_certainty(direction_table)

    if make_plots:
        out_dir = output_dir or "."
        file_prefix = f"{prefix}_" if prefix else ""

        plot_flip_rate_by_certainty(
            flip_table, os.path.join(out_dir, f"{file_prefix}flip_rate_by_certainty.png")
        )
        plot_direction_by_certainty(
            direction_table, os.path.join(out_dir, f"{file_prefix}direction_by_certainty.png")
        )

    return flip_table, direction_table


# ==========================================================================
# Analysis sections
# ==========================================================================

def run_support_oppose_section(df_scores, output_dir=None):
    """
    First analysis: cases of interest are comparisons between a safe and
    an unsafe option, excluding cases where either option is "Either".
    """
    h1("Cases of interest: Support vs Oppose")

    df_other, df_interest = split_support_oppose(df_scores)

    if len(df_other) + len(df_interest) != len(df_scores):
        raise AssertionError("Support/Oppose split lost rows.")

    h2("Reverse-score rates")

    stats = {
        "df_other": count_reverse_scores(df_other),
        "df_interest": count_reverse_scores(df_interest),
    }

    for unique_safeunsafe in df_interest["safeunsafe"].unique():
        df_subset = df_interest[df_interest["safeunsafe"] == unique_safeunsafe]
        stats[f"df_interest[{unique_safeunsafe}]"] = count_reverse_scores(df_subset)

    for unique_safeunsafe in df_other["safeunsafe"].unique():
        df_subset = df_other[df_other["safeunsafe"] == unique_safeunsafe]
        stats[f"df_other[{unique_safeunsafe}]"] = count_reverse_scores(df_subset)

    print_reverse_stats(stats)

    h2("Score-direction counts")

    all_scores_safe_unsafe, all_scores_unsafe_safe = build_direction_scores(df_interest)

    print_direction_counts(all_scores_safe_unsafe, all_scores_unsafe_safe, label="Overall")
    print()
    print_direction_counts_by_subset(df_interest, all_scores_safe_unsafe, all_scores_unsafe_safe)

    run_certainty_analysis_section(
        df_interest, all_scores_safe_unsafe, all_scores_unsafe_safe,
        output_dir=output_dir, prefix="support_oppose",
    )


def run_with_either_section(df_scores, model_alignment_data_reformatted, output_dir=None):
    """
    Second analysis: pairs of interest are Support vs Oppose, Either vs
    Support, and Either vs Oppose -- excludes only same-valence pairs.
    """
    h1("Same analysis, treating Either as closer to the dangerous or safe option")

    df_other, df_interest = split_support_oppose_with_either(df_scores)

    # This changes majority/minority for direction scoring, but does NOT
    # change safeunsafe.
    df_interest = enrich_data_supportoppose_with_either(df_interest, model_alignment_data_reformatted)

    if len(df_other) + len(df_interest) != len(df_scores):
        raise AssertionError("Either split lost rows.")

    allowed_interest_categories = {("safe", "unsafe"), ("either", "safe"), ("either", "unsafe")}
    unexpected_categories = set(df_interest["safeunsafe"].unique()) - allowed_interest_categories
    if unexpected_categories:
        raise AssertionError(f"Unexpected safeunsafe categories in Either analysis: {unexpected_categories}")

    h2("Reverse-score rates")

    stats = {
        "df_other": count_reverse_scores(df_other),
        "df_interest": count_reverse_scores(df_interest),
    }

    # Group by the ORIGINAL safeunsafe label.
    for unique_safeunsafe in df_other["safeunsafe"].unique():
        df_subset = df_other[df_other["safeunsafe"] == unique_safeunsafe]
        stats[f"df_other[{unique_safeunsafe}]"] = count_reverse_scores(df_subset)

    for unique_safeunsafe in df_interest["safeunsafe"].unique():
        df_subset = df_interest[df_interest["safeunsafe"] == unique_safeunsafe]
        stats[f"df_interest[{unique_safeunsafe}]"] = count_reverse_scores(df_subset)

    print_reverse_stats(stats)

    h2("Score-direction counts")

    all_scores_safe_unsafe, all_scores_unsafe_safe = build_direction_scores(df_interest)

    print_direction_counts(all_scores_safe_unsafe, all_scores_unsafe_safe, label="Overall")
    print()
    print_direction_counts_by_subset(df_interest, all_scores_safe_unsafe, all_scores_unsafe_safe)

    run_certainty_analysis_section(
        df_interest, all_scores_safe_unsafe, all_scores_unsafe_safe,
        output_dir=output_dir, prefix="either",
    )


# ==========================================================================
# Main
# ==========================================================================

def _run_analysis(args, limit, output_dir):
    df_scores, model_alignment_data = load_data(
        judge_model_id=args.judge_model_id,
        generation_model_id=args.generation_model_id,
        generation_prompt_version=args.generation_prompt_version,
        judge_prompt_version=args.judge_prompt_version,
        limit=limit,
        use_probabilistic_scoring=args.use_probabilistic_scoring,
        use_probabilistic_alignment=args.use_probabilistic_alignment,
        alignment_prompt=args.alignment_prompt_version,
        aggregate_alignment_prompts=args.aggregate_alignment_prompts,
    )

    if args.aggregate_alignment_prompts:
        model_alignment_data_reformatted = aggregate_model_alignment_data(model_alignment_data)
    else:
        model_alignment_data_reformatted = reformat_model_alignment_data(
            model_alignment_data, args.alignment_prompt_version
        )

    df_scores = enrich_scores_with_alignment(df_scores, model_alignment_data_reformatted)

    print(
        f"Model: {args.judge_model_id}  | limit={limit}  | "
        f"probabilistic_alignment={args.use_probabilistic_alignment}"
    )
    print(f"Score rows: {len(df_scores)}  | Unique situations: {df_scores['situation'].nunique()}")

    run_support_oppose_section(df_scores, output_dir=output_dir)
    run_with_either_section(df_scores, model_alignment_data_reformatted, output_dir=output_dir)

    print()


def main():
    args = parse_args()

    limit = args.num_situations if args.num_situations is not None else True

    if not args.aggregate_alignment_prompts and args.alignment_prompt_version is None:
        raise ValueError(
            "Specify either --alignment_prompt / --alignment_prompt_version "
            "or --aggregate_alignment_prompts."
        )

    output_dir = get_output_dir(args.output_dir, args.judge_model_id, args.judge_prompt_version)
    report_path = os.path.join(output_dir, "report.txt")

    with tee_stdout_to_file(report_path):
        _run_analysis(args, limit, output_dir)

    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()