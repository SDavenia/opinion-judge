"""
Analyse stability (RS) and positional stability (PC) of judge-model scores as a
function of various characteristics of the scored pairs (text, perplexity,
judge-model alignment, ...).

For each analysis type in ANALYSIS_REGISTRY, and for every judge model found in
--scoring_dir, the corresponding scoring CSV is enriched with the analysis's
characteristic columns (either computed in this script directly, e.g. text
length, or read from the relevant results folder, e.g. perplexity/alignment),
binned along each characteristic, and RS/PC are computed per bin.

Output layout:
    {output_dir}/{dataset_id}/{analysis_type}/{characteristic}.png
                                                              - RS & PC vs. the
                                                                characteristic,
                                                                one line per
                                                                judge model
    {output_dir}/{dataset_id}/{analysis_type}/{judge_model_id}/{characteristic}.csv
                                                              - per-bin RS/PC
                                                                for that model

--dataset_id selects both the input dataset (habermas or valueprism) and the
{dataset_id} subdirectory read from --scoring_dir/--perplexity_dir/--alignment_dir
and written under --output_dir (matching score.py / extract_model_embedded_opinion.py
/ extract_model_generation_perplexity.py's output layout). Habermas opinions are
human-written, so --generator_model_id/--generator_prompt_version (which only name
a generation-CSV lookup) are ignored for it.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.evaluation_utils import calculate_pc_from_df, calculate_rs_from_df
from utils.prompts_utils import LABEL_MAPPING, CATEGORY_MAPPING

SCORING_DF_COLUMNS = [
    "id_1", "situation_id_1", "text_id_1", "stance_1", "text_1",
    "id_2", "situation_id_2", "text_id_2", "stance_2", "text_2",
    "parsed_score_1to2", "parsed_score_2to1",
]


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    # General args
    parser.add_argument(
        "--dataset_id", type=str, required=True, choices=["habermas", "valueprism"],
        help="Which dataset's scoring/perplexity/alignment results to analyse. Selects the "
             "{dataset_id} subdirectory under --scoring_dir/--perplexity_dir/--alignment_dir "
             "and --output_dir.",
    )
    # Only used for --dataset_id=valueprism (habermas scoring files have no
    # generator prefix, since the opinions are human-written).
    parser.add_argument("--generator_model_id", type=str, default="llama-3.3-70b")
    parser.add_argument("--generator_prompt_version", type=str, default="reflective_person")
    parser.add_argument("--scoring_prompt_version", type=str, default="0_1")

    parser.add_argument("--generation_dir", type=Path, default="generations/")
    parser.add_argument("--scoring_dir", type=Path, default="scoring/")

    parser.add_argument("--output_dir", type=Path, default="analysis_results/")

    # Binning args (shared by all analyses)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--bin_method", type=str, default="quantile", choices=["quantile", "equal_width"])

    # Text-characteristics args
    # (no extra args needed - computed directly from the scoring dataframe)

    # Emotion characteristics args

    # llama-guard generation args
    parser.add_argument("--llama_guard_dir", type=Path, default="safety_scores/")

    # ...

    # Perplexity analysis args
    parser.add_argument("--perplexity_dir", type=Path, default="model_alignment_perplexity/")
    parser.add_argument("--perplexity_prompt_version", type=str, default="base")

    # Alignment analysis args
    parser.add_argument("--alignment_dir", type=Path, default="model_alignment/")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _scoring_prefix(args) -> str:
    """Habermas scoring files have no generator prefix (human-written opinions,
    see score.py's resolve_output_path); valueprism ones do."""
    if args.dataset_id == "habermas":
        return ""
    return f"{args.generator_model_id}_{args.generator_prompt_version}_"


def discover_judge_model_ids(args) -> list[str]:
    """Judge models are whatever scoring CSVs exist for this generator/prompt,
    rather than a hardcoded list, so newly-scored judges are picked up automatically."""
    prefix = _scoring_prefix(args)
    suffix = f"_{args.scoring_prompt_version}.csv"
    judge_model_ids = []
    for f in sorted(args.scoring_dir.glob(f"{prefix}*{suffix}")):
        judge_model_id = f.name[len(prefix):-len(suffix)]
        if judge_model_id:
            judge_model_ids.append(judge_model_id)
    return judge_model_ids


def load_scoring_df(args, judge_model_id: str) -> pd.DataFrame | None:
    path = args.scoring_dir / f"{_scoring_prefix(args)}{judge_model_id}_{args.scoring_prompt_version}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)[SCORING_DF_COLUMNS]


# ---------------------------------------------------------------------------
# Binning + stability
# ---------------------------------------------------------------------------

def bin_and_compute_stability(df: pd.DataFrame, x_col: str, n_bins: int, method: str) -> pd.DataFrame:
    """Bin df by x_col and compute RS/PC within each bin."""
    df = df.dropna(subset=[x_col]).copy()
    if df.empty:
        return pd.DataFrame(columns=["bin", "x_mean", "x_min", "x_max", "n_pairs", "rs", "pc"])

    if method == "quantile":
        ranks = df[x_col].rank(method="first")  # breaks ties -> exact equal-count bins
        n_bins_eff = max(1, min(n_bins, ranks.nunique()))
        df["_bin"] = pd.qcut(ranks, q=n_bins_eff, labels=False)
    elif method == "equal_width":
        df["_bin"] = pd.cut(df[x_col], bins=n_bins, labels=False)
    else:
        raise ValueError(f"Unknown bin_method: {method}")

    records = []
    for bin_id, bin_df in df.groupby("_bin"):
        if bin_df.empty:
            continue
        records.append({
            "bin": int(bin_id),
            "x_mean": bin_df[x_col].mean(),
            "x_min": bin_df[x_col].min(),
            "x_max": bin_df[x_col].max(),
            "n_pairs": len(bin_df),
            "rs": calculate_rs_from_df(bin_df),
            "pc": calculate_pc_from_df(bin_df),
        })
    return pd.DataFrame.from_records(records).sort_values("x_mean").reset_index(drop=True)


def plot_stability_across_models(summaries: dict[str, pd.DataFrame], xlabel: str, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for judge_model_id, summary in summaries.items():
        if summary.empty:
            continue
        axes[0].plot(summary["x_mean"], summary["rs"], marker="o", label=judge_model_id)
        axes[1].plot(summary["x_mean"], summary["pc"], marker="o", label=judge_model_id)

    axes[0].set_title("Stability (RS)", fontsize=10)
    axes[1].set_title("Positional stability (PC)", fontsize=10)
    for ax, ylabel in zip(axes, ["RS", "PC"]):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(xlabel)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Text-characteristics analysis ([GEN] text length) - self-contained, computed
# directly from the scoring dataframe's text_1/text_2 columns.
# ---------------------------------------------------------------------------

def enrich_text(scoring_df: pd.DataFrame, args, judge_model_id: str) -> pd.DataFrame | None:
    df = scoring_df.copy()
    df["text_1_len"] = df["text_1"].str.split().str.len()
    df["text_2_len"] = df["text_2"].str.split().str.len()
    df["avg_text_length"] = (df["text_1_len"] + df["text_2_len"]) / 2
    df["max_text_length"] = df[["text_1_len", "text_2_len"]].max(axis=1)
    df["min_text_length"] = df[["text_1_len", "text_2_len"]].min(axis=1)
    df["diff_text_length"] = (df["text_1_len"] - df["text_2_len"]).abs()
    return df


TEXT_CHARACTERISTICS = {
    "avg_text_length": "Average text length (words)",
    "max_text_length": "Maximum text length (words)",
    "min_text_length": "Minimum text length (words)",
    "diff_text_length": "Text length difference (words)",
}


# ---------------------------------------------------------------------------
# Perplexity analysis ([GEN] perplexity of the generator's text, under the
# judge model) - read from --perplexity_dir.
# ---------------------------------------------------------------------------

def load_perplexity_df(args, judge_model_id: str) -> pd.DataFrame | None:
    if args.dataset_id == "habermas":
        # No generator subdir -- habermas opinions are human-written, see
        # extract_model_generation_perplexity.py's output layout.
        path = args.perplexity_dir / judge_model_id / f"{args.perplexity_prompt_version}.csv"
    else:
        path = (
            args.perplexity_dir / judge_model_id
            / f"{args.generator_model_id}_{args.generator_prompt_version}"
            / f"{args.perplexity_prompt_version}.csv"
        )
    if not path.exists():
        return None
    return pd.read_csv(path)[["text_id", "perplexity"]]


def enrich_perplexity(scoring_df: pd.DataFrame, args, judge_model_id: str) -> pd.DataFrame | None:
    perplexity_df = load_perplexity_df(args, judge_model_id)
    if perplexity_df is None:
        return None

    df = scoring_df.copy()
    df = df.merge(
        perplexity_df.rename(columns={"text_id": "text_id_1", "perplexity": "perplexity_1"}),
        on="text_id_1", how="left",
    )
    df = df.merge(
        perplexity_df.rename(columns={"text_id": "text_id_2", "perplexity": "perplexity_2"}),
        on="text_id_2", how="left",
    )

    # Rescale perplexity within this model's range [0, 1] for cross-model comparability
    ppl_min = df[["perplexity_1", "perplexity_2"]].min().min()
    ppl_max = df[["perplexity_1", "perplexity_2"]].max().max()
    ppl_range = ppl_max - ppl_min if ppl_max > ppl_min else 1.0

    df["perplexity_1"] = (df["perplexity_1"] - ppl_min) / ppl_range
    df["perplexity_2"] = (df["perplexity_2"] - ppl_min) / ppl_range

    df["avg_perplexity"] = (df["perplexity_1"] + df["perplexity_2"]) / 2
    df["max_perplexity"] = df[["perplexity_1", "perplexity_2"]].max(axis=1)
    df["min_perplexity"] = df[["perplexity_1", "perplexity_2"]].min(axis=1)
    df["diff_perplexity"] = (df["perplexity_1"] - df["perplexity_2"]).abs()
    return df


PERPLEXITY_CHARACTERISTICS = {
    "avg_perplexity": "Average perplexity (rescaled [0,1])",
    "max_perplexity": "Maximum perplexity (rescaled [0,1])",
    "min_perplexity": "Minimum perplexity (rescaled [0,1])",
    "diff_perplexity": "Perplexity difference (rescaled [0,1])",
}


# ---------------------------------------------------------------------------
# Alignment analysis ([SIT] judge model's own embedded opinion on the
# situation) - read from --alignment_dir. Each judge model has one CSV per
# evaluator prompt (choice, agree_disagree, ...); every evaluator's raw labels
# are collapsed onto a shared {positive, negative, neutral} schema (A/U/N in
# the notes) and averaged across evaluators, per situation.
# ---------------------------------------------------------------------------

def infer_evaluator_type(filename: str) -> str:
    """Match a filename to a LABEL_MAPPING key. Picks the longest match to
    avoid collisions like 'evaluator' matching multiple keys."""
    matches = [k for k in LABEL_MAPPING if k in filename]
    if not matches:
        raise ValueError(
            f"Could not infer evaluator type from filename '{filename}'. "
            f"Expected one of: {list(LABEL_MAPPING.keys())}"
        )
    return max(matches, key=len)


def parse_alignment_distribution(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    return json.loads(raw)


def row_to_pos_neg_neutral(dist_dict: dict, evaluator_type: str) -> dict:
    mapping = CATEGORY_MAPPING[evaluator_type]
    probs = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for key, val in dist_dict.items():
        if key == "total_evaluations":
            continue
        label = key[2:-1] if key.startswith("p(") and key.endswith(")") else key
        probs[mapping[label]] += val
    return probs


def compute_entropy(probs) -> float:
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs)))


def load_unified_alignment_df(args, judge_model_id: str) -> pd.DataFrame | None:
    align_dir = args.alignment_dir / judge_model_id
    if not align_dir.exists():
        return None

    alignment_files = [f for f in align_dir.glob("*.csv") if "_generations" not in f.name]
    if not alignment_files:
        return None

    records = []
    for f in alignment_files:
        evaluator_type = infer_evaluator_type(f.name)
        evaluator_df = pd.read_csv(f)
        for _, row in evaluator_df.iterrows():
            dist = parse_alignment_distribution(row["alignment_distribution"])
            probs = row_to_pos_neg_neutral(dist, evaluator_type)
            records.append({"situation_id": row["situation_id"], **probs})

    long_df = pd.DataFrame.from_records(records)
    agg = long_df.groupby("situation_id")[["positive", "negative", "neutral"]].mean().reset_index()
    agg = agg.rename(columns={"positive": "p_positive", "negative": "p_negative", "neutral": "p_neutral"})

    agg["entropy"] = agg.apply(
        lambda r: compute_entropy([r["p_positive"], r["p_negative"], r["p_neutral"]]), axis=1
    )
    agg["p_positive_minus_negative"] = agg["p_positive"] - agg["p_negative"]
    agg["max_p_positive_negative"] = agg[["p_positive", "p_negative"]].max(axis=1)
    agg["min_p_positive_negative"] = agg[["p_positive", "p_negative"]].min(axis=1)
    return agg


def enrich_alignment(scoring_df: pd.DataFrame, args, judge_model_id: str) -> pd.DataFrame | None:
    unified = load_unified_alignment_df(args, judge_model_id)
    if unified is None:
        return None
    # situation_id_1 == situation_id_2 within a pair (both texts are opinions
    # on the same situation), so the judge's alignment on the situation
    # applies to the pair as a whole.
    return scoring_df.merge(unified, left_on="situation_id_1", right_on="situation_id", how="left")


ALIGNMENT_CHARACTERISTICS = {
    "p_positive": "P(A) - positive/accept-like",
    "p_negative": "P(U) - negative/unaccept-like",
    "p_neutral": "P(N) - neutral/ambiguous",
    "entropy": "Judge alignment entropy",
    "p_positive_minus_negative": "P(A) - P(U)",
    "max_p_positive_negative": "max(P(A), P(U))",
    "min_p_positive_negative": "min(P(A), P(U))",
}


# ---------------------------------------------------------------------------
# Analyses not implemented yet - registered so the runner reports them as
# skipped rather than silently missing. Fill in `enrich` + `characteristics`
# once the underlying data/model exists.
# ---------------------------------------------------------------------------

# [GEN] [0,1] Lexicon-based - ex-post (no ex-ante lexicon split available yet).
LEXICON_SPEC = None

# [GEN] [score] Emotion-based - average/max/min/diff polarity or emotion score
# over the generator's texts.
EMOTION_SPEC = None

# [GEN] [score] Safety-guardrails - Llama-Guard probability of "safe" on the
# generator's generations.

def enrich_llamaguard(
    scoring_df: pd.DataFrame, args, judge_model_id
) -> pd.DataFrame | None:

    # Situation scores
    path_llama_situation = args.llama_guard_dir / "situation_scores.csv"
    if not path_llama_situation.is_file():
        return None

    df_llama_situations = pd.read_csv(path_llama_situation)

    df_llama_situations = df_llama_situations.rename(
        columns={"unsafe_score": "unsafe_score_situation"}
    )

    df = scoring_df.merge(
        df_llama_situations[["situation_id", "unsafe_score_situation"]],
        left_on="situation_id_1",
        right_on="situation_id",
        how="left",
    )

    # Opinion scores
    path_llama_opinion = args.llama_guard_dir / "opinion_scores.csv"
    if not path_llama_opinion.is_file():
        return None

    df_llama_opinions = pd.read_csv(path_llama_opinion)

    print("Opinion CSV columns:", df_llama_opinions.columns.tolist())
    print("Scoring DF columns:", df.columns.tolist())

    # Make sure the source column has the expected name.
    df_llama_opinions = df_llama_opinions.rename(
        columns={"unsafe_score": "unsafe_score_opinion"}
    )

    df = df.merge(
        df_llama_opinions[
            ["text_id", "unsafe_score_opinion"]
        ].rename(
            columns={
                "text_id": "text_id_1",
                "unsafe_score_opinion": "unsafe_score_opinion_1",
            }
        ),
        on="text_id_1",
        how="left",
    )

    df = df.merge(
        df_llama_opinions[
            ["text_id", "unsafe_score_opinion"]
        ].rename(
            columns={
                "text_id": "text_id_2",
                "unsafe_score_opinion": "unsafe_score_opinion_2",
            }
        ),
        on="text_id_2",
        how="left",
    )

    # print(df.head(10))
    # print("Final columns:", df.columns.tolist())

    df["avg_unsafe_score_opinion"] = (
        df["unsafe_score_opinion_1"]
        + df["unsafe_score_opinion_2"]
    ) / 2

    df["max_unsafe_score_opinion"] = df[
        ["unsafe_score_opinion_1", "unsafe_score_opinion_2"]
    ].max(axis=1)

    df["min_unsafe_score_opinion"] = df[
        ["unsafe_score_opinion_1", "unsafe_score_opinion_2"]
    ].min(axis=1)

    df["diff_unsafe_score_opinion"] = (
        df["unsafe_score_opinion_1"]
        - df["unsafe_score_opinion_2"]
    ).abs()

    return df

LLAMA_GUARD_CHARACTERISTICS = {
    "avg_unsafe_score_opinion": "Average Llama-guard Unsafe Score for opinion",
    "max_unsafe_score_opinion": "Maximum Llama-guard Unsafe Score for opinion",
    "min_unsafe_score_opinion": "Minimum Llama-guard Unsafe Score for opinion",
    "diff_unsafe_score_opinion": "Perplexity Llama-guard Unsafe Score for opinion",
    "unsafe_score_situation": "Llama-guard Unsafe Score for situation"

}

# [SIT] [score] Safety-guardrails - judge model's probability of a refusal-
# to-comply option (+ similarity to Arditi et al.'s refusal direction).
REFUSAL_SPEC = None

# [SIT] [score] Safety-guardrails - Llama-Guard on the judge model's own
# generations for the situation.
LLAMA_GUARD_JUDGE_SPEC = None


ANALYSIS_REGISTRY = {
    "text": {"enrich": enrich_text, "characteristics": TEXT_CHARACTERISTICS},
    "perplexity": {"enrich": enrich_perplexity, "characteristics": PERPLEXITY_CHARACTERISTICS},
    "alignment": {"enrich": enrich_alignment, "characteristics": ALIGNMENT_CHARACTERISTICS},
    "lexicon": LEXICON_SPEC,
    "emotion": EMOTION_SPEC,
    "llama_guard_generator": {"enrich": enrich_llamaguard,
                "characteristics": LLAMA_GUARD_CHARACTERISTICS},
    "refusal": REFUSAL_SPEC,
    "llama_guard_judge": LLAMA_GUARD_JUDGE_SPEC,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_analysis(analysis_type: str, spec: dict | None, args, judge_model_ids: list[str]):
    if spec is None:
        print(f"[{analysis_type}] Not implemented yet - skipping.")
        return

    out_dir = args.output_dir / analysis_type
    per_characteristic_summaries: dict[str, dict[str, pd.DataFrame]] = {}

    for judge_model_id in judge_model_ids:
        scoring_df = load_scoring_df(args, judge_model_id)
        if scoring_df is None:
            print(f"[{analysis_type}] No scoring data for judge={judge_model_id} - skipping.")
            continue

        enriched_df = spec["enrich"](scoring_df, args, judge_model_id)
        if enriched_df is None:
            print(f"[{analysis_type}] No source data for judge={judge_model_id} - skipping.")
            continue

        judge_out_dir = out_dir / judge_model_id
        judge_out_dir.mkdir(parents=True, exist_ok=True)

        for char_name in spec["characteristics"]:
            summary = bin_and_compute_stability(enriched_df, char_name, args.n_bins, args.bin_method)
            summary.to_csv(judge_out_dir / f"{char_name}.csv", index=False)
            per_characteristic_summaries.setdefault(char_name, {})[judge_model_id] = summary

    for char_name, char_label in spec["characteristics"].items():
        summaries = per_characteristic_summaries.get(char_name, {})
        if not summaries:
            continue
        plot_stability_across_models(summaries, char_label, out_dir / f"{char_name}.png")

    print(f"[{analysis_type}] Done -> {out_dir}")


def main():
    args = parse_command_line_args()
    args.scoring_dir = args.scoring_dir / args.dataset_id
    args.perplexity_dir = args.perplexity_dir / args.dataset_id
    args.alignment_dir = args.alignment_dir / args.dataset_id
    args.output_dir = args.output_dir / args.dataset_id
    args.output_dir.mkdir(parents=True, exist_ok=True)

    judge_model_ids = discover_judge_model_ids(args)
    if not judge_model_ids:
        if args.dataset_id == "valueprism":
            raise SystemExit(
                f"No scoring files found for generator={args.generator_model_id!r}, "
                f"prompt={args.generator_prompt_version!r} in {args.scoring_dir}"
            )
        raise SystemExit(f"No scoring files found in {args.scoring_dir}")
    print(f"Judge models: {judge_model_ids}")

    for analysis_type, spec in ANALYSIS_REGISTRY.items():
        run_analysis(analysis_type, spec, args, judge_model_ids)


if __name__ == "__main__":
    main()
