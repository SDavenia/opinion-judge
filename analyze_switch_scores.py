import argparse
import json

import pandas as pd

from utils.models_utils import REGISTRY
from utils.prompts_utils import GENERATION_PROMPTS

DROP_COLUMNS = [
    "vrd_1", "text_1", "explanation_1", "generated_opinion_1", "model_used_1",
    "vrd_2", "text_2", "explanation_2", "generated_opinion_2", "model_used_2",
    "generated_score_1to2", "generated_score_2to1", "generation_model_id", "generation_prompt_version"
]

TERM_WIDTH = 78


# --------------------------------------------------------------------------
# Pretty-printing helpers
# --------------------------------------------------------------------------

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
    """Sub-sub-section header (e.g. one per subset)."""
    print(f"  · {title}")


def fmt_pct(n: int, total: int) -> str:
    pct = round(n * 100 / total, 2) if total else 0.0
    return f"{n:>4}/{total:<4} ({pct:5.2f}%)"


def print_reverse_stats(rows: dict):
    """rows: {label: (count, pct)} -> aligned table."""
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


# --------------------------------------------------------------------------
# Data loading / preparation
# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Detailed breakdown of safe/unsafe score directions for a given model."
    )
    parser.add_argument(
        "--judge_model_id", required=True,
        choices=list(REGISTRY.keys()),
        help="Judge Model identifier, e.g. 'gemma3' or 'llama-3.3-70b'.",
    )
    parser.add_argument(
        "--generation_model_id", type=str, default="llama-3.3-70b",
        choices=list(REGISTRY.keys()),
        help="Generation Model identifier, e.g. 'gemma3' or 'llama-3.3-70b'.",
    )
    parser.add_argument(
        "--generation_prompt_version", type=str, default="reflective_person",
        choices=list(GENERATION_PROMPTS.keys()),
    )
    # Flags to choose whether to use greedy or probabilistic scoring/alignment data.
    parser.add_argument(
        "--use_probabilistic_scoring", action="store_true",
        help="Use the '_probability' variant of the output_scores directory.",
    )
    parser.add_argument(
        "--use_probabilistic_alignment", action="store_true",
        help="Use the '_probability' variant of the model_alignment directory.",
    )
    parser.add_argument(
        "--alignment_prompt_version", type=str
    )
    # Flags to modify the input path based on the number of situations used in the generation and scoring steps.
    limit_group = parser.add_mutually_exclusive_group()
    limit_group.add_argument(
        "--limit", dest="limit", action="store_true",
        help="Use the '_limit' variant of the input files (default).",
    )
    limit_group.add_argument(
        "--num_situations", type=int, default=None,
        help="If passed, the scores are read from the corresponding directory in scoring/results{num_situations}."
    )
    return parser.parse_args()


def load_data(judge_model_id: str, 
              generation_model_id: str,
              generation_prompt_version: str,
              limit: bool | int | None, 
              use_probabilistic_scoring: bool, 
              use_probabilistic_alignment: bool,
              alignment_prompt: str):
    ### Read scores
    # If num instances is specified, the path was changed
    if isinstance(limit, int):
        # limit == num_situations in this case
        scores_path = f"scoring/results{limit}/{'prob' if use_probabilistic_scoring else 'greedy'}/{generation_model_id}_{generation_prompt_version}_{judge_model_id}_base.csv"
    # if limit is passed as a bool
    elif limit is True:
        scores_path = f"scoring/results/{'prob' if use_probabilistic_scoring else 'greedy'}/{generation_model_id}_{generation_prompt_version}_{judge_model_id}_base_limit.csv"
    df_scores = pd.read_csv(scores_path)
    df_scores = df_scores.drop(columns=DROP_COLUMNS)

    ### Read model alignment data
    if isinstance(limit, int):
        alignment_path = (
            f"model_alignment{'_probability' if use_probabilistic_alignment else ''}/results{limit}/{judge_model_id}_random_selected_{alignment_prompt}.json"
        )
    else:
        alignment_path = f"model_alignment{'_probability' if use_probabilistic_alignment else ''}/results/{judge_model_id}_{alignment_prompt}.json"
    model_alignment_data = json.load(open(alignment_path))
    return df_scores, model_alignment_data


def reformat_model_alignment_data(model_alignment_data):
    model_alignment_data_reformatted = {}
    for situation, entry in model_alignment_data.items():
        p_acceptable = entry["p(ACCEPTABLE)"]
        p_unacceptable = entry["p(UNACCEPTABLE)"]
        model_alignment_data_reformatted[situation] = {
            "p_acceptable": p_acceptable,
            "p_unacceptable": p_unacceptable,
            "majority_opinion": "Supports" if p_acceptable >= p_unacceptable else "Opposes",
            "minority_opinion": "Supports" if p_acceptable < p_unacceptable else "Opposes",
            "majority_opinion_probability": max(p_acceptable, p_unacceptable),
            "minority_opinion_probability": min(p_acceptable, p_unacceptable),
        }
    return model_alignment_data_reformatted


def enrich_scores_with_alignment(df_scores, model_alignment_data_reformatted):
    """Add majority/minority opinion columns and the safe/unsafe/either
    classification for each row of the full df_scores (no Either-aware
    resolution needed here, since this just tags each row with the
    situation's overall majority/minority)."""
    df_scores = df_scores.copy()

    majority_opinions, minority_opinions = [], []
    majority_opinions_probs, minority_opinions_probs = [], []
    for _, row in df_scores.iterrows():
        situation = row["situation"]
        info = model_alignment_data_reformatted[situation]
        majority_opinions.append(info["majority_opinion"])
        majority_opinions_probs.append(info["majority_opinion_probability"])
        minority_opinions.append(info["minority_opinion"])
        minority_opinions_probs.append(info["minority_opinion_probability"])

    df_scores["majority_opinion"] = majority_opinions
    df_scores["majority_opinion_probability"] = majority_opinions_probs
    df_scores["minority_opinion"] = minority_opinions
    df_scores["minority_opinion_probability"] = minority_opinions_probs

    all_valence1_safeunsafe, all_valence2_safeunsafe = [], []
    for _, row in df_scores.iterrows():
        all_valence1_safeunsafe.append(
            "safe" if row["valence_1"] == row["majority_opinion"]
            else "either" if row["valence_1"] == "Either"
            else "unsafe"
        )
        all_valence2_safeunsafe.append(
            "safe" if row["valence_2"] == row["majority_opinion"]
            else "either" if row["valence_2"] == "Either"
            else "unsafe"
        )
    df_scores["valence_1_safeunsafe"] = all_valence1_safeunsafe
    df_scores["valence_2_safeunsafe"] = all_valence2_safeunsafe

    # NOTE: sort by the su label itself (not by the raw valence text). Sorting
    # by valence text would split the "safe vs unsafe" rows into two separate
    # groups purely based on which opinion string ("Opposes"/"Supports")
    # happens to be alphabetically first for a given situation -- an artifact
    # unrelated to the safe/unsafe meaning we actually care about. Sorting by
    # the su label instead gives a stable, meaningful grouping, and
    # conveniently "either" < "safe" < "unsafe" alphabetically already.
    df_scores["safeunsafe"] = df_scores.apply(
        lambda row: tuple(
            sorted([row["valence_1_safeunsafe"], row["valence_2_safeunsafe"]])
        ),
        axis=1,
    )
    return df_scores


def count_reverse_scores(df):
    cnt_reverse = 0
    for _, row in df.iterrows():
        if row["parsed_score1to2"] != row["parsed_score2to1"]:
            cnt_reverse += 1
    pct = round(cnt_reverse * 100 / len(df), 2) if len(df) else 0.0
    return cnt_reverse, pct


def split_support_oppose(df_scores):
    """Cases of interest: Support vs Oppose (Either rows excluded entirely)."""
    df_other = df_scores[
        (df_scores["valence_1"] == df_scores["valence_2"])
        | (df_scores["valence_1"] == "Either")
        | (df_scores["valence_2"] == "Either")
    ]
    df_interest = df_scores[df_scores.index.isin(df_other.index) == False]

    df_other = df_other.reset_index(drop=True)
    df_interest = df_interest.reset_index(drop=True)
    return df_other, df_interest


def split_support_oppose_with_either(df_scores):
    """Same analysis but treating Either as closer to the dangerous or safe option."""
    df_other = df_scores[(df_scores["valence_1"] == df_scores["valence_2"])]
    df_interest = df_scores[df_scores.index.isin(df_other.index) == False]

    df_other = df_other.reset_index(drop=True)
    df_interest = df_interest.reset_index(drop=True)
    return df_other, df_interest


def enrich_data_supportoppose_with_either(df, model_alignment_data_reformatted):
    """Resolve majority/minority opinion per row, treating 'Either' as
    absorbing whichever side (majority or minority) is NOT already present
    as the other valence. Each row is resolved independently from the
    situation's own alignment data (row_majority/row_minority), never from
    a leftover value carried over from a previous row."""
    df = df.copy()
    majority_opinions, minority_opinions = [], []
    majority_opinions_probs, minority_opinions_probs = [], []

    for _, row in df.iterrows():
        situation = row["situation"]
        valence_1 = row["valence_1"]
        valence_2 = row["valence_2"]

        row_majority = model_alignment_data_reformatted[situation]["majority_opinion"]
        row_majority_prob = model_alignment_data_reformatted[situation]["majority_opinion_probability"]
        row_minority = model_alignment_data_reformatted[situation]["minority_opinion"]
        row_minority_prob = model_alignment_data_reformatted[situation]["minority_opinion_probability"]

        if valence_1 != "Either" and valence_2 != "Either":
            majority_opinion = row_majority
            majority_opinion_prob = row_majority_prob
            minority_opinion = row_minority
            minority_opinion_prob = row_minority_prob
        elif row_majority in (valence_1, valence_2):
            # Either becomes the minority; majority remains the one with majority
            majority_opinion = row_majority
            majority_opinion_prob = row_majority_prob
            minority_opinion = "Either"
            minority_opinion_prob = row_minority_prob
        elif row_minority in (valence_1, valence_2):
            majority_opinion = "Either"
            majority_opinion_prob = row_majority_prob
            minority_opinion = row_minority
            minority_opinion_prob = row_minority_prob
        else:
            raise ValueError(
                f"Neither majority nor minority opinion found in valences for "
                f"situation {situation}: valence_1={valence_1}, valence_2={valence_2}"
            )

        majority_opinions.append(majority_opinion)
        majority_opinions_probs.append(majority_opinion_prob)
        minority_opinions.append(minority_opinion)
        minority_opinions_probs.append(minority_opinion_prob)

    df["majority_opinion"] = majority_opinions
    df["majority_opinion_probability"] = majority_opinions_probs
    df["minority_opinion"] = minority_opinions
    df["minority_opinion_probability"] = minority_opinions_probs
    return df


def build_direction_scores(df_interest):
    all_scores_safe_unsafe, all_scores_unsafe_safe = [], []
    for _, row in df_interest.iterrows():
        minority_opinion = row["minority_opinion"]  # Proxy for danger
        idx_minority_opinion = "1" if row["valence_1"] == minority_opinion else "2"
        idx_majority_opinion = "2" if row["valence_1"] == minority_opinion else "1"

        score_unsafe_safe = row[f"parsed_score{idx_minority_opinion}to{idx_majority_opinion}"]
        score_safe_unsafe = row[f"parsed_score{idx_majority_opinion}to{idx_minority_opinion}"]

        all_scores_unsafe_safe.append(score_unsafe_safe)
        all_scores_safe_unsafe.append(score_safe_unsafe)
    return all_scores_safe_unsafe, all_scores_unsafe_safe


def direction_counts(all_scores_safe_unsafe, all_scores_unsafe_safe):
    safeunsafe_higher = len([
        i for i in range(len(all_scores_safe_unsafe))
        if all_scores_safe_unsafe[i] > all_scores_unsafe_safe[i]
    ])
    unsafesafe_higher = len([
        i for i in range(len(all_scores_safe_unsafe))
        if all_scores_unsafe_safe[i] > all_scores_safe_unsafe[i]
    ])
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
    for unique_safeunsafe in df_interest["safeunsafe"].unique():
        df_subset = df_interest[df_interest["safeunsafe"] == unique_safeunsafe]
        subset_safe_unsafe = [all_scores_safe_unsafe[x] for x in df_subset.index]
        subset_unsafe_safe = [all_scores_unsafe_safe[x] for x in df_subset.index]
        print_direction_counts(subset_safe_unsafe, subset_unsafe_safe, label=f"Subset {unique_safeunsafe}")
        print()


# --------------------------------------------------------------------------
# Analysis sections
# --------------------------------------------------------------------------

def run_support_oppose_section(df_scores):
    h1("Cases of interest: Support vs Oppose")
    df_other, df_interest = split_support_oppose(df_scores)

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


def run_with_either_section(df_scores, model_alignment_data_reformatted):
    h1("Same analysis, treating Either as closer to the dangerous or safe option")
    df_other, df_interest = split_support_oppose_with_either(df_scores)
    df_interest = enrich_data_supportoppose_with_either(df_interest, model_alignment_data_reformatted)

    h2("Reverse-score rates")
    stats = {
        "df_other": count_reverse_scores(df_other),
        "df_interest": count_reverse_scores(df_interest),
    }
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

def main():
    args = parse_args()
    df_scores, model_alignment_data = load_data(
        judge_model_id=args.judge_model_id,
        generation_model_id=args.generation_model_id,
        generation_prompt_version=args.generation_prompt_version,
        limit=args.limit if args.limit else args.num_situations, # If both None this will default to full set.
        use_probabilistic_scoring=args.use_probabilistic_scoring,
        use_probabilistic_alignment=args.use_probabilistic_alignment,
        alignment_prompt=args.alignment_prompt_version
    )
    model_alignment_data_reformatted = reformat_model_alignment_data(model_alignment_data)
    df_scores = enrich_scores_with_alignment(df_scores, model_alignment_data_reformatted)

    print(f"Model: {args.judge_model_id}  |  limit={args.limit}  |  probabilistic={args.use_probabilistic_alignment}")

    run_support_oppose_section(df_scores)
    run_with_either_section(df_scores, model_alignment_data_reformatted)
    print()


if __name__ == "__main__":
    main()