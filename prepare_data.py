"""Download, filter, and save opinion/value datasets used for training and eval.

Datasets
--------
- Habermas Machine  (real)      -> habermas_sample.csv
- ValuePrism        (synthetic) -> valueprism_sample.csv
- Humanual Opinion  (real)      -> not yet implemented

Pipeline (same shape for Habermas and ValuePrism)
--------------------------------------------------
1. Keep only situations with MORE than `*_MIN_INSTANCES` total opinions.
2. Rank the surviving situations by how balanced their stance mix is
   (1.0 = perfectly uniform across stances, 0.0 = maximally skewed), and
   keep the top `--max_situations` of them.
3. Within each kept situation, sample up to `*_OPINIONS_PER_STANCE` (2)
   opinions per stance:
     - ValuePrism: the 2 opinions per stance are chosen to be as different
       from each other as possible (by explanation text), so we don't end
       up scoring near-duplicate opinions.
     - Habermas: opinions are sampled per stance with no diversity
       requirement.
   If a stance has fewer opinions than requested, whatever is available is
   kept (down to a single opinion, or none if the stance is absent) — no
   situation is dropped just for being short on one stance.

Each dataset is written out as a CSV with a common schema:

    situation_id, text_id, situation, stance, dataset, ...extra columns
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")
log = logging.getLogger("prepare_datasets")


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

HABERMAS_BASE_URL = "https://storage.googleapis.com/habermas_machine/datasets"
HABERMAS_RATINGS_FILE = "hm_all_position_statement_ratings.parquet"
HABERMAS_CANDIDATES_FILE = "hm_all_candidate_comparisons.parquet"

AGREE_LABELS = {"STRONGLY_AGREE", "AGREE", "SOMEWHAT_AGREE"}
DISAGREE_LABELS = {"STRONGLY_DISAGREE", "DISAGREE", "SOMEWHAT_DISAGREE"}
NEUTRAL_LABELS = {"NEUTRAL"}
HABERMAS_STANCES = ["agree", "disagree", "neutral"]
HABERMAS_MIN_INSTANCES = 6
HABERMAS_OPINIONS_PER_STANCE = 2

VALUEPRISM_URL = "hf://datasets/allenai/ValuePrism/full/full.csv"
# ValuePrism's third stance is labeled "Either" in the raw data (i.e. "neutral").
VALUEPRISM_STANCES = ["Supports", "Opposes", "Either"]
VALUEPRISM_MIN_INSTANCES = 6
VALUEPRISM_OPINIONS_PER_STANCE = 2

DEFAULT_MAX_SITUATIONS = 1000


# --------------------------------------------------------------------------- #
# Shared helpers: situation ranking by stance balance
# --------------------------------------------------------------------------- #

def _compute_balance_scores(df: pd.DataFrame, situation_col: str, stance_col: str, stance_values: list[str]) -> pd.DataFrame:
    """Per-situation opinion counts, total n, and a stance-balance score in [0, 1].

    balance_score == 1.0 means the situation's opinions are split perfectly
    evenly across `stance_values`; 0.0 means they are maximally skewed to a
    single stance.
    """
    counts = (
        df.groupby([situation_col, stance_col])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=stance_values, fill_value=0)
    )
    counts["n"] = counts[stance_values].sum(axis=1)

    target = 1 / len(stance_values)
    proportions = counts[stance_values].div(counts["n"], axis=0)
    uniformity_distance = proportions.sub(target).abs().sum(axis=1)
    counts["balance_score"] = 1 - uniformity_distance / (2 * (1 - target))

    return counts


def _select_top_situations(counts: pd.DataFrame, min_instances: int, max_situations: int, dataset_label: str) -> pd.Index:
    """Keep situations with more than `min_instances` opinions, ranked by balance_score."""
    valid = counts[counts["n"] > min_instances]

    log.info(
        "[%s] %d / %d situations have more than %d entries",
        dataset_label, len(valid), len(counts), min_instances,
    )
    if len(valid) < max_situations:
        log.warning(
            "[%s] Only %d situations passed the filter — fewer than the requested top %d, "
            "keeping all of them",
            dataset_label, len(valid), max_situations,
        )

    return valid.nlargest(max_situations, "balance_score").index


# --------------------------------------------------------------------------- #
# Shared helpers: per-situation opinion sampling
# --------------------------------------------------------------------------- #

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def _jaccard_distance(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return 1 - len(a & b) / len(union)


def _pick_diverse_subset(texts: list[str], k: int) -> list[int]:
    """Indices of up to `k` texts chosen to be as mutually different as possible.

    Uses simple token-overlap (Jaccard) distance and greedy farthest-point
    sampling: start from the single most dissimilar pair, then repeatedly add
    whichever remaining text maximizes its minimum distance to what's already
    selected. Dependency-free by design (no embeddings/network calls needed).
    """
    n = len(texts)
    if k <= 0 or n == 0:
        return []
    if n <= k:
        return list(range(n))
    if k == 1:
        return [0]

    token_sets = [_tokenize(t) for t in texts]
    dist = [[_jaccard_distance(token_sets[i], token_sets[j]) for j in range(n)] for i in range(n)]

    best_pair, best_dist = (0, 1), -1.0
    for i, j in combinations(range(n), 2):
        if dist[i][j] > best_dist:
            best_dist, best_pair = dist[i][j], (i, j)
    selected = list(best_pair)

    while len(selected) < k:
        remaining = [i for i in range(n) if i not in selected]
        next_idx = max(remaining, key=lambda i: min(dist[i][s] for s in selected))
        selected.append(next_idx)

    return selected


def _balanced_target_allocation(capacities: dict[str, int], target_total: int) -> dict[str, int]:
    """Split `target_total` slots across groups as evenly as possible, respecting caps.

    Each group gets an equal share on every pass; a group that can't absorb its
    share (because it doesn't have enough opinions) gives up its shortfall,
    which gets redistributed across the remaining, non-saturated groups. This
    is what lets a situation with e.g. only 1 "Opposes" opinion still end up
    with `target_total` opinions overall, by taking more from the stances that
    have plenty.
    """
    allocation = {g: 0 for g in capacities}
    remaining = min(target_total, sum(capacities.values()))
    active = [g for g in capacities if capacities[g] > 0]

    while remaining > 0 and active:
        share, extra = divmod(remaining, len(active))
        if share == 0:
            # Not enough left for a full round: hand out the leftovers one at a
            # time, prioritizing groups with the most spare capacity.
            for g in sorted(active, key=lambda g: -(capacities[g] - allocation[g]))[:extra]:
                allocation[g] += 1
                remaining -= 1
            break
        for g in active:
            room = capacities[g] - allocation[g]
            take = min(share, room)
            allocation[g] += take
            remaining -= take
        active = [g for g in active if capacities[g] - allocation[g] > 0]

    return allocation


def _sample_diverse_balanced(df: pd.DataFrame, stance_col: str, stance_values: list[str], text_col: str, target_total: int) -> pd.DataFrame:
    """Within one situation, sample `target_total` opinions spread as evenly as
    possible across stances, and as mutually diverse as possible within each
    stance. If a stance is short on opinions, the shortfall is made up from
    the other stances rather than shrinking the total below `target_total`.
    """
    capacities = {stance: (df[stance_col] == stance).sum() for stance in stance_values}
    allocation = _balanced_target_allocation(capacities, target_total)

    parts = []
    for stance in stance_values:
        k = allocation[stance]
        if k == 0:
            continue
        group = df[df[stance_col] == stance]
        idxs = _pick_diverse_subset(group[text_col].fillna("").tolist(), k)
        parts.append(group.iloc[idxs])
    return pd.concat(parts) if parts else df.iloc[0:0]


def _sample_diverse_sublabels(group: pd.DataFrame, raw_label_col: str, k: int, random_state: int = 0) -> pd.DataFrame:
    """Sample up to `k` rows, preferring to spread across distinct raw labels
    before repeating one (e.g. for the "agree" stance, prefer one AGREE and
    one SOMEWHAT_AGREE opinion over two AGREE opinions) — a proxy for opinion
    diversity within a coarse stance when we don't have free-text to compare.
    Degrades to plain sampling when only one raw label is present (e.g. the
    "neutral" stance, which only maps from NEUTRAL).
    """
    if len(group) <= k:
        return group

    shuffled = group.sample(frac=1, random_state=random_state)
    selected_idxs: list = []
    seen_labels: set = set()

    for idx, row in shuffled.iterrows():
        if len(selected_idxs) >= k:
            break
        label = row[raw_label_col]
        if label not in seen_labels:
            selected_idxs.append(idx)
            seen_labels.add(label)

    if len(selected_idxs) < k:
        return _fill_remaining(group, shuffled, selected_idxs, k)
    return group.loc[selected_idxs]


def _fill_remaining(group: pd.DataFrame, shuffled: pd.DataFrame, selected_idxs: list, k: int) -> pd.DataFrame:
    """Top up `selected_idxs` to length `k` with any not-yet-picked rows."""
    remaining = [idx for idx in shuffled.index if idx not in selected_idxs]
    selected_idxs = selected_idxs + remaining[: k - len(selected_idxs)]
    return group.loc[selected_idxs]


def _sample_up_to_k_per_stance(df: pd.DataFrame, stance_col: str, stance_values: list[str], k: int, raw_label_col: str, random_state: int = 0) -> pd.DataFrame:
    """Within one situation, take up to `k` opinions per stance, preferring
    diversity of raw sub-label within each stance (see `_sample_diverse_sublabels`).

    Falls back to whatever is available when a stance has fewer than `k`
    opinions (down to a single opinion, or the stance is simply skipped if
    it's entirely absent).
    """
    parts = []
    for stance in stance_values:
        group = df[df[stance_col] == stance]
        if group.empty:
            continue
        parts.append(_sample_diverse_sublabels(group, raw_label_col, k, random_state))
    return pd.concat(parts) if parts else df.iloc[0:0]


# --------------------------------------------------------------------------- #
# Habermas Machine
# --------------------------------------------------------------------------- #

def _download_habermas_parquets(tmpdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download the two Habermas Machine parquet files and load them."""
    for filename in (HABERMAS_RATINGS_FILE, HABERMAS_CANDIDATES_FILE):
        log.info("Downloading %s", filename)
        subprocess.run(
            ["wget", "-q", "-P", str(tmpdir), f"{HABERMAS_BASE_URL}/{filename}"],
            check=True,
        )

    ratings = pd.read_parquet(tmpdir / HABERMAS_RATINGS_FILE)
    candidates = pd.read_parquet(tmpdir / HABERMAS_CANDIDATES_FILE)
    return ratings, candidates


def _clean_habermas_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Keep first-stage ratings only, unwrap the single-element rating list."""
    cols = [
        "metadata.participant_id",
        "question.id",
        "question.text",
        "question.affirming_statement",
        "question.negating_statement",
        "ratings.agreement",
    ]
    ratings = ratings[ratings["rating_index"] == 0][cols].copy()
    ratings["ratings.agreement"] = ratings["ratings.agreement"].apply(lambda x: x[0])
    ratings = ratings[ratings["ratings.agreement"] != "[MOCK]"]
    return ratings


def _clean_habermas_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Keep completed, human-written opinions (one per participant/question)."""
    cols = [
        "question.id",
        "question.text",
        "question.split",
        "question.affirming_statement",
        "question.negating_statement",
        "question.topic",
        "own_opinion.text",
        "metadata.participant_id",
        "own_opinion.metadata.provenance",
        "own_opinion.metadata.status",
        "ratings.agreement",
    ]
    candidates = candidates[cols]
    candidates = candidates[candidates["own_opinion.metadata.provenance"] != "BOT_CITIZEN"]
    candidates = candidates[candidates["own_opinion.metadata.status"] == "COMPLETED"]
    candidates = candidates.drop_duplicates(subset=["metadata.participant_id", "question.id"])

    keep = [
        "metadata.participant_id",
        "question.id",
        "question.text",
        "question.affirming_statement",
        "question.negating_statement",
        "own_opinion.text",
        "ratings.agreement",
        "question.topic",
    ]
    return candidates[keep]


def _map_agreement_to_group(label: str) -> str | None:
    """Collapse the fine-grained Likert label into agree / disagree / neutral."""
    if label in AGREE_LABELS:
        return "agree"
    if label in DISAGREE_LABELS:
        return "disagree"
    if label in NEUTRAL_LABELS:
        return "neutral"
    return None


def prepare_habermas(output_dir: Path, max_situations: int) -> None:
    """Download, filter, and save the Habermas Machine dataset."""
    log.info("Preparing Habermas Machine dataset")
    with TemporaryDirectory() as tmpdir:
        ratings_raw, candidates_raw = _download_habermas_parquets(Path(tmpdir))

    ratings = _clean_habermas_ratings(ratings_raw)
    candidates = _clean_habermas_candidates(candidates_raw)

    merged = pd.merge(
        ratings,
        candidates,
        on=[
            "metadata.participant_id",
            "question.id",
            "question.text",
            "question.affirming_statement",
            "question.negating_statement",
        ],
        suffixes=("_position", "_candidate"),
    )

    merged["stance_group"] = merged["ratings.agreement_position"].map(_map_agreement_to_group)
    merged = merged.dropna(subset=["stance_group"])

    counts = _compute_balance_scores(merged, "question.id", "stance_group", HABERMAS_STANCES)
    selected_situations = _select_top_situations(counts, HABERMAS_MIN_INSTANCES, max_situations, "habermas")
    merged = merged[merged["question.id"].isin(selected_situations)].copy()

    merged = merged.groupby("question.id", group_keys=False).apply(
        lambda g: _sample_up_to_k_per_stance(
            g, "stance_group", HABERMAS_STANCES, HABERMAS_OPINIONS_PER_STANCE,
            raw_label_col="ratings.agreement_position",
        )
    )

    merged["text_id"] = [f"HB_{idx}" for idx in merged.index]
    merged["question.id"] = merged["question.id"].str.replace("S", "HB_", regex=False)

    merged = merged.rename(
        columns={
            "question.id": "situation_id",
            "metadata.participant_id": "hb_participant_id",
            "question.affirming_statement": "situation",
            "ratings.agreement_position": "stance",
            "own_opinion.text": "text",
        }
    )
    merged["dataset"] = "habermas"

    out_cols = ["situation_id", "text_id", "situation", "stance", "stance_group", "hb_participant_id", "dataset"]
    merged = merged[out_cols]

    out_path = output_dir / "habermas_sample.csv"
    merged.to_csv(out_path, index=False)
    log.info(
        "Wrote %d rows across %d situations to %s",
        len(merged), merged["situation_id"].nunique(), out_path,
    )


# --------------------------------------------------------------------------- #
# ValuePrism
# --------------------------------------------------------------------------- #

def prepare_valueprism(output_dir: Path, max_situations: int) -> None:
    """Download, filter, and save the ValuePrism dataset."""
    log.info("Preparing ValuePrism dataset")
    df = pd.read_csv(VALUEPRISM_URL, dtype={"vrd": str, "text": str})

    counts = _compute_balance_scores(df, "situation", "valence", VALUEPRISM_STANCES)
    selected_situations = _select_top_situations(counts, VALUEPRISM_MIN_INSTANCES, max_situations, "valueprism")
    df = df[df["situation"].isin(selected_situations)].copy()

    target_total = len(VALUEPRISM_STANCES) * VALUEPRISM_OPINIONS_PER_STANCE  # 6: balanced target, not a hard per-stance cap
    df = df.groupby("situation", group_keys=False).apply(
        lambda g: _sample_diverse_balanced(g, "valence", VALUEPRISM_STANCES, "explanation", target_total)
    )

    df["text_id"] = [f"VP_{i}" for i in df.index]
    situation_ids = {s: f"VPS_{i}" for i, s in enumerate(df["situation"].unique())}
    df["situation_id"] = df["situation"].map(situation_ids)

    df = df.rename(
        columns={
            "valence": "stance",
            "vrd": "vp_vrd",
            "text": "vp_vrd_value",
            "explanation": "vp_explanation",
        }
    )
    df["dataset"] = "valueprism"

    out_cols = ["situation_id", "text_id", "situation", "stance", "vp_vrd", "vp_vrd_value", "vp_explanation", "dataset"]
    df = df[out_cols]

    out_path = output_dir / "valueprism_sample.csv"
    df.to_csv(out_path, index=False)
    log.info(
        "Wrote %d rows across %d situations to %s",
        len(df), df["situation_id"].nunique(), out_path,
    )


# --------------------------------------------------------------------------- #
# Humanual Opinion
# --------------------------------------------------------------------------- #

def prepare_humanual_opinion(output_dir: Path, max_situations: int) -> None:
    """Placeholder: Humanual Opinion dataset is not yet implemented."""
    log.warning("Humanual Opinion dataset preparation is not yet implemented — skipping")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

DATASET_PREPARERS = {
    "habermas": prepare_habermas,
    "valueprism": prepare_valueprism,
    "humanual": prepare_humanual_opinion,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output_dir", type=Path, default=Path("data/"),
        help="Directory to write the filtered CSV files to.",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASET_PREPARERS), default=list(DATASET_PREPARERS),
        help="Which datasets to prepare (default: all).",
    )
    parser.add_argument(
        "--max_situations", type=int, default=DEFAULT_MAX_SITUATIONS,
        help=(
            "Max number of situations to keep per dataset, ranked by stance balance "
            f"(default: {DEFAULT_MAX_SITUATIONS})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        try:
            DATASET_PREPARERS[name](args.output_dir, args.max_situations)
        except Exception:
            log.exception("Failed to prepare dataset %r", name)
            raise


if __name__ == "__main__":
    main()