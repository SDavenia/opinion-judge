"""Download, filter, and save opinion/value datasets used for training and eval.

Datasets
--------
- Habermas Machine  (real)   -> habermas_sample.csv
- ValuePrism        (synthetic) -> valueprism_sample.csv
- Humanual Opinion  (real)   -> not yet implemented

Each dataset is reduced to a small set of "situations" for which annotators
disagree (i.e. we can observe agree / disagree / neutral, or a balanced mix
of stances), then written out as a CSV with a common schema:

    situation_id, text_id, situation, stance, dataset, ...extra columns
"""

from __future__ import annotations

import argparse
import logging
import subprocess
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

VALUEPRISM_URL = "hf://datasets/allenai/ValuePrism/full/full.csv"
VALUEPRISM_STANCES = ["Supports", "Opposes", "Either"]
VALUEPRISM_MIN_INSTANCES = 10
VALUEPRISM_TOP_N = 3000


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


def _keep_questions_with_all_stances(merged: pd.DataFrame) -> pd.DataFrame:
    """Keep only questions where at least one rater agreed, disagreed, AND was neutral."""
    def has_all_stances(group: pd.Series) -> bool:
        return (
            group.isin(AGREE_LABELS).any()
            and group.isin(DISAGREE_LABELS).any()
            and group.isin(NEUTRAL_LABELS).any()
        )

    is_valid = merged.groupby("question.id")["ratings.agreement_position"].agg(has_all_stances)
    valid_ids = is_valid[is_valid].index
    return merged[merged["question.id"].isin(valid_ids)]


def prepare_habermas(output_dir: Path) -> None:
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

    filtered = _keep_questions_with_all_stances(merged).copy()
    filtered["text_id"] = [f"HB_{idx}" for idx in filtered.index]
    filtered["question.id"] = filtered["question.id"].str.replace("S", "HB_", regex=False)

    filtered = filtered.rename(
        columns={
            "question.id": "situation_id",
            "metadata.participant_id": "hb_participant_id",
            "question.affirming_statement": "situation",
            "ratings.agreement_position": "stance",
            "own_opinion.text": "text",
        }
    )
    filtered["dataset"] = "habermas"

    out_cols = ["situation_id", "text_id", "situation", "stance", "hb_participant_id", "dataset"]
    filtered = filtered[out_cols]

    out_path = output_dir / "habermas_sample.csv"
    filtered.to_csv(out_path, index=False)
    log.info("Wrote %d rows to %s", len(filtered), out_path)


# --------------------------------------------------------------------------- #
# ValuePrism
# --------------------------------------------------------------------------- #

def _select_balanced_situations(df: pd.DataFrame) -> pd.Index:
    """Pick the `VALUEPRISM_TOP_N` situations with the most balanced stance mix."""
    counts = (
        df.groupby(["situation", "valence"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=VALUEPRISM_STANCES, fill_value=0)
    )
    counts["n"] = counts[VALUEPRISM_STANCES].sum(axis=1)

    valid = counts[
        (counts["n"] >= VALUEPRISM_MIN_INSTANCES) & (counts[VALUEPRISM_STANCES].gt(0).all(axis=1))
    ].copy()

    proportions = valid[VALUEPRISM_STANCES].div(valid["n"], axis=0)
    target = 1 / len(VALUEPRISM_STANCES)

    # L1 distance from a uniform stance distribution -> balance score in [0, 1].
    uniformity_distance = proportions.sub(target).abs().sum(axis=1)
    valid["balance_score"] = 1 - uniformity_distance / (2 * (1 - target))

    top = valid.nlargest(VALUEPRISM_TOP_N, "balance_score")
    return top.index


def prepare_valueprism(output_dir: Path) -> None:
    """Download, filter, and save the ValuePrism dataset."""
    log.info("Preparing ValuePrism dataset")
    df = pd.read_csv(VALUEPRISM_URL, dtype={"vrd": str, "text": str})

    balanced_situations = _select_balanced_situations(df)
    df = df[df["situation"].isin(balanced_situations)].copy()

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
    log.info("Wrote %d rows to %s", len(df), out_path)


# --------------------------------------------------------------------------- #
# Humanual Opinion
# --------------------------------------------------------------------------- #

def prepare_humanual_opinion(output_dir: Path) -> None:
    """Placeholder: Humanual Opinion dataset is not yet implemented."""
    log.warning("Humanual Opinion dataset preparation is not yet implemented — skipping")


DATASET_PREPARERS = {
    "habermas": prepare_habermas,
    "valueprism": prepare_valueprism,
    "humanual": prepare_humanual_opinion,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir", type=Path, default=Path("data/"),
        help="Directory to write the filtered CSV files to.",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASET_PREPARERS), default=list(DATASET_PREPARERS),
        help="Which datasets to prepare (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        try:
            DATASET_PREPARERS[name](args.output_dir)
        except Exception:
            log.exception("Failed to prepare dataset %r", name)
            raise


if __name__ == "__main__":
    main()