"""
Score a generations CSV (as produced by the opinion-generation pipeline)
with siebert/sentiment-roberta-large-english, applied only to the generated
opinion text (the `text` column) -- NOT the situation.

For each row we extract a continuous 0-1 "sentiment_score" (softmax prob
mass on the "POSITIVE" label vs "NEGATIVE"), plus the greedy label, the
same way extract_generations_llamaguard.py reports unsafe_score/label.

Usage:
    python extract_generations_sentiment.py \
        --dataset_id valueprism \
        --generation_model_id gemma3 --generation_prompt_version base \
        --output_dir sentiment_scores/ --batch_size 32

    python extract_generations_sentiment.py \
        --dataset_id habermas \
        --output_dir sentiment_scores/ --batch_size 32

--dataset_id selects the input generations CSV (see resolve_path_generations)
and is appended as a subdirectory to --output_dir (matching score.py /
extract_model_embedded_opinion.py / extract_generations_llamaguard.py's
output layout), unless --path_generations is given explicitly.

Requires: transformers, torch, pandas, tqdm
    pip install transformers torch pandas tqdm --break-system-packages
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "siebert/sentiment-roberta-large-english"


class SentimentScorer:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device)
        self.model.eval()

        id2label = {k: v.upper() for k, v in self.model.config.id2label.items()}
        positive_ids = [i for i, label in id2label.items() if label == "POSITIVE"]
        negative_ids = [i for i, label in id2label.items() if label == "NEGATIVE"]
        if len(positive_ids) != 1 or len(negative_ids) != 1:
            raise ValueError(
                f"Expected exactly one POSITIVE and one NEGATIVE label id, got "
                f"id2label={id2label!r}"
            )
        self.positive_id = positive_ids[0]
        self.negative_id = negative_ids[0]

    @torch.no_grad()
    def score_batch(self, texts: list[str], max_length: int = 512) -> list[dict]:
        """
        Returns, for each text, a dict with:
          - sentiment_score: float in [0, 1], softmax prob mass on "POSITIVE"
                              vs "NEGATIVE"
          - label: "POSITIVE" or "NEGATIVE" (argmax)
        """
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**encodings).logits
        two_logits = logits[:, [self.negative_id, self.positive_id]]
        probs = torch.softmax(two_logits.float(), dim=-1)

        results = []
        for pos_prob in probs[:, 1].cpu().tolist():
            results.append({
                "sentiment_score": pos_prob,
                "label": "POSITIVE" if pos_prob >= 0.5 else "NEGATIVE",
            })
        return results


def batch_iterable(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def resolve_path_generations(args) -> Path | None:
    """Returns None (rather than erroring) when valueprism defaults can't be resolved,
    so the caller can report it via parser.error for a consistent CLI error message."""
    if args.path_generations is not None:
        return args.path_generations
    if args.dataset_id == "habermas":
        # Habermas opinions are human-written and ship with 'text' already
        # filled in -- no separate generation step/file, so we just read the
        # prepared dataset CSV directly.
        return Path("data/habermas_sample.csv")
    if args.generation_model_id is None or args.generation_prompt_version is None:
        return None
    return args.generation_dir / f"{args.generation_model_id}_{args.generation_prompt_version}.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_id", type=str, required=True, choices=["habermas", "valueprism"],
        help="Which dataset's generations to score. Selects the default --path_generations "
             "and is appended as a subdirectory to --output_dir.",
    )
    parser.add_argument(
        "--path_generations", type=Path, default=None,
        help="Path to the generations CSV. Defaults to data/habermas_sample.csv for habermas, "
             "or {generation_dir}/{generation_model_id}_{generation_prompt_version}.csv for valueprism.",
    )
    parser.add_argument("--generation_model_id", type=str, default=None,
                         help="Only used for --dataset_id=valueprism to resolve the default --path_generations.")
    parser.add_argument("--generation_prompt_version", type=str, default=None,
                         help="Only used for --dataset_id=valueprism to resolve the default --path_generations.")
    parser.add_argument("--generation_dir", type=Path, default=Path("generations/"))
    parser.add_argument("--output_dir", type=Path, default=Path("sentiment_scores/"), help="Where to save score CSVs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for sentiment scoring")
    parser.add_argument("--num_examples", type=int, default=None, help="Optional: limit to first N examples for testing")
    args = parser.parse_args()
    args.path_generations = resolve_path_generations(args)
    if args.path_generations is None:
        parser.error(
            "--generation_model_id and --generation_prompt_version are required when "
            "--dataset_id=valueprism and --path_generations is not given."
        )
    args.output_dir = args.output_dir / args.dataset_id
    return args


def main():
    args = parse_args()
    df = pd.read_csv(args.path_generations)
    if args.num_examples is not None:
        df = df.head(args.num_examples)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scorer = SentimentScorer()

    texts = df["text"].tolist()
    text_ids = df["text_id"].tolist()

    results = []
    for batch_ids, batch_texts in tqdm(
        list(zip(batch_iterable(text_ids, args.batch_size), batch_iterable(texts, args.batch_size))),
        desc="Scoring opinions",
    ):
        batch_results = scorer.score_batch(batch_texts)
        for text_id, res in zip(batch_ids, batch_results):
            results.append({"text_id": text_id, **res})

    scores_df = pd.DataFrame(results)
    output_path = args.output_dir / "opinion_sentiment_scores.csv"
    scores_df.to_csv(output_path, index=False)
    print(f"Saved {len(scores_df)} sentiment scores to {output_path}")


if __name__ == "__main__":
    main()
