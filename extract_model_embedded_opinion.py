import argparse
import json
import pandas as pd
import torch
from tqdm import tqdm
from utils.models_utils import load_model, REGISTRY
from pathlib import Path

from utils.prompts_utils import build_messages, apply_chat_template_safe, ALIGNMENT_PROMPTS
from utils.generation_utils import batch_iterable
import re
import os


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_model_id", type=str, default="llama-3.2-1b", help="Model ID to use for judging that we need to extract the embedded opinion from")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Max new tokens to generate per opinion")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of times to repeat the generation for each situation")

    parser.add_argument("--path_dataset", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default="model_alignment", help="Directory to save the output CSV and JSON files")
    parser.add_argument("--alignment_prompt_version", type=str, default="impartial_evaluator", help="Version of the prompt to use for parsing generations")
    return parser.parse_args()


LABEL_MAPPING = {
    "impartial_evaluator": ["ACCEPTABLE", "UNACCEPTABLE", "AMBIGUOUS"],
    "acceptability_evaluator": ["ACCEPTABLE", "UNACCEPTABLE", "AMBIGUOUS"],
    "choice": ["YES", "NO", "UNSURE"],
    "agree_disagree": ["AGREE", "DISAGREE", "NEUTRAL"],
    "personal_stance": ["I WOULD", "I WOULD NOT", "UNSURE"],
    "societal_norm": ["ACCEPTED", "CONDEMNED", "MIXED"],
    "likert_5": ["STRONGLY_DISAGREE", "DISAGREE", "NEUTRAL", "AGREE", "STRONGLY_AGREE"],
    "permissibility": ["PERMISSIBLE", "IMPERMISSIBLE", "AMBIGUOUS"],
    "condemnation": ["CONDEMN", "NOT_CONDEMN", "MIXED"],
    "gut_reaction": ["POSITIVE", "NEGATIVE", "MIXED"],
}

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
    # Sort longest-first so a longer label is tried before a shorter one that
    # could match as a prefix/substring of it.
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


def main():
    args = parse_command_line_args()
    os.makedirs(args.output_dir / args.judge_model_id, exist_ok=True)

    spec = REGISTRY[args.judge_model_id]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    situations, situation_id_lookup = get_situations(args)
    model, proc = load_model(spec)
    tok = proc.tokenizer if spec.is_vlm else proc

    effective_batch_size = max(1, args.batch_size // spec.batch_size_divide)
    print(f"Model: {spec.name} (key '{args.judge_model_id}') | effective batch size: {effective_batch_size} | device: {device}")

    situations, prompts = [[situation] * args.n_samples for situation in situations], [prepare_alignment_prompt(situation, args.alignment_prompt_version, args.n_samples) for situation in situations]

    # Flatten the lists of situations and prompts
    situations = [item for sublist in situations for item in sublist]
    prompts = [item for sublist in prompts for item in sublist]
    messages_list = [build_messages(p, spec, want_thinking=False) for p in prompts]
    generations = [None] * len(prompts)
    indices = list(range(len(prompts)))

    for batch_idx in tqdm(list(batch_iterable(indices, effective_batch_size)), desc=f"Generating ({args.judge_model_id})"):
        batch_messages = [messages_list[i] for i in batch_idx]
        batch_texts = [apply_chat_template_safe(tok, m, spec, want_thinking=False) for m in batch_messages]

        if spec.is_vlm:
            # Text-only call: no images passed, so the processor just does tokenization + padding.
            inputs = proc(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        else:
            inputs = tok(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=1.0,
                top_k=0,
                pad_token_id=tok.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, input_len:]
        decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)

        for i, text in zip(batch_idx, decoded):
            generations[i] = text.strip()

    parsed_generations = parse_generations(generations, args.alignment_prompt_version)
    # Save the raw generations and parsed evaluations to a CSV file for further analysis
    df = pd.DataFrame({
        "situation": situations,
        "generation": generations,
        "parsed_evaluations": parsed_generations
    })

    path_csv = f"{args.output_dir}/{args.judge_model_id}/{args.alignment_prompt_version}_generations.csv"
    print(f"Saving generations and parsed evaluations to {path_csv}")
    #create the directory if it doesn't exist
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    df.to_csv(path_csv, encoding="utf-8", index=False)

    # Compute the distribution over labels for each situation. This now
    # supports an arbitrary number of labels (as defined in LABEL_MAPPING),
    # not just a binary pair, so prompts like "likert_5" work without any
    # special-casing here.
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

    # for info, remove later
    if torch.cuda.is_available():
        total_allocated = sum(
            torch.cuda.memory_allocated(i)
            for i in range(torch.cuda.device_count())
        ) / (1024 ** 2)

        total_reserved = sum(
            torch.cuda.memory_reserved(i)
            for i in range(torch.cuda.device_count())
        ) / (1024 ** 2)

        print(f"Total GPU memory allocated: {total_allocated:.2f} MB")
        print(f"Total GPU memory reserved: {total_reserved:.2f} MB")

    path_distribution_csv = f"{args.output_dir}/{args.judge_model_id}/{args.alignment_prompt_version}.csv"
    print(f"Saving alignment distributions to {path_distribution_csv}")
    distribution_df = pd.DataFrame(output_rows)
    # alignment_distribution is written out via json.dumps so the cell is a
    # valid JSON string (parse back with json.loads(...) rather than eval).
    distribution_df["alignment_distribution"] = distribution_df["alignment_distribution"].apply(json.dumps)
    distribution_df.to_csv(path_distribution_csv, encoding="utf-8", index=False)


if __name__ == "__main__":
    main()