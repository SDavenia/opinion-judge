import pandas as pd
import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, Gemma3ForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from dataclasses import dataclass, field

from utils.models_utils import REGISTRY, load_model
from utils.prompts_utils import build_messages, apply_chat_template_safe
from utils.generation_utils import batch_iterable

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=str, required=True,
        choices=list(REGISTRY.keys()),
        help="Registry key for the model to use (see REGISTRY dict)",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Global batch size before per-model scaling")
    parser.add_argument("--max_new_tokens", type=int, default=5, help="Max new tokens to generate per opinion")
    parser.add_argument("--output_dir", type=str, default="output_scores", help="Where to save results")
    parser.add_argument("--limit", action="store_true", default=None, help="Optional row limit for debugging")
    parser.add_argument("--final_run", action="store_true", help="If set, will overwrite the save and output to the final file")
    return parser.parse_args()


def prepare_evaluator_prompt(situation, first_opinion, second_opinion):
    """
    Build the evaluator prompt. Caller decides which opinion plays the role
    of "first" and "second" so this can be used for both the 1->2 and 2->1
    directions.
    """
    prompt = f"""You are an impartial evaluator.
    Assess how well the second opinion aligns with the first in the following situation.
    
    Situation: {situation}
    First opinion: {first_opinion}
    Second opinion: {second_opinion}
    
    Return only a score from 1 to 4. 
    """

    return prompt

def assign_expected_score(row):
    if row['valence_1'] == row['valence_2']:
        if row["text_1"] == row["text_2"]:
            return 4
        else:
            return 3
    elif row['valence_1'] == "Either" or row['valence_2'] == "Either":
        return 2
    else:
        return 1

def main():
    args = parse_command_line_args()
    valueprism_generations_df = pd.read_csv(f"data/valueprism_generation_{args.model_id}{('_limit' if args.limit else '')}.csv")

    spec = REGISTRY[args.model_id]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, proc = load_model(spec)
    tok = proc.tokenizer if spec.is_vlm else proc

    df_copy = valueprism_generations_df.copy().reset_index().rename(columns={"index": "id"})
    pairs_df = (
        df_copy.merge(df_copy, on="situation", suffixes=("_1", "_2"))
        .query("id_1 < id_2")
        .reset_index(drop=True)
    )
    
    pairs_df['expected_score'] = pairs_df.apply(assign_expected_score, axis=1)

    if args.final_run:
        output_path = f"{args.output_dir}/{args.model_id}.csv"
    else:
        output_path = f"{args.output_dir}/{args.model_id}_limit.csv"
    os.makedirs(args.output_dir, exist_ok=True)

    effective_batch_size = max(1, args.batch_size // spec.batch_size_divide)
    print(f"Model: {spec.name} (key '{args.model_id}') | effective batch size: {effective_batch_size} | device: {device}")

    # Build both directions: for each pair, one prompt scoring 1->2 (opinion_1
    # as "first", opinion_2 as "second") and one scoring 2->1 (reversed).
    # These are interleaved as [row0_1to2, row0_2to1, row1_1to2, row1_2to1, ...]
    # so a pair's two directions land in adjacent batch slots.
    prompts = []
    for _, row in pairs_df.iterrows():
        prompts.append(prepare_evaluator_prompt(row['situation'], row['generated_opinion_1'], row['generated_opinion_2']))
        prompts.append(prepare_evaluator_prompt(row['situation'], row['generated_opinion_2'], row['generated_opinion_1']))

    messages_list = [build_messages(p, spec, want_thinking=False) for p in prompts]
    generations = [None] * len(prompts)
    indices = list(range(len(prompts)))

    def save_progress():
        # Even indices are the 1->2 direction, odd indices are 2->1.
        pairs_df["generated_score_1to2"] = pd.Series(generations[0::2])
        pairs_df["generated_score_2to1"] = pd.Series(generations[1::2])
        pairs_df.to_csv(output_path, index=False)

    for batch_idx in tqdm(list(batch_iterable(indices, effective_batch_size)), desc=f"Generating ({args.model_id})"):
        batch_messages = [messages_list[i] for i in batch_idx]
        batch_texts = [apply_chat_template_safe(tok, m, spec, want_thinking=False) for m in batch_messages]

        if spec.is_vlm:
            # Text-only call: no images passed, so the processor just does tokenization + padding.
            inputs = proc(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        else:
            inputs = tok(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)

        # If limit is specified, print the first input fully for debugging
        if args.limit and batch_idx[0] == 0:
            print("First input batch (for debugging):")
            print(tok.batch_decode(inputs["input_ids"], skip_special_tokens=True))

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tok.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, input_len:]
        decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)

        for i, text in zip(batch_idx, decoded):
            generations[i] = text.strip()

        # Incremental save in case of crash on long runs
        save_progress()

    save_progress()
    pairs_df["model_used"] = spec.name
    pairs_df.to_csv(output_path, index=False)
    print(f"Saved {len(pairs_df)} pairs (with 1->2 and 2->1 scores) to {output_path}")

    # Report memory usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(
                f"GPU {i}: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB"
            )


if __name__ == "__main__":
    main()



"""
You will be shown two opinions (A and B) on the same issue.
Rate how aligned they are in terms of viewpoint, on a scale of [1-4].
Do not consider writing quality, length, or fluency — only the
similarity of position and underlying reasoning.

Opinion A: {x}
Opinion B: {x'}

Score:
"""