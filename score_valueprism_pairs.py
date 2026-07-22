import argparse
import pandas as pd
import torch
from tqdm import tqdm

from utils.models_utils import REGISTRY, load_model
from utils.prompts_utils import build_messages, apply_chat_template_safe
from utils.generation_utils import batch_iterable
from utils.scoring_utils import (
    add_common_args,
    resolve_output_path,
    load_generation_df,
    build_pairs,
    build_direction_prompts,
    tokenize_for_scoring,
)


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--max_new_tokens", type=int, default=5, help="Max new tokens to generate per opinion")
    return parser.parse_args()


def main():
    args = parse_command_line_args()

    gen_df = load_generation_df(args)
    pairs_df = build_pairs(gen_df)

    spec = REGISTRY[args.judge_model_id]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, proc = load_model(spec)
    tok = proc.tokenizer if spec.is_vlm else proc
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    output_path = resolve_output_path(args, script_name="greedy")
    effective_batch_size = max(1, args.batch_size // spec.batch_size_divide)
    print(f"Judge: {spec.name} (key '{args.judge_model_id}') | effective batch size: {effective_batch_size} | device: {device}")

    entries = build_direction_prompts(pairs_df, args.scoring_prompt_version) # List of tuples: (direction, id1, id2, prompt)
    prompts = [e[3] for e in entries]

    messages_list = [build_messages(p, spec, want_thinking=False) for p in prompts]
    texts = [apply_chat_template_safe(tok, m, spec, want_thinking=False) for m in messages_list]

    generations = [None] * len(texts)
    indices = list(range(len(texts)))

    def save_progress():
        # Even indices are the 1->2 direction, odd indices are 2->1.
        pairs_df["generated_score_1to2"] = pd.Series(generations[0::2])
        pairs_df["generated_score_2to1"] = pd.Series(generations[1::2])
        pairs_df["judge_model_used"] = spec.name
        pairs_df["generation_model_id"] = args.generation_model_id
        pairs_df["generation_prompt_version"] = args.generation_prompt_version
        pairs_df.to_csv(output_path, index=False)

    for batch_idx in tqdm(list(batch_iterable(indices, effective_batch_size)), desc=f"Generating ({args.judge_model_id})"):
        batch_texts = [texts[i] for i in batch_idx]
        inputs = tokenize_for_scoring(tok, proc, spec, batch_texts, device)

        # If limit is specified, print the first input fully for debugging
        if args.limit and batch_idx[0] == 0:
            print("First input batch (for debugging):")
            print(tok.batch_decode(inputs["input_ids"], skip_special_tokens=True))

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, input_len:]
        decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)

        for i, text in zip(batch_idx, decoded):
            generations[i] = text.strip()

    save_progress()
    print(f"Saved {len(pairs_df)} pairs (with 1->2 and 2->1 scores) to {output_path}")

    valid = {"1", "2", "3", "4"}
    compliant = pairs_df["generated_score_1to2"].isin(valid) & pairs_df["generated_score_2to1"].isin(valid)
    print(f"Clean single-digit compliance: {compliant.mean():.1%} of pairs "
          f"({(~compliant).sum()} pairs had a non-digit or malformed response in at "
          f"least one direction).")

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