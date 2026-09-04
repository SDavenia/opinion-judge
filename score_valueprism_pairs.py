import argparse
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
from utils.models_utils import REGISTRY, load_model
from utils.prompts_utils import build_messages, apply_chat_template_safe, SCORING_PROMPTS, GENERATION_PROMPTS
from utils.generation_utils import batch_iterable
from utils.scoring_utils import (
    
    resolve_output_path,
    load_generation_df,
    build_pairs,
    build_direction_prompts,
    tokenize_for_scoring,
    parse_generation_scoring,
    expand_pairs_for_variations
)


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--judge_model_id", type=str, required=True, choices=list(REGISTRY.keys()),
            help="Registry key for the model used as the LLM-as-judge (scorer).",
        )
    parser.add_argument(
        "--generation_model_id", type=str, choices=list(REGISTRY.keys()),
        help="Registry key for the model whose generated opinions to load and score.",
    )
    parser.add_argument(
        "--generation_prompt_version", type=str, default="base", choices=list(GENERATION_PROMPTS.keys()),
        help="Which generation-prompt style's opinions to load and score (must match "
                "what's actually in the generation data).",
    )
    parser.add_argument(
        "--scoring_prompt_version", type=str, default="base", choices=list(SCORING_PROMPTS.keys()),
        help="Version of the judge prompt to use.",
    )
    parser.add_argument("--generation_csv_path", type=str, default=None, help="Optional path to a CSV of generated opinions to score (if not using the default path).")
    parser.add_argument("--batch_size", type=int, default=32, help="Global batch size before per-model scaling")
    parser.add_argument("--output_dir", type=str, default="output_scores", help="Where to save results")
    parser.add_argument("--num_examples", type=int, default=None, help="Optional limit on number of examples to score (for debugging)")
    parser.add_argument("--option_setting", type=str, default="four", 
                        choices=["four", "0_1"], help="The setting for the options used in the scoring part.")

    parser.add_argument("--extract_ids_from", type=Path,
                        default=None,
                        help="Path of .csv file from where to extract the ids of the pairs to score.")
    
    parser.add_argument("--max_new_tokens", type=int, default=5, help="Max new tokens to generate per opinion")
    return parser.parse_args()


def main():
    args = parse_command_line_args()

    gen_df = load_generation_df(args)
    pairs_df = build_pairs(gen_df)
    if args.num_examples is not None:
        pairs_df = pairs_df.head(args.num_examples)

    if args.extract_ids_from is not None:
        reference_df_ids = pd.read_csv(args.extract_ids_from)
        #now we filter pairs_df using the couples in reference_df_ids, meaning that we select a row r only if there is a row x for which r["id_1"] == x["id_1"] and r["id_2"] == x["id_2"]
        pairs_df = pairs_df.merge(
    reference_df_ids[["id_1", "id_2"]],
    on=["id_1", "id_2"],
    how="inner"
)

    spec = REGISTRY[args.judge_model_id]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, proc = load_model(spec)
    tok = proc.tokenizer if spec.is_vlm else proc
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    output_path = resolve_output_path(args)
    
    effective_batch_size = max(1, args.batch_size // spec.batch_size_divide)
    print(f"Judge: {spec.name} (key '{args.judge_model_id}') | effective batch size: {effective_batch_size} | device: {device}")

    entries = build_direction_prompts(pairs_df, args.scoring_prompt_version) # (variation, direction, id1, id2, prompt)
    prompts = [e[4] for e in entries]   # was e[3]

    messages_list = [build_messages(p, spec, want_thinking=False) for p in prompts]
    texts = [apply_chat_template_safe(tok, m, spec, want_thinking=False) for m in messages_list]

    generations = [None] * len(texts)
    indices = list(range(len(texts)))

    # Expand pairs_df to one row per (pair, variation) BEFORE save_progress,
    # so it lines up with the 0::2 / 1::2 slicing of `generations`.
    pairs_df = expand_pairs_for_variations(pairs_df, args.scoring_prompt_version)

    def save_progress():
        # Even indices are the 1->2 direction, odd indices are 2->1, within each
        # (row, variation) block -- expand_pairs_for_variations already put
        # pairs_df rows in matching order.
        pairs_df["generated_score_1to2"] = pd.Series(generations[0::2])
        pairs_df["generated_score_2to1"] = pd.Series(generations[1::2])
        pairs_df["judge_model"] = spec.name
        pairs_df["generation_model"] = args.generation_model_id
        pairs_df["generation_prompt_version"] = args.generation_prompt_version
        pairs_df["parsed_score_1to2"] = pairs_df["generated_score_1to2"].apply(parse_generation_scoring, option_setting=args.option_setting)
        pairs_df["parsed_score_2to1"] = pairs_df["generated_score_2to1"].apply(parse_generation_scoring, option_setting=args.option_setting)

        pairs_df.to_csv(output_path, index=False)

    for batch_idx in tqdm(list(batch_iterable(indices, effective_batch_size)), desc=f"Generating ({args.judge_model_id})"):
        batch_texts = [texts[i] for i in batch_idx]
        inputs = tokenize_for_scoring(tok, proc, spec, batch_texts, device)

        

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