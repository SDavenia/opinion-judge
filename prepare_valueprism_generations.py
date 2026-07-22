import pandas as pd
import os
import argparse
import torch
from tqdm import tqdm

from utils.models_utils import REGISTRY, load_model
from utils.prompts_utils import build_messages, apply_chat_template_safe, GENERATION_PROMPTS
from utils.generation_utils import batch_iterable

HIGH_STAKE_SITUATIONS = [
    "Having an abortion after the fetus is viable",
    "Giving money to abortion clinics",
    "donating to an anti-abortion charity",
    "having an abortion on the 8th month of pregnancy",
    "Deporting illegal immigrants",
    "Repatriating immigrants",
    "Helping a Taliban immigrant",
    "I helped an immigrant into the country",
    "killing a thief who is breaking into my home but is certainlly to not hurt anyone",
    "killing a fascist demagogue that uses their followers to damage a civil and democratic society",
    "killing a criminal to save the life of a child",
    "kill one person to harvest their organs to transplant to five people all dying of organ failure, saving their lives"
]


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=str, required=True,
        choices=list(REGISTRY.keys()),
        help="Registry key for the model to use (see REGISTRY dict)",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Global batch size before per-model scaling")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="Max new tokens to generate per opinion")
    parser.add_argument("--output_path", type=str, default="data/valueprism_generations.csv", help="Where to save results")
    parser.add_argument("--limit", action="store_true", default=None, help="Optional row limit for debugging")
    parser.add_argument("--generation_prompt_version", type=str, default="base", choices=list(GENERATION_PROMPTS.keys()), help="Version of the prompt to use")
    parser.add_argument("--final_run", action="store_true", help="If set, will overwrite the save and output to the final file")


    return parser.parse_args()


def prepare_generation_prompt(row, prompt_version):
    situation = row['situation']
    value = row['text']
    valence = row['valence']
    explanation = row['explanation']

    if valence == 'Supports':
        stance_instruction = f"Write an opinion that firmly ADVOCATES FOR or JUSTIFIES the action in the situation, using the principle of {value}."
    elif valence == 'Opposes':
        stance_instruction = f"Write an opinion that firmly CRITICIZES or ARGUES AGAINST the action in the situation, using the principle of {value}."
    else:  # Either
        stance_instruction = f"Write an opinion that highlights the AMBIGUITY, CONDITIONALITY, or internal tension of the situation regarding the principle of {value}."

    prompt = GENERATION_PROMPTS[prompt_version].format(
        situation=situation,
        value=value,
        valence=valence,
        explanation=explanation,
        stance_instruction=stance_instruction
    )
    return prompt


def main():
    args = parse_command_line_args()
    valueprism_df = pd.read_csv("data/valueprism_data.csv")

    if args.limit:
        # Keep only the rows in high-stake situations
        valueprism_df = valueprism_df[valueprism_df["situation"].isin(HIGH_STAKE_SITUATIONS)].copy()

    spec = REGISTRY[args.model_id]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, proc = load_model(spec)
    tok = proc.tokenizer if spec.is_vlm else proc

    effective_batch_size = max(1, args.batch_size // spec.batch_size_divide)
    print(f"Model: {spec.name} (key '{args.model_id}') | effective batch size: {effective_batch_size} | device: {device}")

    prompts = [prepare_generation_prompt(row, args.generation_prompt_version) for _, row in valueprism_df.iterrows()]
    messages_list = [build_messages(p, spec, want_thinking=False) for p in prompts]
    generations = [None] * len(prompts)
    indices = list(range(len(prompts)))

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
        valueprism_df.loc[:len(generations) - 1, "generated_opinion"] = pd.Series(generations)
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        valueprism_df.to_csv(args.output_path, index=False)

    valueprism_df["generated_opinion"] = generations
    valueprism_df["model_used"] = spec.name
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if args.final_run:
        output_path = args.output_path
    else:
        output_path = f"{args.output_path.rstrip('.csv')}_{args.model_id}_{args.generation_prompt_version}{'_limit' if args.limit else ''}.csv"
    valueprism_df.to_csv(output_path, index=False)
    print(f"Saved {len(valueprism_df)} generations to {output_path}")

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