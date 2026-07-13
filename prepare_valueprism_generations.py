import pandas as pd
import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Model ID for the tokenizer and model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing the data")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="Max new tokens to generate per opinion")
    parser.add_argument("--output_path", type=str, default="data/valueprism_generations.csv", help="Where to save results")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for debugging")
    return parser.parse_args()


def generate_controlled_opinion(row):
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

    prompt = f"""You are a precise text generation engine for an NLP evaluation dataset.
    Situation: {situation}
    Moral Principle: {value}
    Target Relationship to Situation: {valence}
    Core Rationale: {explanation}
    Task:
    {stance_instruction}
    
    Constraints:
    1. The output must be exactly 2-3 sentences long.
    2. Write in a natural, first-person or third-person argumentative tone (as if written by a human expressing a genuine opinion).
    3. Do not explicitly mention the words "Valence", "Core Rationale", or quote the instructions. Integrate the Core Rationale seamlessly into the stance.
    Output only the opinion text."""
    return prompt


def batch_iterable(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def main():
    args = parse_command_line_args()
    valueprism_df = pd.read_csv("data/valueprism_data.csv")

    if args.limit:
        valueprism_df = valueprism_df.head(args.limit).copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    # Build all prompts up front
    prompts = [generate_controlled_opinion(row) for _, row in valueprism_df.iterrows()]

    generations = [None] * len(prompts)

    indices = list(range(len(prompts)))

    for batch_idx in tqdm(list(batch_iterable(indices, args.batch_size)), desc="Generating"):
        batch_prompts = [prompts[i] for i in batch_idx]

        # Use chat template if available, else fall back to raw prompt
        if tokenizer.chat_template is not None:
            batch_texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in batch_prompts
            ]
        else:
            batch_texts = batch_prompts

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Slice off the input portion so we only decode newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, input_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for i, text in zip(batch_idx, decoded):
            generations[i] = text.strip()

        # Incremental save in case of crash on long runs
        valueprism_df.loc[:len(generations) - 1, "generated_opinion"] = pd.Series(generations)
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        valueprism_df.to_csv(args.output_path, index=False)

    valueprism_df["generated_opinion"] = generations
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    valueprism_df.to_csv(args.output_path, index=False)
    print(f"Saved {len(valueprism_df)} generations to {args.output_path}")


if __name__ == "__main__":
    main()