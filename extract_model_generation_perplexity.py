import argparse
import difflib
import os
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from utils.models_utils import REGISTRY, load_model
from utils.prompts_utils import build_messages, apply_chat_template_safe
from utils.generation_utils import batch_iterable

PERPLEXITY_PROMPT_TEMPLATES = {
    "base": """ "Situation: {situation}\n\n"
"Write a short, honest personal opinion (2-3 sentences) about this situation.\n\n"
"Opinion:"
"""
}


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_model_id", type=str, default="llama-3.2-1b",
                         help="Model ID to use for generating the embedded opinion")
    parser.add_argument("--generation_prompt_version", type=str)
    parser.add_argument("--generation_dir", type=Path, default="generations/")
    parser.add_argument("--judge_model_id", type=str, default="llama-3.2-1b",
                         help="Model ID to use for judging that we need to extract the embedded opinion from")
    
    parser.add_argument("--perplexity_prompt", type=str, default="base")

    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for generation")
    parser.add_argument("--max_length", type=int, default=512,
                         help="Max token length (prompt + opinion) before truncation.")
    parser.add_argument("--output_dir", type=Path, default="model_alignment_perplexity",
                         help="Where to write the CSV with a 'perplexity' column.")
    args = parser.parse_args()
    return args


def get_situations_and_generations(args):
    generation_df_path = args.generation_dir / f"{args.generation_model_id}_{args.generation_prompt_version}.csv"
    generation_df = pd.read_csv(generation_df_path)
    return generation_df

def _text_similarity(a, b):
    """Cheap character-level similarity ratio in [0, 1], used only to flag likely
    misalignment (e.g. tokenizer re-merging across the prefix/opinion boundary),
    not as a precise metric."""
    return difflib.SequenceMatcher(None, a.strip(), b.strip()).ratio()


@torch.no_grad()
def compute_perplexities(model, tok, spec, situations, opinions, device, perplexity_prompt, max_length=512,
                         batch_idx=0):
    """
    For each (situation, opinion) pair, compute the perplexity of `opinion` under `model`
    when conditioned on a neutral, stance-free prompt built from `situation`.

    Method (teacher forcing):
      1. Build the chat-templated prefix (situation + neutral instruction), the same way
         the model would see it if it were about to generate.
      2. Concatenate prefix + opinion, run a single forward pass over the full sequence.
      3. Compute the mean negative log-likelihood of ONLY the opinion tokens.
      4. perplexity = exp(mean NLL).

    Padding/masking is done manually (not via HF's built-in `labels` loss) because:
      - the tokenizer uses left-padding globally (set in load_model), so the prefix/opinion
        boundary shifts per-example once padded;
      - HF's built-in loss averages over the whole batch, which would hide per-example scores.
    """
    prompts = [PERPLEXITY_PROMPT_TEMPLATES[perplexity_prompt].format(situation=s) for s in situations]

    # Chat-template the prefixes so the opinion is scored under the same formatting
    # (role tokens, generation-prompt suffix, etc.) the model would actually see.
    prefixes = []
    for prompt in prompts:
        messages = build_messages(prompt, spec, want_thinking=False)
        prefix_text = apply_chat_template_safe(tok, messages, spec, want_thinking=False)
        prefixes.append(prefix_text)

    # Leading space avoids the opinion's first token merging with the prefix's last token
    # under BPE tokenization.
    full_texts = [prefix + " " + opinion for prefix, opinion in zip(prefixes, opinions)]

    # Tokenize prefixes alone (unpadded) just to learn, per example, how many tokens
    # belong to the prefix once the concatenated text is tokenized.
    prefix_lens = [
        len(tok(prefix, add_special_tokens=False)["input_ids"])
        for prefix in prefixes
    ]

    encodings = tok(
        full_texts,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    batch_size, seq_len = input_ids.shape

    # label_mask is True only where a token is (a) real, non-padding, and (b) part of the
    # opinion continuation (not the prefix).
    label_mask = attention_mask.clone().bool()
    if tok.padding_side == "left":
        true_lengths = attention_mask.sum(dim=1)
        pad_amounts = seq_len - true_lengths
        for i in range(batch_size):
            prefix_end = pad_amounts[i].item() + prefix_lens[i]
            label_mask[i, :prefix_end] = False
    else:
        for i in range(batch_size):
            label_mask[i, :prefix_lens[i]] = False

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (batch, seq_len, vocab)

    # Standard causal-LM shift: logits at position t predict the token at t+1.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_label_mask = label_mask[:, 1:].contiguous()

    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * shift_label_mask

    token_counts = shift_label_mask.sum(dim=1).clamp(min=1)
    mean_nll = -(token_log_probs.sum(dim=1) / token_counts)
    perplexities = torch.exp(mean_nll)

    # If an opinion got fully truncated away, report NaN instead of a misleading exp(0)=1.
    empty_mask = shift_label_mask.sum(dim=1) == 0
    perplexities[empty_mask] = float("nan")

    return perplexities.cpu().tolist()


def run_perplexity_pipeline(args, df):
    spec = REGISTRY[args.judge_model_id]
    model, proc = load_model(spec)
    tok = proc.tokenizer if spec.is_vlm else proc
    device = next(model.parameters()).device

    situations = df["situation"].tolist()
    opinions = df["text"].tolist()

    all_perplexities = []
    batches = list(zip(
        batch_iterable(situations, args.batch_size),
        batch_iterable(opinions, args.batch_size),
    ))
    for batch_idx, (batch_situations, batch_opinions) in enumerate(
        tqdm(batches, desc=f"Computing perplexity with {args.judge_model_id}")
    ):
        batch_ppls = compute_perplexities(
            model, tok, spec, batch_situations, batch_opinions, device, args.perplexity_prompt,
            max_length=args.max_length, batch_idx=batch_idx,
        )
        all_perplexities.extend(batch_ppls)

    result_df = df.copy()
    result_df["perplexity"] = all_perplexities
    return result_df


def main():
    args = parse_command_line_arguments()
    df = get_situations_and_generations(args)
    df = df.head(5)
    assert len(df["text"].unique()) == len(df), "There are duplicate generated opinions in the dataframe. Please check the data."

    result_df = run_perplexity_pipeline(args, df)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = args.output_dir / f"generator_{args.generation_model_id}_{args.generation_prompt_version}/{args.judge_model_id}_{args.perplexity_prompt}.csv" 
    result_df.to_csv(output_path, index=False)
    print(f"Saved perplexity scores to {output_path}")


if __name__ == "__main__":
    main()