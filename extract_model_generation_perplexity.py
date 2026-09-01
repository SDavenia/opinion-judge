import argparse
import difflib

import torch
import pandas as pd
from tqdm import tqdm

from utils.models_utils import REGISTRY, load_model
from utils.prompts_utils import build_messages, apply_chat_template_safe
from utils.generation_utils import batch_iterable

PATH_GENERATIONS = "generations/output_llama-3.3-70b_reflective_person.csv"

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
    parser.add_argument("--judge_model_id", type=str, default="llama-3.2-1b",
                         help="Model ID to use for judging that we need to extract the embedded opinion from")
    parser.add_argument("--perplexity_prompt", type=str, default="base")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for generation")
    parser.add_argument("--max_length", type=int, default=512,
                         help="Max token length (prompt + opinion) before truncation.")
    parser.add_argument("--output_dir", type=str, default="model_alignment_perplexity",
                         help="Where to write the CSV with a 'perplexity' column. "
                              "Defaults to perplexity_<perplexity_model_id>.csv")

    parser.add_argument("--situations", type=str, choices=[None, "first_n", "random_selected", "sampled_situations_trial"])
    parser.add_argument("--n_situations", type=int, default=10, help="Number of situations to select if situations is first_n or random_selected")

    parser.add_argument("--debug", action="store_true",
                         help="Print per-example diagnostics (padding layout, prefix/opinion boundary, "
                              "decoded masked tokens vs. original opinion text) to verify masking is correct.")
    parser.add_argument("--debug_batches", type=int, default=1,
                         help="Number of leading batches to print full debug output for when --debug is set. "
                              "Kept small by default so logs don't explode on large runs.")
    args = parser.parse_args()
    return args


def get_situations_and_generations(args):
    if args.situations == "first_n":
        if args.n_situations is None or args.n_situations <= 0:
            raise ValueError(f"When using 'first_n' situations, you have to set 'n_situations, meanwhile you set: {args.n_situations}")

        generations_df = pd.read_csv(PATH_GENERATIONS, encoding="utf-8")
        if args.n_situations > len(generations_df):
            raise ValueError(f"Requested n_situations ({args.n_situations}) is greater than the number of available situations ({len(generations_df)}) in the dataset.")
        return generations_df[["situation", "generated_opinion", "valence", "text"]].head(args.n_situations)
    elif args.situations == "random_selected":
        generations_df = pd.read_csv(f"generations/output_{args.generation_model_id}_reflective_person.csv", encoding="utf-8")
        
        return generations_df[["situation", "generated_opinion", "valence", "text"]]

    elif args.situations == "sampled_situations_trial":
        generations_df = pd.read_csv("data/sampled.csv")
        id1 = (
            generations_df[["id_1", "situation", "generated_opinion_1", "valence_1", "text_1"]]
            .drop_duplicates(subset="id_1", keep="first")
            .rename(columns={
                "id_1": "generation_id",
                "generated_opinion_1": "generated_opinion",
                "valence_1": "valence",
                "text_1": "text"
            })
        )
        id2 = (
            generations_df[["id_2", "situation", "generated_opinion_2", "valence_2", "text_2"]]
            .drop_duplicates(subset="id_2", keep="first")
            .rename(columns={"id_2": "generation_id", 
                             "valence_2": "valence",
                             "text_2": "text",
                             "generated_opinion_2": "generated_opinion"})
        )

        id2 = id2[~id2["generation_id"].isin(id1["generation_id"])]
        result = pd.concat([id1, id2], ignore_index=True)
        result = (
            result
            .drop_duplicates(subset="generation_id", keep="first")
            .reset_index(drop=True)
        )

        return result[["situation", "generated_opinion", "valence", "text"]]
    else:
        raise ValueError(f"Invalid situations argument: {args.situations}. Must be 'high_stake' or 'first_n'.")


def _text_similarity(a, b):
    """Cheap character-level similarity ratio in [0, 1], used only to flag likely
    misalignment (e.g. tokenizer re-merging across the prefix/opinion boundary),
    not as a precise metric."""
    return difflib.SequenceMatcher(None, a.strip(), b.strip()).ratio()


def _debug_report_batch(batch_idx, tok, situations, opinions, input_ids, attention_mask,
                         label_mask, prefix_lens, shift_label_mask, token_counts, perplexities,
                         max_length):
    """
    Per-example sanity checks, printed to stdout. Verifies:
      1. Padding layout matches expectations (left-padded, attention_mask consistent).
      2. The prefix/opinion boundary lands where we think it does.
      3. The tokens actually selected by label_mask decode back to (approximately)
         the original opinion text -- catches off-by-one boundary bugs and
         tokenizer re-merging across the concatenation point.
      4. Pre-shift and post-shift mask counts agree (sanity check on the causal shift).
    """
    batch_size, seq_len = input_ids.shape
    print(f"\n{'=' * 80}\n[DEBUG] Batch {batch_idx}  (batch_size={batch_size}, seq_len={seq_len})\n{'=' * 80}")

    for i in range(batch_size):
        true_len = int(attention_mask[i].sum().item())
        pad_amount = seq_len - true_len
        # Compare against max_length (the global cap passed to the tokenizer), NOT seq_len
        # (which is just the padded width of *this batch*, i.e. the longest example in it).
        # The longest example in any batch will always have true_len == seq_len, which would
        # make this a false positive for "got truncated" if compared against seq_len instead.
        was_truncated = true_len >= max_length

        print(f"\n--- Example {i} ---")
        print(f"  situation (first 80 chars): {situations[i][:80]!r}")
        print(f"  original opinion:           {opinions[i]!r}")
        print(f"  prefix_len (tokens):        {prefix_lens[i]}")
        print(f"  true_len (non-pad tokens):  {true_len}")
        print(f"  pad_amount:                 {pad_amount}  (padding_side={tok.padding_side})")

        # 1. Padding sanity: with left-padding, all pad positions should be a contiguous
        # block at the start, and attention_mask should be 0 there / 1 elsewhere.
        expected_pad_mask = torch.zeros(seq_len, dtype=torch.bool)
        expected_pad_mask[:pad_amount] = True
        actual_pad_mask = (attention_mask[i] == 0)
        if not torch.equal(expected_pad_mask, actual_pad_mask):
            print("  [WARN] attention_mask padding layout is NOT a clean left-aligned block. "
                  "Check tok.padding_side and that no mid-sequence tokens are being masked unexpectedly.")

        # 2. Decode the full row (including special/pad tokens) so the boundary is visible.
        full_decoded = tok.decode(input_ids[i], skip_special_tokens=False)
        print(f"  full decoded (raw, w/ specials): {full_decoded[:200]!r}{'...' if len(full_decoded) > 200 else ''}")

        # 3. Decode ONLY the tokens selected by label_mask (pre-shift) -- this is exactly
        # the span we intend to score -- and compare it against the original opinion.
        selected_ids = input_ids[i][label_mask[i]]
        decoded_selected = tok.decode(selected_ids, skip_special_tokens=False)
        sim = _text_similarity(decoded_selected, opinions[i])
        flag = "" if sim >= 0.8 else "  [WARN] low similarity -- possible boundary/masking bug"
        print(f"  decoded label_mask span:    {decoded_selected!r}")
        print(f"  similarity to original:     {sim:.2f}{flag}")

        # 4. Pre-shift vs post-shift mask count sanity check. Position 0 is always part of
        # the prefix in our setup, so shifting off the first column should not drop any
        # opinion tokens -- the two counts should match exactly.
        pre_shift_count = int(label_mask[i].sum().item())
        post_shift_count = int(shift_label_mask[i].sum().item())
        if pre_shift_count != post_shift_count:
            print(f"  [WARN] pre-shift opinion-token count ({pre_shift_count}) != "
                  f"post-shift count ({post_shift_count}). This would only happen if the very "
                  f"first token in the sequence were marked as an opinion token, which should "
                  f"never occur (the prefix is never empty).")

        print(f"  token_counts used for mean NLL: {token_counts[i].item()}")
        print(f"  truncated (hit max_length)?     {was_truncated}")
        print(f"  --> perplexity: {perplexities[i].item():.4f}")


@torch.no_grad()
def compute_perplexities(model, tok, spec, situations, opinions, device, perplexity_prompt, max_length=512,
                          debug=False, batch_idx=0):
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

    Set `debug=True` to print per-example diagnostics verifying the padding layout and the
    prefix/opinion boundary are correct (see `_debug_report_batch`).
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

    if debug:
        _debug_report_batch(
            batch_idx=batch_idx,
            tok=tok,
            situations=situations,
            opinions=opinions,
            input_ids=input_ids,
            attention_mask=attention_mask,
            label_mask=label_mask,
            prefix_lens=prefix_lens,
            shift_label_mask=shift_label_mask,
            token_counts=token_counts,
            perplexities=perplexities,
            max_length=max_length,
        )

    return perplexities.cpu().tolist()


def run_perplexity_pipeline(args, df):
    spec = REGISTRY[args.judge_model_id]
    model, proc = load_model(spec)
    tok = proc.tokenizer if spec.is_vlm else proc
    device = next(model.parameters()).device

    situations = df["situation"].tolist()
    opinions = df["generated_opinion"].tolist()

    all_perplexities = []
    batches = list(zip(
        batch_iterable(situations, args.batch_size),
        batch_iterable(opinions, args.batch_size),
    ))
    for batch_idx, (batch_situations, batch_opinions) in enumerate(
        tqdm(batches, desc=f"Computing perplexity with {args.judge_model_id}")
    ):
        batch_debug = args.debug and batch_idx < args.debug_batches
        batch_ppls = compute_perplexities(
            model, tok, spec, batch_situations, batch_opinions, device, args.perplexity_prompt,
            max_length=args.max_length, debug=batch_debug, batch_idx=batch_idx,
        )
        all_perplexities.extend(batch_ppls)

    result_df = df.copy()
    result_df["perplexity"] = all_perplexities
    return result_df


def main():
    args = parse_command_line_arguments()
    df = get_situations_and_generations(args)
    assert len(df["generated_opinion"].unique()) == len(df), "There are duplicate generated opinions in the dataframe. Please check the data."

    result_df = run_perplexity_pipeline(args, df)

    output_path = f"{args.output_dir}/{args.judge_model_id}_{args.perplexity_prompt}_perplexities.csv" 
    result_df.to_csv(output_path, index=False)
    print(f"Saved perplexity scores to {output_path}")


if __name__ == "__main__":
    main()