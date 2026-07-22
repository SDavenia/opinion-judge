import argparse
import numpy as np
import torch
from tqdm import tqdm

from utils.models_utils import REGISTRY, load_model
from utils.prompts_utils import build_messages, apply_chat_template_safe
from utils.generation_utils import batch_iterable
from utils.scoring_utils import (
    DIGITS,
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
    parser.add_argument(
        "--test", action="store_true",
        help="Diagnostic mode: for a handful of pairs, print the raw (unstripped) "
             "greedy generation next to the logit-based digit prediction, so you can "
             "verify the digit really is the first generated token before trusting "
             "the probability-extraction path. Writes nothing to disk.",
    )
    parser.add_argument("--n_test", type=int, default=20, help="Number of pairs (x2 directions) to inspect in --test mode.")
    parser.add_argument("--max_new_tokens", type=int, default=5, help="Only used in --test mode, for the raw-generation comparison.")
    return parser.parse_args()


def resolve_digit_token_aliases(tok, verbose=True):
    """
    One-time, deterministic, no model calls: scan the full vocabulary and
    collect every token id whose decoded (stripped) text is a bare digit
    '1'-'4'. A single digit can be represented by several distinct token
    ids depending on BPE merges / leading-space conventions (e.g. "1" and
    " 1" may both exist and be usable by the model in different contexts),
    so we keep *all* aliases per digit rather than guessing a single
    canonical id. Downstream, a digit's probability mass is the sum over
    all of its alias ids.

    This deliberately does not depend on the judge model ever actually
    producing a given digit -- it only depends on the tokenizer's vocab,
    so there's no "unseen in calibration sample -> unverified fallback"
    gap, even for scales with more than 4 possible values.
    """
    aliases = {d: [] for d in DIGITS}
    vocab_size = len(tok)
    for tid in range(vocab_size):
        try:
            s = tok.decode([tid])
        except Exception:
            continue
        stripped = s.strip()
        if stripped in DIGITS:
            aliases[stripped].append(tid)

    missing = [d for d in DIGITS if not aliases[d]]
    if missing:
        raise ValueError(
            f"No vocab token decodes to digit(s) {missing} for this tokenizer "
            f"-- inspect this model's tokenizer manually."
        )

    if verbose:
        print(f"[resolve_digit_token_aliases] resolved aliases: {aliases}")
    return aliases


def score_batch(model, tok, proc, spec, texts, digit_token_aliases, device):
    """
    Single forward pass over a batch of already chat-templated prompts
    (generation prompt included, no assistant text). Returns:
      - restricted_probs: (batch, 4) softmax-like distribution over the 4
        digits, where each digit's mass is the sum of the full-vocab
        probabilities of ALL of its alias token ids, renormalized to sum
        to 1 across digits.
      - mass: (batch,) total probability the FULL vocab distribution puts
        on all digit-alias tokens combined -- a diagnostic for whether the
        judge is confidently heading straight for a bare digit at all.
    """
    inputs = tokenize_for_scoring(tok, proc, spec, texts, device)

    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]  # next-token logits after the prompt

    full_probs = torch.softmax(logits, dim=-1)
    digit_probs = torch.zeros(full_probs.shape[0], len(DIGITS), device=full_probs.device)
    for k, d in enumerate(DIGITS):
        ids = digit_token_aliases[d]
        digit_probs[:, k] = full_probs[:, ids].sum(dim=-1)

    mass = digit_probs.sum(dim=-1)
    restricted_probs = digit_probs / digit_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return restricted_probs.cpu(), mass.cpu()


def run_test_mode(args, model, tok, proc, spec, pairs_df, device):
    entries = build_direction_prompts(pairs_df.head(args.n_test), args.scoring_prompt_version)
    prompts = [e[3] for e in entries]

    digit_token_aliases = resolve_digit_token_aliases(tok)
    # Reverse lookup: any alias token id -> the digit it represents. Multiple
    # ids can map to the same digit; that's fine, we only need id -> digit here.
    id_to_digit = {tid: d for d, ids in digit_token_aliases.items() for tid in ids}

    messages_list = [build_messages(p, spec, want_thinking=False) for p in prompts]
    texts = [apply_chat_template_safe(tok, m, spec, want_thinking=False) for m in messages_list]
    inputs = tokenize_for_scoring(tok, proc, spec, texts, device)

    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
        gen_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)

    full_probs = torch.softmax(logits, dim=-1)
    digit_probs = torch.zeros(len(prompts), len(DIGITS))
    for k, d in enumerate(DIGITS):
        ids = digit_token_aliases[d]
        digit_probs[:, k] = full_probs[:, ids].cpu().sum(dim=-1)
    mass = digit_probs.sum(dim=-1)
    restricted_probs = digit_probs / digit_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    input_len = inputs["input_ids"].shape[1]
    raw_new = gen_ids[:, input_len:]

    header = (
        f"{'dir':5} {'pair':10} {'raw (unstripped)':22} {'first_tok_id':13} "
        f"{'gen_digit':9} {'argmax':7} {'match':6} {'mass(1-4)':10} {'P1':6} {'P2':6} {'P3':6} {'P4':6}"
    )
    print(header)
    print("-" * len(header))

    n_mismatch = 0
    n_low_mass = 0
    n_unrecognized = 0
    for i, (direction, id1, id2, _) in enumerate(entries):
        raw_text = tok.decode(raw_new[i], skip_special_tokens=True)
        first_tok_id = raw_new[i, 0].item()
        # None here means generate()'s actual first token isn't any known
        # digit-alias id at all -- worth flagging separately from a mismatch
        # where it IS a digit alias, just not the argmax one.
        first_tok_digit = id_to_digit.get(first_tok_id)
        argmax_digit = DIGITS[int(restricted_probs[i].argmax())]
        match = first_tok_digit == argmax_digit
        n_mismatch += int(not match)
        n_unrecognized += int(first_tok_digit is None)
        n_low_mass += int(mass[i].item() < 0.9)
        probs_str = " ".join(f"{restricted_probs[i, j].item():.3f}" for j in range(4))
        pair_str = f"{id1}-{id2}"
        print(
            f"{direction:5} {pair_str:10} {repr(raw_text):22} {first_tok_id:13} "
            f"{str(first_tok_digit):>9} {argmax_digit:>7} {str(match):6} {mass[i].item():10.3f} {probs_str}"
        )

    n = len(entries)
    print("-" * len(header))
    print(
        f"{n_mismatch}/{n} mismatches between generate()'s actual first token's digit "
        f"and the restricted-distribution argmax (should be 0 if the digit is reliably token 0)."
    )
    print(
        f"{n_unrecognized}/{n} examples where generate()'s first token wasn't recognized as "
        f"ANY digit alias at all (should be 0 -- if nonzero, the judge isn't leading with a bare digit)."
    )
    print(
        f"{n_low_mass}/{n} examples with <90% of total probability mass on digit-alias "
        f"tokens (low mass = judge isn't confidently heading straight for a digit)."
    )
    print(f"Resolved digit token aliases: {digit_token_aliases}")


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

    if args.test:
        print(
            f"Judge: {spec.name} (key '{args.judge_model_id}') | scoring prompt: "
            f"{args.scoring_prompt_version} | generations from '{args.generation_model_id}' "
            f"/ '{args.generation_prompt_version}' | testing {args.n_test} pairs "
            f"({2 * args.n_test} directional prompts)"
        )
        run_test_mode(args, model, tok, proc, spec, pairs_df, device)
        return

    output_path = resolve_output_path(args)
    effective_batch_size = max(1, args.batch_size // spec.batch_size_divide)
    print(f"Judge: {spec.name} (key '{args.judge_model_id}') | effective batch size: {effective_batch_size} | device: {device}")

    entries = build_direction_prompts(pairs_df, args.scoring_prompt_version)
    prompts = [e[3] for e in entries]

    digit_token_aliases = resolve_digit_token_aliases(tok)

    messages_list = [build_messages(p, spec, want_thinking=False) for p in prompts]
    texts = [apply_chat_template_safe(tok, m, spec, want_thinking=False) for m in messages_list]

    n = len(texts)
    all_probs = torch.zeros((n, 4))
    all_mass = torch.zeros(n)
    indices = list(range(n))

    def save_progress():
        probs_np = all_probs.numpy()
        mass_np = all_mass.numpy()
        soft_score = probs_np @ np.array([1, 2, 3, 4])
        argmax_score = probs_np.argmax(axis=1) + 1
        entropy = -(probs_np * np.log(np.clip(probs_np, 1e-12, 1.0))).sum(axis=1)

        for k, d in enumerate(DIGITS):
            pairs_df[f"prob_{d}_1to2"] = probs_np[0::2, k]
            pairs_df[f"prob_{d}_2to1"] = probs_np[1::2, k]
        pairs_df["mass_on_digits_1to2"] = mass_np[0::2]
        pairs_df["mass_on_digits_2to1"] = mass_np[1::2]
        pairs_df["soft_score_1to2"] = soft_score[0::2]
        pairs_df["soft_score_2to1"] = soft_score[1::2]
        pairs_df["generated_score_1to2"] = argmax_score[0::2]
        pairs_df["generated_score_2to1"] = argmax_score[1::2]
        pairs_df["entropy_1to2"] = entropy[0::2]
        pairs_df["entropy_2to1"] = entropy[1::2]
        pairs_df["judge_model_used"] = spec.name
        pairs_df["generation_model_id"] = args.generation_model_id
        pairs_df["generation_prompt_version"] = args.generation_prompt_version
        pairs_df.to_csv(output_path, index=False)

    for batch_idx in tqdm(list(batch_iterable(indices, effective_batch_size)), desc=f"Scoring ({args.judge_model_id})"):
        batch_texts = [texts[i] for i in batch_idx]
        probs, mass = score_batch(model, tok, proc, spec, batch_texts, digit_token_aliases, device)
        for j, i in enumerate(batch_idx):
            all_probs[i] = probs[j]
            all_mass[i] = mass[j]
        # Incremental save in case of crash on long runs
        save_progress()

    save_progress()
    print(f"Saved {len(pairs_df)} pairs (with probability-based scores in both directions) to {output_path}")

    mean_mass = all_mass.mean().item()
    if mean_mass < 0.9:
        print(
            f"[warn] mean probability mass on the 4 digit-alias tokens across all directional "
            f"prompts was only {mean_mass:.3f}. This judge may not be reliably answering "
            f"with a bare digit as the first token -- re-run with --test to inspect."
        )

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")


if __name__ == "__main__":
    main()