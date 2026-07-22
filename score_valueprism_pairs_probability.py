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
    select_calibration_prompts,
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
    parser.add_argument(
        "--calib_per_bucket", type=int, default=2,
        help="Pairs to sample per expected_score bucket (1-4) when resolving digit token ids.",
    )
    parser.add_argument("--calib_seed", type=int, default=0, help="Random seed for calibration-pair sampling.")
    return parser.parse_args()


def resolve_digit_token_ids(model, tok, proc, spec, calib_prompts, device, verbose=True):
    """
    Empirically determine which token id corresponds to each digit '1'-'4'
    as the *first* generated token, for this judge model/tokenizer/chat
    template -- by greedily generating one token on each of `calib_prompts`
    and reading off what actually comes out (handles leading-space /
    BPE-boundary token variants automatically). Falls back to a direct
    tokenizer.encode(digit) guess -- flagged as unverified -- for any digit
    that doesn't happen to show up in the calibration sample. Caller decides
    how calib_prompts was chosen (see select_calibration_prompts).
    """
    messages_list = [build_messages(p, spec, want_thinking=False) for p in calib_prompts]
    texts = [apply_chat_template_safe(tok, m, spec, want_thinking=False) for m in messages_list]
    inputs = tokenize_for_scoring(tok, proc, spec, texts, device)

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=1, do_sample=False, pad_token_id=tok.pad_token_id)

    first_tok_ids = gen_ids[:, inputs["input_ids"].shape[1]]
    mapping = {}
    for tid in first_tok_ids.tolist():
        char = tok.decode([tid]).strip()
        if char in DIGITS and char not in mapping:
            mapping[char] = tid

    for d in DIGITS:
        if d not in mapping:
            candidate_ids = tok.encode(d, add_special_tokens=False)
            if len(candidate_ids) == 1:
                mapping[d] = candidate_ids[0]
                if verbose:
                    print(
                        f"[resolve_digit_token_ids] digit '{d}' not seen in the "
                        f"prompt calibration sample; falling back to "
                        f"tokenizer.encode('{d}') -> id {candidate_ids[0]} "
                        f"(UNVERIFIED -- double check with --test)."
                    )
            else:
                raise ValueError(
                    f"Could not resolve a single token id for digit '{d}' "
                    f"(tokenizer.encode gave {candidate_ids}); inspect this "
                    f"model's tokenizer manually."
                )
    if verbose:
        print(f"[resolve_digit_token_ids] resolved: {mapping}")
    return mapping


def score_batch(model, tok, proc, spec, texts, digit_token_ids, device):
    """
    Single forward pass over a batch of already chat-templated prompts
    (generation prompt included, no assistant text). Returns:
      - restricted_probs: (batch, 4) softmax over just the 4 digit tokens
      - mass: (batch,) total probability the FULL vocab distribution puts
        on those 4 tokens combined -- a diagnostic for whether the judge is
        confidently heading straight for a bare digit at all.
    """
    inputs = tokenize_for_scoring(tok, proc, spec, texts, device)

    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]  # next-token logits after the prompt

    full_probs = torch.softmax(logits, dim=-1)
    digit_id_list = [digit_token_ids[d] for d in DIGITS]
    mass = full_probs[:, digit_id_list].sum(dim=-1)
    restricted_probs = torch.softmax(logits[:, digit_id_list], dim=-1)
    return restricted_probs.cpu(), mass.cpu()


def run_test_mode(args, model, tok, proc, spec, pairs_df, device):
    entries = build_direction_prompts(pairs_df.head(args.n_test), args.scoring_prompt_version)
    prompts = [e[3] for e in entries]

    # Calibration is drawn from the *full* pairs_df (stratified by expected_score)
    calib_prompts = select_calibration_prompts(
        pairs_df, args.scoring_prompt_version, n_per_bucket=args.calib_per_bucket, seed=args.calib_seed
    )
    digit_token_ids = resolve_digit_token_ids(model, tok, proc, spec, calib_prompts, device)
    digit_id_list = [digit_token_ids[d] for d in DIGITS]
    id_to_digit = {v: k for k, v in digit_token_ids.items()}

    messages_list = [build_messages(p, spec, want_thinking=False) for p in prompts]
    texts = [apply_chat_template_safe(tok, m, spec, want_thinking=False) for m in messages_list]
    inputs = tokenize_for_scoring(tok, proc, spec, texts, device)

    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
        gen_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)

    full_probs = torch.softmax(logits, dim=-1)
    restricted_probs = torch.softmax(logits[:, digit_id_list], dim=-1)
    mass = full_probs[:, digit_id_list].sum(dim=-1)

    input_len = inputs["input_ids"].shape[1]
    raw_new = gen_ids[:, input_len:]

    header = f"{'dir':5} {'pair':10} {'raw (unstripped)':22} {'first_tok_id':13} {'logit_argmax':13} {'match':6} {'mass(1-4)':10} {'P1':6} {'P2':6} {'P3':6} {'P4':6}"
    print(header)
    print("-" * len(header))

    n_mismatch = 0
    n_low_mass = 0
    for i, (direction, id1, id2, _) in enumerate(entries):
        raw_text = tok.decode(raw_new[i], skip_special_tokens=True)
        first_tok_id = raw_new[i, 0].item()
        argmax_digit_id = digit_id_list[int(restricted_probs[i].argmax())]
        argmax_digit = id_to_digit[argmax_digit_id]
        match = first_tok_id == argmax_digit_id
        n_mismatch += int(not match)
        n_low_mass += int(mass[i].item() < 0.9)
        probs_str = " ".join(f"{restricted_probs[i, j].item():.3f}" for j in range(4))
        pair_str = f"{id1}-{id2}"
        print(
            f"{direction:5} {pair_str:10} {repr(raw_text):22} {first_tok_id:13} "
            f"{argmax_digit:>13} {str(match):6} {mass[i].item():10.3f} {probs_str}"
        )

    n = len(entries)
    print("-" * len(header))
    print(
        f"{n_mismatch}/{n} mismatches between generate()'s actual first token and the "
        f"restricted-softmax argmax (should be 0 if the digit is reliably token 0)."
    )
    print(
        f"{n_low_mass}/{n} examples with <90% of total probability mass on the 4 digit "
        f"tokens (low mass = judge isn't confidently heading straight for a digit)."
    )
    print(f"Resolved digit token ids: {digit_token_ids}")


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

    calib_prompts = select_calibration_prompts(
        pairs_df, args.scoring_prompt_version, n_per_bucket=args.calib_per_bucket, seed=args.calib_seed
    )
    digit_token_ids = resolve_digit_token_ids(model, tok, proc, spec, calib_prompts, device)

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
        probs, mass = score_batch(model, tok, proc, spec, batch_texts, digit_token_ids, device)
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
            f"[warn] mean probability mass on the 4 digit tokens across all directional "
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