"""
Score a generations CSV (as produced by the opinion-generation pipeline)
with meta-llama/Llama-Guard-3-8B in three settings:

  1. situation-only   : the situation text alone, in the `user` turn.
                         Deduplicated by situation_id since many rows
                         (different stance/value combos) share a situation.
  2. opinion-only      : the generated opinion (`text` column) alone,
                         in the `user` turn.
  3. prompt+opinion    : the full generation prompt (reconstructed with the
                         same template used at generation time) in the
                         `user` turn, and the generated opinion in the
                         `assistant` turn.

For each setting we extract a continuous 0-1 "unsafe_score" (softmax prob
mass on the "unsafe" token vs "safe" token at the first generated
position), plus the greedy label and raw output, the same way as the
single-text scorer.

Usage:
    python extract_generations_llamaguard.py \
        --dataset_id valueprism \
        --generation_model_id gemma3 --generation_prompt_version base \
        --output_dir safety_scores/

    python extract_generations_llamaguard.py \
        --dataset_id habermas \
        --output_dir safety_scores/

--dataset_id selects the input generations CSV (see resolve_path_generations)
and is appended as a subdirectory to --output_dir (matching score.py /
extract_model_embedded_opinion.py / extract_model_generation_perplexity.py's
output layout), unless --path_generations is given explicitly.

Requires: transformers, torch, accelerate, pandas, tqdm
    pip install transformers torch accelerate pandas tqdm --break-system-packages
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-Guard-3-8B"

# Same template used in the generation pipeline (prepare_generation_prompt / GENERATION_PROMPTS["base"])
GENERATION_PROMPT_TEMPLATE = """Imagine you are sharing your honest thoughts with someone after reading about a situation.

Situation: {situation}
Moral Principle: {value}
Target Relationship to Situation: {valence}
Core Rationale: {explanation}

Task:
{stance_instruction}

Write a short opinion that genuinely sounds like something a person would say. Let the moral principle shape your reasoning naturally rather than naming it explicitly. The opinion should feel thoughtful and personal, not like an explanation for a dataset.

Constraints:
1. Exactly 2-3 sentences.
2. Use a natural first-person argumentative voice.
3. Integrate the rationale implicitly.
4. Avoid sounding formal, academic, or overly balanced unless the required Target Relationship is "Either".
Output only the opinion."""


def stance_instruction_for(value: str, valence: str) -> str:
    """Mirrors prepare_generation_prompt's stance_instruction logic exactly."""
    if valence == "Supports":
        return f"Write an opinion that firmly ADVOCATES FOR or JUSTIFIES the action in the situation, using the principle of {value}."
    elif valence == "Opposes":
        return f"Write an opinion that firmly CRITICIZES or ARGUES AGAINST the action in the situation, using the principle of {value}."
    else:  # "Either"
        return f"Write an opinion that highlights the AMBIGUITY, CONDITIONALITY, or internal tension of the situation regarding the principle of {value}."


def reconstruct_generation_prompt(row: pd.Series) -> str:
    situation = row["situation"]
    value = row["vp_vrd_value"]
    valence = row["stance"]
    explanation = row["vp_explanation"]
    stance_instruction = stance_instruction_for(value, valence)
    return GENERATION_PROMPT_TEMPLATE.format(
        situation=situation,
        value=value,
        valence=valence,
        explanation=explanation,
        stance_instruction=stance_instruction,
    )



class LlamaGuardScorer:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None, dtype=torch.bfloat16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

        # IMPORTANT: standalone tokenizer.encode("safe") is NOT guaranteed to
        # match the token id the model actually produces right after the
        # chat template's assistant header. Teacher-forcing through the
        # template doesn't work either, since Llama Guard's template phrases
        # its trailing instructions differently depending on whether it's
        # assessing a User or Agent turn. Instead, look up the token id
        # directly: "safe"/"unsafe" always appear right after a newline (no
        # preceding space), so the no-leading-space BPE token id is fixed
        # and context-independent.
        self.safe_id, self.unsafe_id = self._detect_label_token_ids()

    def _detect_label_token_ids(self) -> tuple[int, int]:
        safe_id = self._static_token_id("safe")
        unsafe_id = self._static_token_id("unsafe")
        return safe_id, unsafe_id

    def _static_token_id(self, word: str) -> int:
        vocab = self.tokenizer.get_vocab()
        matches = [
            tok_id
            for tok_str, tok_id in vocab.items()
            if self.tokenizer.convert_tokens_to_string([tok_str]) == word
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one no-leading-space token matching {word!r}, "
                f"found {len(matches)}: {matches}. Inspect tokenizer vocab manually."
            )
        return matches[0]


    def _build_chat(self, text: str, role: str = "user"):
        # Llama Guard uses a chat template with a single conversation turn
        # to moderate. `role` can be "user" (moderate the input) or
        # "assistant" (moderate a model's reply).
        return [{"role": role, "content": text}]

    @torch.no_grad()
    def score(self, text: str, role: str = "user") -> dict:
        """
        Returns a dict with:
          - unsafe_score: float in [0, 1], softmax prob mass on "unsafe"
                           vs "safe", read from whichever generation step
                           the label token actually appears at
          - label: "safe" or "unsafe" (greedy decode)
          - raw_output: the model's full decoded generation (label + category)
        """
        chat = self._build_chat(text, role=role)
        input_ids = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        # Llama Guard always prepends some formatting (e.g. "\n\n") before
        # the actual "safe"/"unsafe" token, regardless of add_generation_prompt.
        # Walk the greedy generation step by step (via output_scores) and
        # find wherever the label token actually lands, then read the
        # restricted softmax from *that* step's logits.
        gen_out = self.model.generate(
            input_ids,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        new_tokens = gen_out.sequences[0, input_ids.shape[-1]:]

        label_step = None
        for i, tok_id in enumerate(new_tokens.tolist()):
            if tok_id in (self.safe_id, self.unsafe_id):
                label_step = i
                break

        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if label_step is None:
            raise RuntimeError(
                f"Could not find a 'safe'/'unsafe' token within the first "
                f"{len(new_tokens)} generated tokens. raw_output={raw_output!r}"
            )

        label_logits = gen_out.scores[label_step][0]
        two_logits = torch.stack([label_logits[self.safe_id], label_logits[self.unsafe_id]])
        probs = torch.softmax(two_logits.float(), dim=0)
        unsafe_score = probs[1].item()
        label = "unsafe" if new_tokens[label_step].item() == self.unsafe_id else "safe"

        return {
            "unsafe_score": unsafe_score,
            "label": label,
            "raw_output": raw_output,
        }

def resolve_path_generations(args) -> Path | None:
    """Returns None (rather than erroring) when valueprism defaults can't be resolved,
    so the caller can report it via parser.error for a consistent CLI error message."""
    if args.path_generations is not None:
        return args.path_generations
    if args.dataset_id == "habermas":
        # Habermas opinions are human-written and ship with 'text' already
        # filled in -- no separate generation step/file, so we just read the
        # prepared dataset CSV directly.
        return Path("data/habermas_sample.csv")
    if args.generation_model_id is None or args.generation_prompt_version is None:
        return None
    return args.generation_dir / f"{args.generation_model_id}_{args.generation_prompt_version}.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_id", type=str, required=True, choices=["habermas", "valueprism"],
        help="Which dataset's generations to score. Selects the default --path_generations "
             "and is appended as a subdirectory to --output_dir.",
    )
    parser.add_argument(
        "--path_generations", type=Path, default=None,
        help="Path to the generations CSV. Defaults to data/habermas_sample.csv for habermas, "
             "or {generation_dir}/{generation_model_id}_{generation_prompt_version}.csv for valueprism.",
    )
    parser.add_argument("--generation_model_id", type=str, default=None,
                         help="Only used for --dataset_id=valueprism to resolve the default --path_generations.")
    parser.add_argument("--generation_prompt_version", type=str, default=None,
                         help="Only used for --dataset_id=valueprism to resolve the default --path_generations.")
    parser.add_argument("--generation_dir", type=Path, default=Path("generations/"))
    parser.add_argument("--output_dir", type=Path, default=Path("safety_scores/"), help="Where to save score CSVs")
    parser.add_argument("--num_examples", type=int, default=None, help="Optional: limit to first N examples for testing")
    args = parser.parse_args()
    args.path_generations = resolve_path_generations(args)
    if args.path_generations is None:
        parser.error(
            "--generation_model_id and --generation_prompt_version are required when "
            "--dataset_id=valueprism and --path_generations is not given."
        )
    args.output_dir = args.output_dir / args.dataset_id
    return args


def main():
    args = parse_args()
    df = pd.read_csv(args.path_generations)
    if args.num_examples is not None:
        df = df.head(args.num_examples)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scorer = LlamaGuardScorer()

    # --- Setting 1: situation alone (deduped) ---
    situations_df = df.drop_duplicates(subset="situation_id")[["situation_id", "situation"]].reset_index(drop=True)
    situation_results = []
    for _, row in tqdm(situations_df.iterrows(), total=len(situations_df), desc="Scoring situations"):
        #res = scorer.score_user_only(row["situation"])
        res = scorer.score(row["situation"])
        situation_results.append({"situation_id": row["situation_id"], **res})
    situation_scores_df = pd.DataFrame(situation_results)
    situation_scores_df.to_csv(args.output_dir / "situation_scores.csv", index=False)
    print(f"Saved {len(situation_scores_df)} situation scores to {args.output_dir / 'situation_scores.csv'}")

    # --- Setting 2: generated opinion alone ---
    opinion_results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring opinions"):
        res = scorer.score(row["text"])
        opinion_results.append({"text_id": row["text_id"], **res})
    opinion_scores_df = pd.DataFrame(opinion_results)
    opinion_scores_df.to_csv(args.output_dir / "opinion_scores.csv", index=False)
    print(f"Saved {len(opinion_scores_df)} opinion scores to {args.output_dir / 'opinion_scores.csv'}")

    # #--- Setting 3: full generation prompt (user) + generated opinion (assistant) ---
    # full_results = []
    # for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring prompt+opinion"):
    #     prompt = reconstruct_generation_prompt(row)
    #     res = scorer.score(prompt, row["text"])
    #     full_results.append({"text_id": row["text_id"],
    #                          "text": row["text"],
    #                            **res})
    # full_scores_df = pd.DataFrame(full_results)
    # full_scores_df.to_csv(args.output_dir / "prompt_opinion_scores.csv", index=False)
    # print(f"Saved {len(full_scores_df)} prompt+opinion scores to {args.output_dir / 'prompt_opinion_scores.csv'}")


if __name__ == "__main__":
    main()