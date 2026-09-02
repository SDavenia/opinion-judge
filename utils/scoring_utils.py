import os
import pandas as pd
import re
from utils.models_utils import REGISTRY
from pathlib import Path

GENERATION_PROMPTS = {
    "base": """You are a precise text generation engine for an NLP evaluation dataset.
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
Output only the opinion text.""",

    "human_simulator": """You are a person that is asked to write an opinion on a moral situation. You are given a situation, a moral principle, and a target relationship to the situation (Supports, Opposes, or Either). 
You are also given a core rationale that explains why the moral principle is relevant to the situation. 
Your task is to write an opinion that reflects the target relationship to the situation, using the moral principle and the core rationale.

Situation: {situation}
Moral Principle: {value}
Target Relationship to Situation: {valence}
Core Rationale: {explanation}
Task:
{stance_instruction}

Constraints:
1. The output must be exactly 2-3 sentences long.
2. Write in a natural, first-person argumentative tone (as if written by a human expressing a genuine opinion).
3. Do not explicitly mention the words "Valence", "Core Rationale", or quote the instructions. Integrate the Core Rationale seamlessly into the stance.
Output only the opinion text.""",

    "reflective_person": """Imagine you are sharing your honest thoughts with someone after reading about a situation.

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
Output only the opinion.""",

    "internal_monologue": """You have just read the following situation and are reacting to it in your own mind.

Situation: {situation}
Moral Principle: {value}
Target Relationship to Situation: {valence}
Core Rationale: {explanation}

Task:
{stance_instruction}

Write what you genuinely think. Your reasoning should emerge naturally from your values instead of sounding like a structured argument.

Constraints:
1. Exactly 2-3 sentences.
2. Write as an authentic internal reflection.
3. Don't mention the moral principle directly unless it would naturally come up.
4. Make the opinion feel emotionally believable rather than analytical.
Output only the opinion.""",
}

PROMPTS = {
    "base": """You are an impartial evaluator.
Assess how well the second opinion aligns with the first in the following situation.

Situation: {situation}
First opinion: {first_opinion}
Second opinion: {second_opinion}

Return only a score from 1 to 4.""",

    "base-nofluency": """You are an impartial evaluator
Assess how well the second opinion (Opinion B) aligns with the first (Opinion A) in the following situation.
Rate how aligned they are in terms of viewpoint, on a scale of [1-4].
Do not consider writing quality, length, or fluency, only the similarity of position and underlying reasoning.

Situation: {situation}
Opinion A: {first_opinion}
Opinion B: {second_opinion}

Return only a score from 1 to 4.""",

    "base-with-scale": """Rate how well Opinion B reflects Opinion A on a 1-4 scale:
4 = same overall position, grounded in essentially the same underlying value/reason
3 = same overall position, but grounded in a different underlying value/reason
2 = one opinion takes a clear position, the other is neutral or mixed
1 = the opinions take opposing positions

Situation: {situation}
Opinion A: {first_opinion}
Opinion B: {second_opinion}

Return only the number.""",

    "0_1": {
        "base": """You are an impartial evaluator.
Assess how well the second opinion aligns with the first in the following situation.
You have to assign a score in the range [0,1] following this criteria:
-1.0 — Full alignment: The second opinion has essentially the same overall stance as the first, including the same direction, degree of support/opposition, and relevant ambivalence or qualifications. Differences in values, reasons, emphasis, or wording are minor and do not materially change the position.
-[0.7, 0.9] — Strong alignment: The second opinion substantially shares the first opinion's stance. It may differ in the strength of its position, in how it balances competing considerations, or in its values and reasons, but the main position is clearly aligned.
-[0.4, 0.6] — Partial alignment: The opinions share some meaningful aspects of their stance but also differ in important ways. This may include agreeing on one side of an issue while disagreeing on another, having different degrees of ambivalence, or reaching similar overall conclusions while placing substantially different weight on competing considerations. 
-[0.1, 0.3] — Weak alignment: The opinions largely differ in their overall stance, including the way they resolve or balance competing considerations.
-0.0 — No alignment: The opinions have fundamentally incompatible stances. 

Situation: {situation}
First opinion: {first_opinion}
Second opinion: {second_opinion}

Return only the score.""",

        "lex_var": """You are a helpful and meticulous evaluator.
You have to assess how well the second opinion aligns with the first in a given situation.
You have to assign a score in the range [0,1] following the criteria:
-1.0 — The second opinion has essentially the same overall stance as the first, including the same direction, degree of support/opposition, and relevant ambivalence or qualifications. Differences in values, reasons, emphasis, or wording are minor and do not materially change the position.
-[0.7, 0.9] — The second opinion substantially shares the first opinion's stance. It may differ in the strength of its position, in how it balances competing considerations, or in its values and reasons, but the main position is clearly aligned.
-[0.4, 0.6] — The opinions share some meaningful aspects of their stance but also differ in important ways. This may include agreeing on one side of an issue while disagreeing on another, having different degrees of ambivalence, or reaching similar overall conclusions while placing substantially different weight on competing considerations. 
-[0.1, 0.3] — The opinions largely differ in their overall stance, including the way they resolve or balance competing considerations.
-0.0 — The opinions have fundamentally incompatible stances. 

Situation: {situation}
First opinion: {first_opinion}
Second opinion: {second_opinion}

Answer only with the score.""",

        "reverse_order": """You are an impartial evaluator.
Assess how well the second opinion aligns with the first in the following situation.
You have to assign a score in the range [0,1] following this criteria:
-0.0 — No alignment: The opinions have fundamentally incompatible stances. 
-[0.1, 0.3] — Weak alignment: The opinions largely differ in their overall stance, including the way they resolve or balance competing considerations.
-[0.4, 0.6] — Partial alignment: The opinions share some meaningful aspects of their stance but also differ in important ways. This may include agreeing on one side of an issue while disagreeing on another, having different degrees of ambivalence, or reaching similar overall conclusions while placing substantially different weight on competing considerations. 
-[0.7, 0.9] — Strong alignment: The second opinion substantially shares the first opinion's stance. It may differ in the strength of its position, in how it balances competing considerations, or in its values and reasons, but the main position is clearly aligned.
-1.0 — Full alignment: The second opinion has essentially the same overall stance as the first, including the same direction, degree of support/opposition, and relevant ambivalence or qualifications. Differences in values, reasons, emphasis, or wording are minor and do not materially change the position.

Situation: {situation}
First opinion: {first_opinion}
Second opinion: {second_opinion}

Return only the score.""",

        "lex_var_reverse_order": """You are a helpful and meticulous evaluator.
You have to assess how well the second opinion aligns with the first in a given situation.
You have to assign a score in the range [0,1] following the criteria:
-0.0 — The opinions have fundamentally incompatible stances. 
-[0.1, 0.3] — The opinions largely differ in their overall stance, including the way they resolve or balance competing considerations.
-[0.4, 0.6] — The opinions share some meaningful aspects of their stance but also differ in important ways. This may include agreeing on one side of an issue while disagreeing on another, having different degrees of ambivalence, or reaching similar overall conclusions while placing substantially different weight on competing considerations. 
-[0.7, 0.9] — The second opinion substantially shares the first opinion's stance. It may differ in the strength of its position, in how it balances competing considerations, or in its values and reasons, but the main position is clearly aligned.
-1.0 — The second opinion has essentially the same overall stance as the first, including the same direction, degree of support/opposition, and relevant ambivalence or qualifications. Differences in values, reasons, emphasis, or wording are minor and do not materially change the position.

Situation: {situation}
First opinion: {first_opinion}
Second opinion: {second_opinion}

Answer only with the score.""",

        "criteria_after": """You are an impartial evaluator.
Assess how well the second opinion aligns with the first in the following situation.
Situation: {situation}
First opinion: {first_opinion}
Second opinion: {second_opinion}

You have to assign a score in the range [0,1] following this criteria:
-1.0 — Full alignment: The second opinion has essentially the same overall stance as the first, including the same direction, degree of support/opposition, and relevant ambivalence or qualifications. Differences in values, reasons, emphasis, or wording are minor and do not materially change the position.
-[0.7, 0.9] — Strong alignment: The second opinion substantially shares the first opinion's stance. It may differ in the strength of its position, in how it balances competing considerations, or in its values and reasons, but the main position is clearly aligned.
-[0.4, 0.6] — Partial alignment: The opinions share some meaningful aspects of their stance but also differ in important ways. This may include agreeing on one side of an issue while disagreeing on another, having different degrees of ambivalence, or reaching similar overall conclusions while placing substantially different weight on competing considerations. 
-[0.1, 0.3] — Weak alignment: The opinions largely differ in their overall stance, including the way they resolve or balance competing considerations.
-0.0 — No alignment: The opinions have fundamentally incompatible stances.

Return only the score."""
    }
}


DIGITS = ["1", "2", "3", "4"]


def add_common_args(parser):
    """Args shared by both scoring scripts. Each script adds its own on top
    (e.g. --max_new_tokens, --test)."""
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
        "--scoring_prompt_version", type=str, default="base", choices=list(PROMPTS.keys()),
        help="Version of the judge prompt to use.",
    )
    parser.add_argument("--generation_csv_path", type=str, default=None, help="Optional path to a CSV of generated opinions to score (if not using the default path).")
    parser.add_argument("--batch_size", type=int, default=32, help="Global batch size before per-model scaling")
    parser.add_argument("--output_dir", type=str, default="output_scores", help="Where to save results")
    parser.add_argument("--limit", action="store_true", default=None, help="Optional row limit for debugging")
    parser.add_argument("--final_run", action="store_true", help="If set, writes to the final (non-limit) output path")
    parser.add_argument("--num_examples", type=int, default=None, help="Optional limit on number of examples to score (for debugging)")
    parser.add_argument("--option_setting", type=str, default="four", 
                        choices=["four", "0_1"], help="The setting for the options used in the scoring part.")

    parser.add_argument("--extract_ids_from", type=Path,
                        default=None,
                        help="Path of .csv file from where to extract the ids of the pairs to score.")
    return parser


def resolve_output_path(args,
                        script_name: str,
                        limit: bool):
    """
    # For final run, the generator id is not included (since we will have determined a best generator), while for all others it is.
    """

    limit_str = "_limit" if limit else ""
    if args.final_run:
        path = f"{args.output_dir}/{script_name}/{args.judge_model_id}_{args.scoring_prompt_version}{limit_str}.csv"
    else:
        path = f"{args.output_dir}/{script_name}/{args.generation_model_id}_{args.generation_prompt_version}_{args.judge_model_id}_{args.scoring_prompt_version}{limit_str}.csv"
        
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def get_prompt_variations(prompt_version):
    """
    Returns {variation_name: prompt_template} for the given scoring_prompt_version.
    If PROMPTS[prompt_version] is a plain template string (not a dict of variants),
    wraps it as {prompt_version: template} so downstream code can treat both
    cases uniformly.
    """
    entry = PROMPTS[prompt_version]
    if isinstance(entry, dict):
        return entry
    return {prompt_version: entry}

def prepare_evaluator_prompt(situation, first_opinion, second_opinion, prompt_template):
    """
    Formats the given prompt template for the judge model.
    (Now takes the template text directly, not a lookup key -- see
    get_prompt_variations for resolving a --scoring_prompt_version into templates.)
    """
    return prompt_template.format(situation=situation, first_opinion=first_opinion, second_opinion=second_opinion)

def assign_expected_score(row):
    if row["valence_1"] == "Either" or row["valence_2"] == "Either":
        if row["valence_1"] == "Either" and row["valence_2"] == "Either":
            return 4 if row["text_1"] == row["text_2"] else 3
        else:
            return 2
    elif row["valence_1"] == row["valence_2"]:
        return 4 if row["text_1"] == row["text_2"] else 3
    else:
        return 1


def load_generation_df(args):
    """
    # If final run, the generator model name & prompt are not included.
    """
    generation_model_name = REGISTRY[args.generation_model_id].name

    if args.limit:
        path = f"data/valueprism_generation_{args.generation_model_id}_{args.generation_prompt_version}_limit.csv"
        df = pd.read_csv(path)
        print(f"Loaded {path}: {len(df)} rows (debug/limit mode, assumed pre-filtered to this model+prompt).")
        return df
    else:
        if args.generation_csv_path:
            path = args.generation_csv_path
        else:
            path = f"data/valueprism_generation_{args.generation_model_id}_{args.generation_prompt_version}.csv"

    df = pd.read_csv(path)
    before = len(df)

    if "generation_prompt_version" in df.columns:
        df = df[df["generation_prompt_version"] == args.generation_prompt_version]
    else:
        print(f"[warn] '{path}' has no 'generation_prompt_version' column -- cannot "
              f"filter by --generation_prompt_version; scoring whatever's there.")

    if "model_used" in df.columns:
        df = df[df["model_used"] == generation_model_name]
    else:
        print(f"[warn] '{path}' has no 'model_used' column -- cannot filter by "
              f"--generation_model_id; scoring whatever's there.")

    df = df.reset_index(drop=True)
    print(
        f"Loaded {path}: {before} total rows -> {len(df)} after filtering to "
        f"generation_model='{generation_model_name}', "
        f"generation_prompt_version='{args.generation_prompt_version}'."
    )

    if len(df) == 0:
        raise ValueError(
            f"No rows left after filtering {path} to "
            f"generation_model='{generation_model_name}', "
            f"generation_prompt_version='{args.generation_prompt_version}'. Check that "
            f"these values actually appear in the file's 'model_used' / "
            f"'generation_prompt_version' columns."
        )
    return df


def build_pairs(gen_df):
    df_copy = gen_df.copy().reset_index().rename(columns={"index": "id"})
    pairs_df = (
        df_copy.merge(df_copy, on="situation", suffixes=("_1", "_2"))
        .query("id_1 < id_2")
        .reset_index(drop=True)
    )
    pairs_df["expected_score"] = pairs_df.apply(assign_expected_score, axis=1)
    return pairs_df


def build_direction_prompts(pairs_df, prompt_version):
    """
    Returns a flat list of (variation, direction, id_1, id_2, prompt).

    For each row in pairs_df, iterates over every prompt variation returned by
    get_prompt_variations(prompt_version) and both directions, interleaved as:
    [row0_var0_1to2, row0_var0_2to1, row0_var1_1to2, row0_var1_2to1, ..., row1_var0_1to2, ...]

    i.e. a block of `2 * num_variations` consecutive entries corresponds to one
    row of pairs_df, and within that block, consecutive pairs are (1to2, 2to1)
    for a given variation, in the same order as get_prompt_variations(...).keys().
    This ordering must match expand_pairs_for_variations below.
    """
    variations = get_prompt_variations(prompt_version)
    entries = []
    for _, row in pairs_df.iterrows():
        for var_name, template in variations.items():
            entries.append((
                var_name, "1to2", row["id_1"], row["id_2"],
                prepare_evaluator_prompt(row["situation"], row["generated_opinion_1"], row["generated_opinion_2"], template),
            ))
            entries.append((
                var_name, "2to1", row["id_1"], row["id_2"],
                prepare_evaluator_prompt(row["situation"], row["generated_opinion_2"], row["generated_opinion_1"], template),
            ))
    return entries


def expand_pairs_for_variations(pairs_df, prompt_version):
    """
    Repeats each row of pairs_df once per prompt variation in
    get_prompt_variations(prompt_version), tagging each copy with a
    'prompt_variation' column. Row order is [row0_var0, row0_var1, ...,
    row1_var0, row1_var1, ...] to match build_direction_prompts's grouping,
    so expanded row i lines up with the (2*i, 2*i+1) slice of entries/generations.
    """
    variations = list(get_prompt_variations(prompt_version).keys())
    n = len(variations)
    expanded = pairs_df.loc[pairs_df.index.repeat(n)].reset_index(drop=True)
    expanded["prompt_variation"] = variations * len(pairs_df)
    return expanded


def select_calibration_prompts(pairs_df, prompt_version, n_per_bucket=2, seed=0):
    """
    Build a small calibration set of directional prompts, stratified by
    `expected_score`, for use with resolve_digit_token_ids. Plain
    first-N-rows sampling tends to draw from just one or two situations
    (pairs_df is built via a per-situation merge, so same-situation rows are
    contiguous) and often just one expected_score bucket -- this instead
    pulls up to `n_per_bucket` pairs from each of the 4 expected_score
    buckets, so calibration has a real chance of seeing all 4 digits.

    NOTE: this does NOT guarantee all 4 digits appear in the judge's actual
    outputs -- the judge may not track expected_score at all, which is the
    thing you're studying. Any digit that still doesn't show up falls back
    to the unverified tokenizer.encode() guess inside
    resolve_digit_token_ids; watch for [warn] lines from that function.
    """
    sampled = []
    for score, group in pairs_df.groupby("expected_score"):
        n = min(n_per_bucket, len(group))
        if n < n_per_bucket:
            print(
                f"[select_calibration_prompts] only {len(group)} pairs available for "
                f"expected_score={score}, using all of them (wanted {n_per_bucket})."
            )
        sampled.append(group.sample(n=n, random_state=seed))

    missing_buckets = sorted({1, 2, 3, 4} - set(pairs_df["expected_score"].unique()))
    if missing_buckets:
        print(
            f"[select_calibration_prompts] expected_score bucket(s) {missing_buckets} "
            f"don't appear anywhere in pairs_df -- can't stratify on them."
        )

    calib_df = pd.concat(sampled, ignore_index=True) if sampled else pairs_df.head(0)
    entries = build_direction_prompts(calib_df, prompt_version)
    prompts = [e[3] for e in entries]
    print(
        f"[select_calibration_prompts] built {len(prompts)} calibration prompts from "
        f"{len(calib_df)} pairs across expected_score buckets "
        f"{sorted(calib_df['expected_score'].unique())}."
    )
    return prompts


def tokenize_for_scoring(tok, proc, spec, texts, device):
    """
    Left pad for generation + use processor if VLM.
    """
    prev_padding_side = tok.padding_side
    tok.padding_side = "left"
    if spec.is_vlm:
        inputs = proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    else:
        inputs = tok(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    tok.padding_side = prev_padding_side
    return inputs


def parse_generation_scoring(generation: str,
                              option_setting: str = None,
                              decimals: int = 1) -> str | float | None:

    if option_setting is None:
        option_setting = "four"

    if option_setting == "four":
        options = ["1", "2", "3", "4"]

        op_found = []
        for option in options:
            if option in generation:
                op_found.append(option)

        if len(op_found) == 1:
            return op_found[0]
        else:
            # both the case of no option found or multiple
            return None

    elif option_setting == "0_1":
        # Matches: "0.75", ".75", "1.0", "0", "1" — but not the "1" inside "10"
        pattern = r"(?<!\d)(?:0?\.\d+|1\.0+|0|1)(?!\d)"
        matches = re.findall(pattern, generation)

        candidates = []
        for m in matches:
            try:
                val = float(m)
            except ValueError:
                continue
            if 0.0 <= val <= 1.0:
                candidates.append(round(val, decimals))

        if len(candidates) == 0:
            return None

        distinct = set(candidates)
        if len(distinct) == 1:
            return candidates[0]
        else:
            # multiple different candidate scores found -> ambiguous
            return None

    else:
        raise ValueError(f"Unknown option_setting: {option_setting}")


def add_parsed_generation_scoring(df: pd.DataFrame,
                                   option_setting: str = None,
                                   decimals: int = 1) -> pd.DataFrame:
    """
    Adds parsed_score_1to2 and parsed_score_2to1 columns to the dataframe,
    based on the generated scores.
    """

    if option_setting is None:
        option_setting = "four"

    df["parsed_score_1to2"] = df["generated_score_1to2"].apply(
        parse_generation_scoring, option_setting=option_setting, decimals=decimals
    )
    df["parsed_score_2to1"] = df["generated_score_2to1"].apply(
        parse_generation_scoring, option_setting=option_setting, decimals=decimals
    )
    return df
