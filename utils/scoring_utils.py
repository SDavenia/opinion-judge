import os
import pandas as pd
import re
from pathlib import Path

from utils.models_utils import REGISTRY
from utils.prompts_utils import SCORING_PROMPTS

def resolve_output_path(args,
                        ):
    """
    # For final run, the generator id is not included (since we will have determined a best generator), while for all others it is.
    """

    
    path = f"{args.output_dir}/{args.generation_model_id}_{args.generation_prompt_version}_{args.judge_model_id}_{args.scoring_prompt_version}.csv"
        
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def get_prompt_variations(prompt_version):
    """
    Returns {variation_name: prompt_template} for the given scoring_prompt_version.
    If PROMPTS[prompt_version] is a plain template string (not a dict of variants),
    wraps it as {prompt_version: template} so downstream code can treat both
    cases uniformly.
    """
    entry = SCORING_PROMPTS[prompt_version]
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



def load_generation_df(args):
    """
    # If final run, the generator model name & prompt are not included.
    """

    
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

    if "generation_model" in df.columns:
        df = df[df["generation_model"] == args.generation_model_id]
    else:
        print(f"[warn] '{path}' has no 'generation_model' column -- cannot filter by "
              f"--generation_model_id; scoring whatever's there.")

    df = df.reset_index(drop=True)
    print(
        f"Loaded {path}: {before} total rows -> {len(df)} after filtering to "
        f"generation_model='{args.generation_model_id}', "
        f"generation_prompt_version='{args.generation_prompt_version}'."
    )

    if len(df) == 0:
        raise ValueError(
            f"No rows left after filtering {path} to "
            f"generation_model='{args.generation_model_id}', "
            f"generation_prompt_version='{args.generation_prompt_version}'. Check that "
            f"these values actually appear in the file's 'generation_model' / "
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
                prepare_evaluator_prompt(row["situation"], row["text_1"], row["text_2"], template),
            ))
            entries.append((
                var_name, "2to1", row["id_1"], row["id_2"],
                prepare_evaluator_prompt(row["situation"], row["text_2"], row["text_1"], template),
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

