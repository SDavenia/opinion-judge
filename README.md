# opinion-judge

Pipeline for generating opinions on situations, and for using LLM judges to score/evaluate those opinions (as pairwise comparisons, embedded-opinion probes, and generation perplexity).

## Setup

```bash
pip install -r requirements.txt
```

Model definitions (HF checkpoint id, chat-template quirks, batching) live in [utils/models_utils.py](utils/models_utils.py) under `REGISTRY`; the `--*_model_id` / `--judge_model_id` / `--generation_model_id` CLI flags below always refer to keys of that registry (e.g. `gemma3`, `llama-3.3-70b`, `qwen2.5-72b`, `mistral-24b`, `llama-3.1-8b`). Prompt templates for each stage live in [utils/prompts_utils.py](utils/prompts_utils.py) (`GENERATION_PROMPTS`, `SCORING_PROMPTS`, `PERPLEXITY_PROMPTS`, `ALIGNMENT_PROMPTS`), and the `--*_prompt_version` flags select a key from the matching dict.

## Pipeline order

1. [prepare_data.py](prepare_data.py)
2. [prepare_valueprism_generations.py](prepare_valueprism_generations.py) (ValuePrism only — the other datasets already ship free-form user opinions)
3. [score_valueprism_pairs.py](score_valueprism_pairs.py) (currently ValuePrism-specific; to be generalized to all datasets)
4. [extract_model_embedded_opinion.py](extract_model_embedded_opinion.py) and [extract_model_generation_perplexity.py](extract_model_generation_perplexity.py) — two independent ways to ground a judge model's own biased position on a situation (not part of the main scoring pipeline)

[valueprism_clustering_all.py](valueprism_clustering_all.py) exists but is currently unused/dead code — ignored here.

---

### [prepare_data.py](prepare_data.py)

Downloads, filters, and balances the raw opinion/value datasets (Habermas Machine, ValuePrism; Humanual Opinion not yet implemented) into a common schema.

- **Input:** none (downloads Habermas Machine parquet files and the ValuePrism CSV directly from their sources).
- **Output:** written to `--output_dir` (default `data/`):
  - `habermas_sample.csv`
  - `valueprism_sample.csv`
  - Common columns: `situation_id, text_id, situation, stance, dataset, ...` (extra dataset-specific columns, e.g. `vp_vrd`, `vp_vrd_value`, `vp_explanation` for ValuePrism).

### [prepare_valueprism_generations.py](prepare_valueprism_generations.py)

Prompts a model to generate a free-form opinion for each row of a prepared ValuePrism CSV (advocating for / against / highlighting ambiguity, depending on the row's stance).

- **Input:** `--path_dataset` — a CSV produced by `prepare_data.py` (e.g. `data/valueprism_sample.csv`), must have `situation`, `vp_vrd_value`, `stance`, `vp_explanation` columns.
- **Output:** `{output_dir}/{generation_model_id}_{generation_prompt_version}.csv` (default `output_dir` is `generations/`), i.e. e.g. `generations/gemma3_base.csv`. Same rows as the input, plus `text` (the generated opinion) and `generation_model`.

> Note: [score_valueprism_pairs.py](score_valueprism_pairs.py)'s default generation-CSV lookup path (`data/valueprism_generation_{generation_model_id}_{generation_prompt_version}.csv`, see `load_generation_df` in [utils/scoring_utils.py](utils/scoring_utils.py)) does not match this script's default output path/name. Either pass `--generation_csv_path` explicitly to `score_valueprism_pairs.py`, or place/rename the generations file accordingly.

### [score_valueprism_pairs.py](score_valueprism_pairs.py)

Builds all same-situation opinion pairs from a generations CSV and has a judge model score each pair in both directions (1→2 and 2→1), for every prompt variation of the chosen scoring prompt.

- **Input:** a generations CSV (columns include `situation`, `text`, `generation_model`, `generation_prompt_version`), resolved as:
  - `--generation_csv_path` if given, otherwise
  - `data/valueprism_generation_{generation_model_id}_{generation_prompt_version}.csv` (see note above).
  - Optionally `--extract_ids_from` — a CSV with `id_1, id_2` columns to restrict which pairs get scored.
- **Output:** `{output_dir}/{generation_model_id}_{generation_prompt_version}_{judge_model_id}_{scoring_prompt_version}.csv` (default `output_dir` is `output_scores/`). One row per (pair, prompt variation), with `generated_score_1to2`, `generated_score_2to1` (raw judge text) and `parsed_score_1to2`, `parsed_score_2to1` (parsed per `--option_setting`), plus `judge_model`, `generation_model`, `generation_prompt_version`.

### [extract_model_embedded_opinion.py](extract_model_embedded_opinion.py)

For each unique situation in a dataset, repeatedly prompts a model to state its own stance (e.g. acceptable/unacceptable/ambiguous) and tallies the label distribution — grounding the model's own embedded opinion on each situation.

- **Input:** `--path_dataset` — any prepared dataset CSV with `situation_id`, `situation` columns.
- **Output:** under `{output_dir}/{judge_model_id}/` (default `output_dir` is `model_alignment/`):
  - `{alignment_prompt_version}_generations.csv` — one row per sample: `situation, generation, parsed_evaluations`.
  - `{alignment_prompt_version}.csv` — one row per unique situation: `situation_id, situation, alignment_distribution` (JSON string of `p(label)` per label + `total_evaluations`; parse with `json.loads`).

### [extract_model_generation_perplexity.py](extract_model_generation_perplexity.py)

For each generated opinion, computes its perplexity under a (judge) model conditioned on a neutral, stance-free prompt for the situation — an alternative way to ground the judge model's bias toward/against a given opinion.

- **Input:** `{generation_dir}/{generation_model_id}_{generation_prompt_version}.csv` (default `generation_dir` is `generations/` — matches [prepare_valueprism_generations.py](prepare_valueprism_generations.py)'s output naming). Must contain `situation_id, situation, text_id, text`.
- **Output:** `{output_dir}/{judge_model_id}/{generation_model_id}_{generation_prompt_version}/{perplexity_prompt}.csv` (default `output_dir` is `model_alignment_perplexity/`). Columns: `situation_id, situation, text_id, text, perplexity`.

## Shared code

- [utils/models_utils.py](utils/models_utils.py) — model registry (`REGISTRY`) and `load_model`.
- [utils/prompts_utils.py](utils/prompts_utils.py) — all prompt templates (`GENERATION_PROMPTS`, `SCORING_PROMPTS`, `PERPLEXITY_PROMPTS`, `ALIGNMENT_PROMPTS`) and chat-template helpers.
- [utils/generation_utils.py](utils/generation_utils.py) — batching helper (`batch_iterable`).
- [utils/scoring_utils.py](utils/scoring_utils.py) — pairwise-scoring helpers used by `score_valueprism_pairs.py`.
- [utils/evaluation_utils.py](utils/evaluation_utils.py), [utils/files_io.py](utils/files_io.py) — additional evaluation/IO helpers.
