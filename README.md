# opinion-judge

Pipeline for generating opinions on situations, and for using LLM judges to score/evaluate those opinions (as pairwise comparisons, embedded-opinion probes, and generation perplexity).

## Setup

```bash
pip install -r requirements.txt
```

Model definitions (HF checkpoint id, chat-template quirks, batching) live in [utils/models_utils.py](utils/models_utils.py) under `REGISTRY`; the `--*_model_id` / `--judge_model_id` / `--generation_model_id` CLI flags below always refer to keys of that registry (e.g. `gemma3`, `llama-3.3-70b`, `qwen2.5-72b`, `mistral-24b`, `llama-3.1-8b`). Prompt templates for each stage live in [utils/prompts_utils.py](utils/prompts_utils.py) (`GENERATION_PROMPTS`, `SCORING_PROMPTS`, `PERPLEXITY_PROMPTS`, `ALIGNMENT_PROMPTS`), and the `--*_prompt_version` flags select a key from the matching dict.

Most scripts below take a `--dataset_id` flag (`habermas` or `valueprism`) that selects both the input dataset and an output subdirectory named after it (e.g. `output_scores/habermas/...` vs. `output_scores/valueprism/...`). Habermas opinions are real, human-written text that ships directly from `prepare_data.py`, so there's no generation step and no `--generation_model_id` / `--generation_prompt_version` for it; those flags (and the `generation_model` / `generation_prompt_version` output columns) only apply to `--dataset_id=valueprism`.

## Pipeline order

1. [prepare_data.py](prepare_data.py)
2. [prepare_valueprism_generations.py](prepare_valueprism_generations.py) (ValuePrism only — Habermas already ships free-form human opinions in its `text` column)
3. [score.py](score.py) (or [score_openrouter.py](score_openrouter.py) for API-based judges) — works for both datasets via `--dataset_id`
4. [extract_model_embedded_opinion.py](extract_model_embedded_opinion.py) and [extract_model_generation_perplexity.py](extract_model_generation_perplexity.py) — two independent ways to ground a judge model's own biased position on a situation (not part of the main scoring pipeline)
5. [analyse_results.py](analyse_results.py) — consumes the outputs of steps 3-4 to compute stability (RS/PC) vs. various characteristics

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

### [score.py](score.py)

Builds all same-situation opinion pairs from a generations CSV and has a judge model score each pair in both directions (1→2 and 2→1), for every prompt variation of the chosen scoring prompt. Takes `--dataset_id` (`habermas` or `valueprism`).

- **Input:** a generations CSV, resolved as:
  - `--generation_csv_path` if given, otherwise
  - for `--dataset_id=habermas`: `data/habermas_sample.csv` directly (already has `text`; no `--generation_model_id`/`--generation_prompt_version` needed).
  - for `--dataset_id=valueprism`: `generations/{generation_model_id}_{generation_prompt_version}.csv` (matches [prepare_valueprism_generations.py](prepare_valueprism_generations.py)'s output naming).
  - Optionally `--extract_ids_from` — a CSV with `id_1, id_2` columns to restrict which pairs get scored.
- **Output:** `{output_dir}/{dataset_id}/{judge_model_id}_{scoring_prompt_version}.csv` for habermas, or `{output_dir}/{dataset_id}/{generation_model_id}_{generation_prompt_version}_{judge_model_id}_{scoring_prompt_version}.csv` for valueprism (default `output_dir` is `output_scores/`). One row per (pair, prompt variation), with `generated_score_1to2`, `generated_score_2to1` (raw judge text) and `parsed_score_1to2`, `parsed_score_2to1` (parsed per `--option_setting`), plus `judge_model`, `generation_model` (`"human"` for habermas), `generation_prompt_version` (`None` for habermas).
- [score_openrouter.py](score_openrouter.py) is the API-based equivalent (same `--dataset_id`/input/output conventions, default `output_dir` is `scoring/`).

### [extract_model_embedded_opinion.py](extract_model_embedded_opinion.py)

For each unique situation in a dataset, repeatedly prompts a model to state its own stance (e.g. acceptable/unacceptable/ambiguous) and tallies the label distribution — grounding the model's own embedded opinion on each situation. Takes `--dataset_id` (`habermas` or `valueprism`).

- **Input:** `--path_dataset` — any prepared dataset CSV with `situation_id`, `situation` columns. Defaults to `data/{dataset_id}_sample.csv` if not given.
- `--alignment_prompt_version` must be a key valid for `--dataset_id`, per `ALIGNMENT_PROMPTS_BY_DATASET` in [utils/prompts_utils.py](utils/prompts_utils.py) — defaults to `impartial_evaluator` for valueprism (behaviour/dilemma framing: ACCEPTABLE/UNACCEPTABLE/AMBIGUOUS, would-you-carry-out-the-action, ...) and `hb_impartial_evaluator` for habermas (a parallel set of 10 `hb_*`-prefixed prompts reframed around agreeing/disagreeing with a stated policy position, since habermas `situation`s are already claims like "We should ban right turns in central London.", not narrated actions to morally judge).
- **Output:** under `{output_dir}/{dataset_id}/{judge_model_id}/` (default `output_dir` is `model_alignment/`):
  - `{alignment_prompt_version}_generations.csv` — one row per sample: `situation, generation, parsed_evaluations`.
  - `{alignment_prompt_version}.csv` — one row per unique situation: `situation_id, situation, alignment_distribution` (JSON string of `p(label)` per label + `total_evaluations`; parse with `json.loads`).

### [extract_model_generation_perplexity.py](extract_model_generation_perplexity.py)

For each generated opinion, computes its perplexity under a (judge) model conditioned on a neutral, stance-free prompt for the situation — an alternative way to ground the judge model's bias toward/against a given opinion. Takes `--dataset_id` (`habermas` or `valueprism`).

- **Input:**
  - for `--dataset_id=habermas`: `data/habermas_sample.csv` directly (no `--generation_model_id`/`--generation_prompt_version` needed).
  - for `--dataset_id=valueprism`: `{generation_dir}/{generation_model_id}_{generation_prompt_version}.csv` (default `generation_dir` is `generations/` — matches [prepare_valueprism_generations.py](prepare_valueprism_generations.py)'s output naming). Must contain `situation_id, situation, text_id, text`.
- `--perplexity_prompt` must be a key valid for `--dataset_id`, per `PERPLEXITY_PROMPTS_BY_DATASET` in [utils/prompts_utils.py](utils/prompts_utils.py) — defaults to `base` for valueprism and `hb_base` for habermas (a parallel set of 10 `hb_*`-prefixed prompts phrased as reacting to a stated claim rather than an opinion "about" a described event).
- **Output:** `{output_dir}/{dataset_id}/{judge_model_id}/{perplexity_prompt}.csv` for habermas, or `{output_dir}/{dataset_id}/{judge_model_id}/{generation_model_id}_{generation_prompt_version}/{perplexity_prompt}.csv` for valueprism (default `output_dir` is `model_alignment_perplexity/`). Columns: `situation_id, situation, text_id, text, perplexity`.

### [analyse_results.py](analyse_results.py)

For each judge model found under `--scoring_dir/{dataset_id}/`, enriches its scoring CSV with per-analysis characteristic columns (text length; perplexity, read from `--perplexity_dir`; judge's own embedded-opinion alignment, read from `--alignment_dir`), bins pairs by each characteristic, and computes stability (RS) and positional consistency (PC) per bin. Takes `--dataset_id` (`habermas` or `valueprism`), which is appended as a subdirectory to `--scoring_dir`, `--perplexity_dir`, `--alignment_dir`, and `--output_dir` alike (e.g. `scoring/habermas/`, `model_alignment_perplexity/habermas/`, `analysis_results/habermas/`).

- **Input:** scoring CSVs from [score.py](score.py)/[score_openrouter.py](score_openrouter.py) under `{scoring_dir}/{dataset_id}/`, perplexity CSVs from [extract_model_generation_perplexity.py](extract_model_generation_perplexity.py) under `{perplexity_dir}/{dataset_id}/`, and alignment CSVs from [extract_model_embedded_opinion.py](extract_model_embedded_opinion.py) under `{alignment_dir}/{dataset_id}/`. For `--dataset_id=valueprism`, scoring/perplexity filenames are further keyed by `--generator_model_id`/`--generator_prompt_version`; for `--dataset_id=habermas` those flags are ignored (human-written opinions have no generator).
- **Output:** `{output_dir}/{dataset_id}/{analysis_type}/{characteristic}.png` (RS & PC vs. the characteristic, one line per judge model) and `{output_dir}/{dataset_id}/{analysis_type}/{judge_model_id}/{characteristic}.csv` (per-bin RS/PC for that model). Default `output_dir` is `analysis_results/`.

## Shared code

- [utils/models_utils.py](utils/models_utils.py) — model registry (`REGISTRY`) and `load_model`.
- [utils/prompts_utils.py](utils/prompts_utils.py) — all prompt templates (`GENERATION_PROMPTS`, `SCORING_PROMPTS`, `PERPLEXITY_PROMPTS`, `ALIGNMENT_PROMPTS`), the `*_BY_DATASET` lookups that restrict each to valid keys per `--dataset_id`, `LABEL_MAPPING`/`CATEGORY_MAPPING` (output labels per `ALIGNMENT_PROMPTS` key, and how each collapses onto positive/negative/neutral — shared by `extract_model_embedded_opinion.py` and `analyse_results.py`), and chat-template helpers.
- [utils/generation_utils.py](utils/generation_utils.py) — batching helper (`batch_iterable`).
- [utils/scoring_utils.py](utils/scoring_utils.py) — pairwise-scoring helpers used by `score.py` and `score_openrouter.py`.
- [utils/evaluation_utils.py](utils/evaluation_utils.py), [utils/files_io.py](utils/files_io.py) — additional evaluation/IO helpers.
