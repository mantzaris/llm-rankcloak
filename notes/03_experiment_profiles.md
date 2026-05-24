# Experiment Profiles

All profiles are implemented in `rankcloak/experiments.py` and can be run through either `scripts/run_experiment.py` or the `rankcloak` CLI entry point.

## Common Outputs

Most model-backed profiles write:

- `tokenization_audit.csv`
- `rank_statistics.csv`
- `codec_roundtrip_trials.csv`
- `stegotext_recovery_trials.csv`
- `cover_examples.jsonl`
- `baseline_cover_examples.jsonl`
- `cover_text_features.csv`
- `MANIFEST.json`
- `summary.json`
- `SUMMARY.md`
- `figures/`

Prompt-focused profiles also write:

- `PROMPT_COMPARISON.md` for strong prompt profiles.
- `DIALOGUE_PROMPT_COMPARISON.md` for dialogue prompt profiles.

The payload granularity profile writes:

- `payload_granularity_comparison.csv`
- `figures/payload_representation_rank_count.png`

The segmented protocol profile writes:

- `control_request_trial.jsonl`
- `segmented_protocol_trials.csv`
- `segmented_protocol_messages.jsonl`
- `SEGMENTED_PROTOCOL_COMPARISON.md`
- segmented protocol figures

The segmented quality-controls profile writes:

- `control_request_trials.jsonl`
- `segmented_quality_trials.csv`
- `segmented_quality_messages.jsonl`
- `SEGMENTED_QUALITY_COMPARISON.md`
- forced-prefix and full-message quality figures

The staged paper-suite profiles write:

- `paper_payloads.csv`
- `paper_rank_pressure.csv`
- `paper_codec_comparison.csv`
- `paper_stegotext_trials.csv`
- `paper_segmented_trials.csv`
- `paper_segmented_messages.jsonl`
- `paper_baseline_examples.jsonl`
- `paper_cover_text_features.csv`
- `detector_dataset.csv`
- `detector_baseline.csv`
- `statistical_summary.csv`
- `effect_size_summary.csv`
- `PAPER_RESULTS_SUMMARY.md`
- `PAPER_COMPARISON_TABLES.md`
- `PAPER_FIGURE_INDEX.md`
- `RUN_PROGRESS.json`
- paper figures

## Profiles

### `codec-only`

Purpose: verify bounded-rank payload codec behavior without requiring the LLM model.

Payloads: all synthetic payloads.

Alphabet sizes: B=2, 4, 8, 16, 32, 64.

Command:

```bash
python3 scripts/run_experiment.py --profile codec-only --overwrite
```

### `audit-only`

Purpose: run tokenization audit and direct subword rank statistics where a model is available. If the model is missing, non-model parts still run.

Command:

```bash
python3 scripts/run_experiment.py --profile audit-only --overwrite
```

### `smoke`

Purpose: very fast model-backed correctness check.

Payloads: first 8 bytes of `sha256_public_test_string`.

Prompts: `play_dialogue`, `recipe_blog`.

Alphabet sizes: B=16, B=32.

Default output: `results/rankcloak_crypto_artifact_exploration/`.

### `small`

Purpose: first full-payload empirical sweep.

Payloads:

- `sha256_public_test_string`
- `random_128_bit_hex`
- `random_256_bit_hex`
- `synthetic_uuid_v4_like`

Prompts:

- `play_dialogue`
- `recipe_blog`
- `forum_reply`
- `technical_documentation`

Alphabet sizes: B=8, B=16, B=32, B=64.

Command used for a completed full run:

```bash
python3 scripts/run_experiment.py \
  --profile small \
  --output-dir results/rankcloak_small_full \
  --overwrite
```

### `strong-prompts-pilot`

Purpose: fast comparison between short prompts and long specific prompts.

Payloads:

- `sha256_public_test_string`
- `random_128_bit_hex`

Prompts:

- `recipe_blog`
- `recipe_long_specific`
- `biology_long_specific`
- `car_buying_long_specific`

Alphabet sizes: B=16, B=32.

Default output: `results/rankcloak_strong_prompt_pilot/`.

### `strong-prompts`

Purpose: broader prompt-quality sweep over longer and more specific key prompts.

Payloads:

- `sha256_public_test_string`
- `random_128_bit_hex`
- `synthetic_uuid_v4_like`

Prompts:

- `recipe_blog`
- `recipe_long_specific`
- `biology_long_specific`
- `car_buying_long_specific`
- `forum_reply`

Alphabet sizes: B=8, B=16, B=32, B=64.

Default output: `results/rankcloak_strong_prompt_sweep/`.

### `dialogue-key-pilot`

Purpose: narrow comparison of dialogue-style and forum-exchange prompts against recipe monologue prompts at low alphabet sizes.

Payloads:

- `sha256_public_test_string`
- `random_128_bit_hex`

Prompts:

- `recipe_blog`
- `recipe_long_specific`
- `recipe_dialogue_specific`
- `recipe_forum_exchange_specific`
- `car_buying_dialogue_specific`
- `biology_tutor_dialogue_specific`

Alphabet sizes: B=8, B=16.

Default output: `results/rankcloak_dialogue_key_pilot/`.

Command:

```bash
python3 scripts/run_experiment.py \
  --profile dialogue-key-pilot \
  --output-dir results/rankcloak_dialogue_key_pilot \
  --overwrite
```

### `payload-granularity-pilot`

Purpose: compare payload-side representations without changing the model or cover-side tokenizer.

Payloads:

- `sha256_public_test_string`
- `random_128_bit_hex`

Representations:

- `ascii_bytes_fixed_radix` at B=8 and B=16.
- `raw_hex_nibbles` for hex-like payloads, one hex character per rank at B=16.
- `raw_subword_direct` as direct LLM-token baseline, with observed direct subword rank pressure when the model is available.

Default output: `results/rankcloak_payload_granularity_pilot/`.

Command:

```bash
python3 scripts/run_experiment.py \
  --profile payload-granularity-pilot \
  --output-dir results/rankcloak_payload_granularity_pilot \
  --overwrite
```

### `segmented-protocol-pilot`

Purpose: test a two-stage segmented multi-cover RankCloak flow with a compact synthetic control code and forced-prefix-only response decoding.

Payloads:

- `sha256_public_test_string`
- `random_128_bit_hex`

Payload codec:

- `raw_hex_nibbles`

Conditions:

- `single_long_recipe_no_tail`
- `single_long_recipe_tail40`
- `segmented_single_topic_no_tail`
- `segmented_single_topic_tail40`
- `segmented_multi_topic_tail40`

Default output: `results/rankcloak_segmented_protocol_pilot/`.

Command:

```bash
python3 scripts/run_experiment.py \
  --profile segmented-protocol-pilot \
  --output-dir results/rankcloak_segmented_protocol_pilot \
  --overwrite
```

### `segmented-quality-controls`

Purpose: follow up the segmented protocol pilot with separate forced-prefix and full-message metrics, sentence-boundary tails, natural control tails, and deterministic safe-text token filtering.

Payloads:

- `sha256_public_test_string`
- `random_128_bit_hex`

Conditions:

- `segmented_single_topic_fixed_tail40_unfiltered`
- `segmented_single_topic_sentence_tail_unfiltered`
- `segmented_multi_topic_sentence_tail_unfiltered`
- `segmented_single_topic_sentence_tail_filtered`
- `segmented_multi_topic_sentence_tail_filtered`

## See Also For Paper Production

- `notes/09_methodology_inventory.md` maps each profile to methodology, code locations, result directories, and paper role.
- `notes/10_results_index.md` indexes the current output artifacts for each profile.
- `notes/13_paper_figures_tables_plan.md` maps profiles to candidate paper figures and tables.

Default output: `results/rankcloak_segmented_quality_controls/`.

Command:

```bash
python3 scripts/run_experiment.py \
  --profile segmented-quality-controls \
  --output-dir results/rankcloak_segmented_quality_controls \
  --overwrite
```

### `paper-smoke`

Purpose: tiny end-to-end paper-suite validation that produces every expected paper
artifact.

Payloads: one `sha256_hex` instance and one `random_128_bit_hex` instance.

Prompts: `recipe_long_specific`.

Variants: direct rank pressure, non-segmented B=16 variants, and one segmented
single-topic filtered variant.

Default output: `results/rankcloak_paper_smoke/`.

Current result: 4 non-segmented rows, 2 segmented rows, 6/6 exact recovery.

### Staged `paper-main-pilot`

Purpose: CPU-practical manuscript package built in resumable stages.

Default output: `results/rankcloak_paper_main_pilot/`.

Current status:

- 12 payload rows.
- 20/96 non-segmented trials complete.
- 7/24 segmented trials complete.
- 22 greedy baseline examples.
- 272 detector dataset rows.
- 97 statistical summary rows.
- 26 recovery passes and 1 recovery failure.

The one failure is in the experimental
`segmented_hex_multi_topic_leadin8_sentence_tail_filtered` variant and should be
reported separately from the non-lead-in segmented variants.

Stages:

- `paper-diagnostics`: payload, direct rank-pressure, and codec diagnostics.
- `paper-nonseg-generation`: non-segmented RankCloak generation.
- `paper-segmented-generation`: segmented RankCloak generation.
- `paper-baselines`: greedy baseline generation.
- `paper-detector`: feature-only detector dataset and baseline.
- `paper-statistics`: bootstrap summaries, effect sizes, figures, and paper Markdown.
- `paper-main-pilot-resume`: staged sequence for batched continuation.

Resume commands:

```bash
python3 scripts/run_experiment.py \
  --profile paper-nonseg-generation \
  --output-dir results/rankcloak_paper_main_pilot \
  --resume \
  --limit-trials 10
```

```bash
python3 scripts/run_experiment.py \
  --profile paper-segmented-generation \
  --output-dir results/rankcloak_paper_main_pilot \
  --resume \
  --limit-trials 10
```

After generation batches:

```bash
python3 scripts/run_experiment.py --profile paper-baselines --output-dir results/rankcloak_paper_main_pilot --resume
python3 scripts/run_experiment.py --profile paper-detector --output-dir results/rankcloak_paper_main_pilot --resume
python3 scripts/run_experiment.py --profile paper-statistics --output-dir results/rankcloak_paper_main_pilot --resume
```

### `paper-main`

Purpose: larger frozen paper-main matrix for later CPU time.

Default output: `results/rankcloak_paper_main/`.

Current status: implemented but not present as a completed result directory.

### `paper-analysis`

Purpose: aggregate existing pilot and paper-suite result directories without model
generation.

Default output: `results/rankcloak_paper_analysis/`.

Current result: present with recovery, payload-representation, prompt-quality,
segmented-protocol, and detector summary tables.

## CLI Equivalents

After installing the package:

```bash
rankcloak run --profile smoke --overwrite
rankcloak run --profile small --output-dir results/rankcloak_small_full --overwrite
rankcloak run --profile strong-prompts --output-dir results/rankcloak_strong_prompt_sweep --overwrite
rankcloak run --profile dialogue-key-pilot --output-dir results/rankcloak_dialogue_key_pilot --overwrite
rankcloak run --profile payload-granularity-pilot --output-dir results/rankcloak_payload_granularity_pilot --overwrite
rankcloak run --profile segmented-protocol-pilot --output-dir results/rankcloak_segmented_protocol_pilot --overwrite
rankcloak run --profile segmented-quality-controls --output-dir results/rankcloak_segmented_quality_controls --overwrite
rankcloak run --profile paper-smoke --output-dir results/rankcloak_paper_smoke --overwrite
rankcloak run --profile paper-nonseg-generation --output-dir results/rankcloak_paper_main_pilot --resume --limit-trials 10
rankcloak run --profile paper-segmented-generation --output-dir results/rankcloak_paper_main_pilot --resume --limit-trials 10
rankcloak run --profile paper-analysis --output-dir results/rankcloak_paper_analysis --overwrite
```
