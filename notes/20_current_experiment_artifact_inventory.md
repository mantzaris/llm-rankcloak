# Current Experiment And Artifact Inventory

This note is the current paper-production index for implemented RankCloak code,
experiment profiles, and result artifacts. It reflects the repository state after the
staged paper-main-pilot cleanup and continuation pass.

## Scope

RankCloak is an empirical study of LLM rank-transcoding steganography for deterministic
synthetic cryptographic-artifact-like payloads. The repository does not claim
encryption, key exchange, authentication, signing, credential handling, cryptographic
security, edit robustness, or undetectability.

All model-backed recovery claims assume exact-copy conditions and a shared
configuration containing the exact model file, tokenizer, quantization, rank ordering,
prompt templates, payload codec, segmentation rule, tail rule, token filter, and decode
rule where applicable.

## GPU Completion Update

The authoritative full frozen matrix is complete at
`results/rankcloak_paper_gpu_main_rank_safe/`. It contains all 475 planned
non-segmented and all 75 planned segmented trials with 550 exact recoveries and zero
recovery or runner failures. Its downstream package contains 25 baselines, 2,360
cover-feature rows, 2,445 detector-dataset rows, 60 detector results, 262 statistical
rows, 14 effect-size rows, and 10 figures. All detector, statistical, and effect-size
statuses are `ok`; an independent 3,578-check audit passed with zero errors.

A separate earlier GPU-native pilot is retained at
`results/rankcloak_paper_gpu_pilot_complete/`. It contains all 96 planned
non-segmented and all 24 planned segmented trials, with 111 exact recoveries and 9
recorded recovery failures. Its canonical downstream package contains 20 baselines,
686 detector-dataset rows, 57 detector results, 223 statistical rows, 14 effect-size
rows, and 10 figures. All detector, statistical, and effect-size status rows are
`ok`. It predates rank-safe single-token batching and is diagnostic rather than the
authoritative continuation point.

The historical `results/rankcloak_paper_main_pilot/` directory remains the partial
CPU manuscript package described below. Do not merge its rows with the complete GPU
dataset for inference; backend and model provenance are part of the decoding
configuration. See `notes/21_gpu_support_and_validation.md` for the physical backend
tests, result comparison, resume fixes, and interpretation.

## Code Inventory

| Code area | Main files | Role |
| --- | --- | --- |
| Model and tokenizer IO | `rankcloak/model_io.py` | Local GGUF model loading, tokenizer wrappers, logits access, detokenization. |
| Synthetic payloads | `rankcloak/synthetic_payloads.py`, `rankcloak/paper_payloads.py` | Deterministic synthetic payload suites for pilots and paper-main profiles. |
| Rank codecs | `rankcloak/rank_codec.py` | Stable rank ordering, bounded-rank byte codec, hex-nibble codec, direct subword rank traces, cover generation and recovery helpers. |
| Token filtering | `rankcloak/token_filters.py` | Deterministic `safe_text_filter_v1` token filter for selected segmented variants. |
| Prompts | `rankcloak/prompts.py` | Original prompt registry and prompt-family labels. |
| Metrics | `rankcloak/metrics.py` | Cover text features, artifact flags, rank and log-probability summaries. |
| Baselines | `rankcloak/baselines.py` | Greedy non-payload cover generation. |
| Standard experiment runner | `rankcloak/experiments.py`, `scripts/run_experiment.py`, `rankcloak/cli.py` | Profile registry, command-line execution, result writing, model loading. |
| Segmented protocols | `rankcloak/segmented_protocol.py` | Two-stage control-code pilot and segmented quality-controls pilot. |
| Paper suite | `rankcloak/paper_suite.py` | Staged paper-main diagnostics, generation, baselines, detector, statistics, figures, Markdown, resume logic. |
| Detector baseline | `rankcloak/detection.py` | Feature-only detector dataset and baseline models. |
| Statistics | `rankcloak/bootstrap_statistics.py` | Deterministic bootstrap intervals and effect-size helpers. |
| Plotting | `rankcloak/plotting.py`, `rankcloak/paper_suite.py` | Matplotlib-only figures for pilots and paper outputs. |
| Reproducibility | `rankcloak/reproducibility.py` | `MANIFEST.json` metadata, git state, package versions, model metadata. |
| Schemas | `rankcloak/schemas.py` | Column schemas for standard, segmented, detector, and paper-suite outputs. |

## Experiment Profiles

| Profile | Output directory | Purpose | Current status |
| --- | --- | --- | --- |
| `smoke` | `results/rankcloak_crypto_artifact_exploration/` | Initial notebook and model-backed smoke test. | Present, 4/4 stegotext recovery. |
| `small` | `results/rankcloak_small_full/` | Full-payload sweep over B=8, 16, 32, 64. | Present, 64/64 stegotext recovery. |
| `strong-prompts-pilot` | `results/rankcloak_strong_prompt_pilot/` | Fast long-prompt sanity check. | Present, 16/16 stegotext recovery. |
| `strong-prompts` | `results/rankcloak_strong_prompt_sweep/` | Strong prompt sweep. | Present, 60/60 stegotext recovery. |
| `dialogue-key-pilot` | `results/rankcloak_dialogue_key_pilot/` | Dialogue and forum prompt comparison at B=8 and B=16. | Present, 24/24 stegotext recovery. |
| `payload-granularity-pilot` | `results/rankcloak_payload_granularity_pilot/` | Payload-side representation comparison. | Present, no cover generation. |
| `segmented-protocol-pilot` | `results/rankcloak_segmented_protocol_pilot/` | Two-stage control-code and segmented multi-cover pilot. | Present, control true and 10/10 response recovery. |
| `segmented-quality-controls` | `results/rankcloak_segmented_quality_controls/` | Forced-prefix/full-message metrics, sentence tails, control tails, filtering. | Present, 5/5 control and 10/10 response recovery. |
| `paper-smoke` | `results/rankcloak_paper_smoke/` | End-to-end tiny paper-suite check. | Present, 6/6 recovery. |
| `paper-main-pilot` and staged `paper-*` profiles | `results/rankcloak_paper_main_pilot/` | Partial paper-main-pilot manuscript package. | Present, partial: 26 pass and 1 fail. |
| `paper-main` | `results/rankcloak_paper_gpu_main_rank_safe/` | Frozen full matrix and all downstream stages. | Present and audited: 550/550 exact recovery. |
| `paper-analysis` | `results/rankcloak_paper_analysis/` | Aggregation across pilot and paper-suite artifacts. | Present, analysis-only. |

## Result Directory Index

| Result directory | Key rows and recovery | Main artifacts |
| --- | --- | --- |
| `results/rankcloak_crypto_artifact_exploration/` | 4 stegotext rows, 4 pass, 0 fail; 63 codec rows. | `tokenization_audit.csv`, `rank_statistics.csv`, `codec_roundtrip_trials.csv`, `stegotext_recovery_trials.csv`, `cover_examples.jsonl`, standard figures. |
| `results/rankcloak_small_full/` | 64 stegotext rows, 64 pass, 0 fail; 68 feature rows. | Alphabet-size sweep tables and figures. |
| `results/rankcloak_strong_prompt_pilot/` | 16 stegotext rows, 16 pass, 0 fail; 20 feature rows. | `PROMPT_COMPARISON.md`, strong-prompt figures. |
| `results/rankcloak_strong_prompt_sweep/` | 60 stegotext rows, 60 pass, 0 fail; 65 feature rows. | Full strong-prompt sweep, prompt comparison Markdown, strong-prompt figures. |
| `results/rankcloak_dialogue_key_pilot/` | 24 stegotext rows, 24 pass, 0 fail; 30 feature rows. | `DIALOGUE_PROMPT_COMPARISON.md`, dialogue prompt figures. |
| `results/rankcloak_payload_granularity_pilot/` | 8 payload-representation rows; no stegotext rows. | `payload_granularity_comparison.csv`, representation figure. |
| `results/rankcloak_segmented_protocol_pilot/` | 1 control row, 10 segmented response rows, 10 pass, 0 fail. | `control_request_trial.jsonl`, `segmented_protocol_trials.csv`, `segmented_protocol_messages.jsonl`, comparison Markdown, segmented figures. |
| `results/rankcloak_segmented_quality_controls/` | 5 control rows, 10 response rows, 10 pass, 0 fail; 130 feature rows. | `control_request_trials.jsonl`, `segmented_quality_trials.csv`, `segmented_quality_messages.jsonl`, forced/full quality figures. |
| `results/rankcloak_paper_smoke/` | 2 payload rows, 4 nonseg rows, 2 segmented rows, 6 pass, 0 fail; 56 detector rows. | Complete tiny paper artifact set and 10 paper figures. |
| `results/rankcloak_paper_main_pilot/` | 12 payload rows, 20 nonseg rows, 7 segmented rows, 26 pass, 1 fail; 272 detector rows; 97 bootstrap rows. | Partial manuscript package, paper figures, detector/statistics outputs, summaries. |
| `results/rankcloak_paper_gpu_main_rank_safe/` | 35 payloads, 475 nonseg rows, 75 segmented rows, 550 pass, 0 fail; 2,445 detector rows; 262 statistical rows. | Authoritative full GPU matrix, manifest, summaries, detector/effects outputs, and 10 figures. |
| `results/rankcloak_paper_analysis/` | 8 recovery summary rows, 622 prompt-quality rows, 29 segmented summary rows, 69 detector summary rows. | Cross-pilot aggregation CSVs, summary Markdown, analysis figures. |

## Paper-Main-Pilot Current State

The staged paper-main-pilot matrix is not complete.

Current completed rows:

- Non-segmented RankCloak trials: 20 of 96 planned.
- Segmented RankCloak trials: 7 of 24 planned.
- Greedy baseline examples: 22.
- Unified cover feature rows: 234.
- Detector dataset rows: 272.
- Detector result rows: 57.
- Statistical summary rows: 97.
- Effect-size rows: 14.

Recovery:

- Non-segmented: 20 pass, 0 fail.
- Segmented: 6 pass, 1 fail.
- Total: 26 pass, 1 fail.

The failed segmented row is in
`segmented_hex_multi_topic_leadin8_sentence_tail_filtered` and should be treated as an
experimental lead-in variant limitation unless rerun and resolved. The code now replays
lead-in tokens during recovery with the same token-by-token evaluation schedule used
during generation, but the existing failed row was not replaced because the targeted
model rerun exceeded practical runtime.

## Paper Artifact Package

Current package:

```text
paper_artifacts/rankcloak_paper_main_pilot/
```

This package mirrors small manuscript-preparation artifacts from
`results/rankcloak_paper_main_pilot/` and `results/rankcloak_paper_analysis/`. It does
not include the GGUF model.

Included artifact classes:

- paper results summary;
- comparison tables;
- figure index;
- human and machine-readable summaries;
- reproducibility manifest;
- detector baseline table;
- statistical and effect-size tables;
- selected paper figures;
- cross-pilot analysis summary.

## Interpretation Status

Current evidence supports careful pilot-level claims:

- bounded-rank and hex-nibble encodings produce predictable rank constraints;
- direct subword encoding is compact but can have high rank pressure;
- B=8 and B=16 are more plausible than B=32 and B=64 in current pilots;
- prompt specificity helps topic anchoring but does not remove rank-pressure damage;
- segmented variants separate forced-prefix and full-message metrics;
- sentence tails improve public full-message metrics, but forced-prefix metrics remain essential;
- detector results are lightweight feature-only baselines on partial data.

Current evidence does not support:

- cryptographic security;
- encryption or key exchange;
- authentication or signing;
- credential handling;
- undetectability;
- robustness to edits, paraphrase, or channel normalization;
- broad cross-model generalization;
- broad human naturalness claims.

## Next Commands For Completing The Paper-Main-Pilot Matrix

Continue non-segmented generation:

```bash
python3 scripts/run_experiment.py \
  --profile paper-nonseg-generation \
  --output-dir results/rankcloak_paper_main_pilot \
  --resume \
  --limit-trials 10
```

Continue segmented generation:

```bash
python3 scripts/run_experiment.py \
  --profile paper-segmented-generation \
  --output-dir results/rankcloak_paper_main_pilot \
  --resume \
  --limit-trials 10
```

Refresh downstream artifacts after any new generation:

```bash
python3 scripts/run_experiment.py --profile paper-baselines --output-dir results/rankcloak_paper_main_pilot --resume
python3 scripts/run_experiment.py --profile paper-detector --output-dir results/rankcloak_paper_main_pilot --resume
python3 scripts/run_experiment.py --profile paper-statistics --output-dir results/rankcloak_paper_main_pilot --resume
python3 scripts/run_experiment.py --profile paper-analysis --output-dir results/rankcloak_paper_analysis --overwrite
```
