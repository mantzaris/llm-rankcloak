# RankCloak

RankCloak is a research codebase for LLM rank-transcoding steganography over deterministic, synthetic cryptographic-artifact-like payloads. It compares direct subword-token rank behavior with bounded-rank encodings for fake hashes, random hex values, nonce-like values, UUID-like values, fake bearer-token-like strings, invalid JWT-like strings, synthetic HMAC-like tags, and ciphertext-like base64 blocks.

## Safety And Scope

RankCloak is a concealment and measurement study under exact-copy conditions. It is not encryption, key exchange, authentication, credential handling, or a claim of cryptographic security. All payloads are deterministic synthetic examples; do not add real API keys, credentials, private keys, accounts, services, or operational secrets.

Exact recovery requires the same model, tokenizer, quantization, deterministic rank ordering, prompt, and unmodified generated text.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[llama,dev]"
python3 -m ipykernel install --user --name rankcloak --display-name "Python (rankcloak)"
```

If `llama-cpp-python` is hard to build on your machine, install the base package first:

```bash
pip install -e ".[dev]"
```

The codec-only profile and tests for deterministic codec logic do not require the model.

## Model

Preferred local model path:

```text
models/llama3_8b/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
```

Fallback local model path:

```text
models/llama3_8b/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

Download with:

```bash
python3 - <<'PY'
from rankcloak.model_io import download_llama3_gguf
print(download_llama3_gguf())
PY
```

If Hugging Face requires authentication or license acceptance, run `huggingface-cli login`, accept any required terms in the browser, or manually place one of the GGUF files in `models/llama3_8b/`. Large model files are ignored by git.

## Quick Commands

```bash
python3 scripts/run_experiment.py --profile codec-only --overwrite
python3 scripts/run_experiment.py --profile smoke --overwrite
python3 scripts/run_experiment.py --profile small --overwrite
python3 scripts/run_experiment.py --profile dialogue-key-pilot --output-dir results/rankcloak_dialogue_key_pilot --overwrite
python3 scripts/run_experiment.py --profile payload-granularity-pilot --output-dir results/rankcloak_payload_granularity_pilot --overwrite
python3 scripts/run_experiment.py --profile segmented-protocol-pilot --output-dir results/rankcloak_segmented_protocol_pilot --overwrite
python3 scripts/run_experiment.py --profile segmented-quality-controls --output-dir results/rankcloak_segmented_quality_controls --overwrite
python3 scripts/run_experiment.py --profile paper-main-pilot --output-dir results/rankcloak_paper_main_pilot --overwrite
python3 scripts/run_experiment.py --profile paper-analysis --output-dir results/rankcloak_paper_analysis --overwrite
python3 scripts/run_experiment.py --profile strong-prompts-pilot --output-dir results/rankcloak_strong_prompt_pilot --overwrite
python3 scripts/run_experiment.py --profile strong-prompts --output-dir results/rankcloak_strong_prompt_sweep --overwrite
```

If installed with the project entry point:

```bash
rankcloak run --profile codec-only --overwrite
rankcloak run --profile smoke --overwrite
rankcloak run --profile small --overwrite
rankcloak run --profile dialogue-key-pilot --output-dir results/rankcloak_dialogue_key_pilot --overwrite
rankcloak run --profile payload-granularity-pilot --output-dir results/rankcloak_payload_granularity_pilot --overwrite
rankcloak run --profile segmented-protocol-pilot --output-dir results/rankcloak_segmented_protocol_pilot --overwrite
rankcloak run --profile segmented-quality-controls --output-dir results/rankcloak_segmented_quality_controls --overwrite
rankcloak run --profile paper-main-pilot --output-dir results/rankcloak_paper_main_pilot --overwrite
rankcloak run --profile paper-analysis --output-dir results/rankcloak_paper_analysis --overwrite
rankcloak run --profile strong-prompts-pilot --output-dir results/rankcloak_strong_prompt_pilot --overwrite
rankcloak run --profile strong-prompts --output-dir results/rankcloak_strong_prompt_sweep --overwrite
```

Profiles:

- `codec-only`: all payloads and all bounded alphabets; no model required.
- `smoke`: first 8 bytes of the SHA-256 payload, two cover prompts, alphabets 16 and 32.
- `small`: full selected payloads, four cover prompts, alphabets 8, 16, 32, and 64. This can be CPU-expensive.
- `dialogue-key-pilot`: a narrow comparison of monologue, dialogue, and forum-exchange key prompts at B=8 and B=16.
- `payload-granularity-pilot`: compares payload-side representations without changing the cover model/tokenizer.
- `segmented-protocol-pilot`: tests a compact control code and short multi-cover response segments with forced-prefix-only decoding.
- `segmented-quality-controls`: separates forced-prefix and full-message metrics while testing sentence tails and a deterministic safe-text token filter.
- `paper-main-pilot`: runs the smaller paper-oriented result matrix and writes paper-ready CSV, JSONL, Markdown, and PNG artifacts.
- `paper-main`: runs the larger frozen paper-oriented result matrix when CPU time is available.
- `paper-analysis`: aggregates existing pilot and paper result directories without model generation.
- `strong-prompts-pilot`: a faster comparison between short and long specific prompts.
- `strong-prompts`: a stronger prompt sweep over recipe, biology, car-buying, and comparison prompts.
- `audit-only`: tokenization audit and direct subword rank statistics when the model is available.

## Strong Prompt Sweep

The strong prompt sweep tests whether longer and more specific key prompts improve cover quality relative to shorter prompts. It compares exact recovery, mean token log probability, rank pressure, generated length, formatting artifacts, and manually inspectable text examples.

Run the pilot first:

```bash
python3 scripts/run_experiment.py --profile strong-prompts-pilot --output-dir results/rankcloak_strong_prompt_pilot --overwrite
```

Run the full sweep when CPU time is available:

```bash
python3 scripts/run_experiment.py --profile strong-prompts --output-dir results/rankcloak_strong_prompt_sweep --overwrite
```

The full sweep writes `PROMPT_COMPARISON.md` with sampled generated examples and neutral quality notes. The long prompt families are recipe writing, safe biology explanation/field-note style, and casual car-buying discussion.

## Dialogue Key Prompt Pilot

The strong prompt sweep showed that longer prompts help topic anchoring but do not solve forced-rank damage. The dialogue key prompt pilot tests whether dialogue and forum-exchange formats absorb forced-rank damage better than monologue prose. It intentionally uses only low alphabet sizes, `B=8` and `B=16`, because larger alphabets were visibly harsher on cover quality.

```bash
python3 scripts/run_experiment.py \
  --profile dialogue-key-pilot \
  --output-dir results/rankcloak_dialogue_key_pilot \
  --overwrite
```

Equivalent CLI form:

```bash
rankcloak run \
  --profile dialogue-key-pilot \
  --output-dir results/rankcloak_dialogue_key_pilot \
  --overwrite
```

The run writes `DIALOGUE_PROMPT_COMPARISON.md` with sampled generated examples and neutral quality notes.

## Payload Granularity Pilot

Do not change the cover-side tokenizer without changing the model; cover-side ranks are defined over the loaded model vocabulary. This pilot compares payload-side representations only: ASCII bytes through fixed-radix ranks, direct hex-nibble character ranks, and raw subword direct rank pressure.

```bash
python3 scripts/run_experiment.py \
  --profile payload-granularity-pilot \
  --output-dir results/rankcloak_payload_granularity_pilot \
  --overwrite
```

## Two-Stage Segmented Multi-Cover RankCloak

This pilot tests whether breaking a synthetic response payload into several short cover messages reduces cover-text drift compared with one long forced-rank message. User A and User B are assumed to already share `K_common`: the exact model file, tokenizer, quantization, deterministic rank ordering, payload codec, prompt templates, compact control prompt, segment-size rule, topic schedule rule, and forced-prefix decode rule.

The simulated flow is:

- User A hides a compact synthetic control code such as `C1`.
- User B decodes `C1` and maps it to a pre-agreed local response configuration.
- User B encodes a deterministic synthetic hex payload with `raw_hex_nibbles`.
- User B splits the rank sequence into short chunks and sends multiple cover messages.
- User A decodes only the known forced prefix of each message and ignores the natural greedy tail.

This is not encryption, key exchange, authentication, signing, or cryptographic security. The control code is not a secret or operational command; it is a compact codebook label for the research simulation.

```bash
python3 scripts/run_experiment.py \
  --profile segmented-protocol-pilot \
  --output-dir results/rankcloak_segmented_protocol_pilot \
  --overwrite
```

Equivalent CLI form:

```bash
rankcloak run \
  --profile segmented-protocol-pilot \
  --output-dir results/rankcloak_segmented_protocol_pilot \
  --overwrite
```

## Segmented Quality Controls Pilot

The first segmented protocol pilot showed exact recovery and better full-message quality when greedy tails were added, but those tail-heavy metrics were dominated by non-payload tail tokens. This follow-up separates forced-prefix metrics from full-message metrics, adds sentence-boundary tails to reduce abrupt endings, adds natural tails to compact control requests, and tests a deterministic `safe_text_filter_v1` token filter intended to reduce markup-like artifacts.

The profile still assumes User A and User B already share `K_common`, including the exact model, tokenizer, quantization, rank ordering, payload codec, prompt templates, token filter, tail policy, and forced-prefix decode rule. It is not encryption, key exchange, authentication, signing, or cryptographic security.

```bash
python3 scripts/run_experiment.py \
  --profile segmented-quality-controls \
  --output-dir results/rankcloak_segmented_quality_controls \
  --overwrite
```

Equivalent CLI form:

```bash
rankcloak run \
  --profile segmented-quality-controls \
  --output-dir results/rankcloak_segmented_quality_controls \
  --overwrite
```

## Paper Main Results Suite

The paper-main suite is the locked, paper-oriented result framework. It compares the
implemented payload representations and protocol variants without broad method search:
direct subword rank pressure, non-segmented ASCII fixed-radix B=8 and B=16,
non-segmented hex-nibble B=16, segmented hex-nibble variants, sentence-boundary tails,
safe-text filtering, and an experimental lead-in segmented variant.

Run the pilot first:

```bash
python3 scripts/run_experiment.py \
  --profile paper-main-pilot \
  --output-dir results/rankcloak_paper_main_pilot \
  --overwrite
```

Run aggregation without new generation:

```bash
python3 scripts/run_experiment.py \
  --profile paper-analysis \
  --output-dir results/rankcloak_paper_analysis \
  --overwrite
```

Run the full matrix when CPU time is available:

```bash
python3 scripts/run_experiment.py \
  --profile paper-main \
  --output-dir results/rankcloak_paper_main \
  --overwrite
```

Equivalent CLI form:

```bash
rankcloak run --profile paper-main-pilot --output-dir results/rankcloak_paper_main_pilot --overwrite
rankcloak run --profile paper-analysis --output-dir results/rankcloak_paper_analysis --overwrite
rankcloak run --profile paper-main --output-dir results/rankcloak_paper_main --overwrite
```

The paper profiles write `paper_payloads.csv`, `paper_rank_pressure.csv`,
`paper_codec_comparison.csv`, `paper_stegotext_trials.csv`,
`paper_segmented_trials.csv`, `paper_cover_text_features.csv`,
`detector_dataset.csv`, `detector_baseline.csv`, `statistical_summary.csv`,
`effect_size_summary.csv`, paper Markdown summaries, reproducibility manifests, and
paper figures.

## Notebook

```bash
jupyter lab notebooks/01_rankcloak_crypto_artifact_exploration.ipynb
```

The notebook explains the research context and loads the tables produced by `scripts/run_experiment.py`; it is not the only source of truth.

## Results

Outputs are written under `results/rankcloak_crypto_artifact_exploration/`.

- `tokenization_audit.csv`: payload lengths and tokenizer behavior when a model is loaded.
- `rank_statistics.csv`: direct subword rank statistics for selected payloads.
- `codec_roundtrip_trials.csv`: byte/rank codec roundtrip results without cover generation.
- `stegotext_recovery_trials.csv`: full RankCloak channel generation and exact recovery results.
- `cover_examples.jsonl`: RankCloak-generated cover text examples.
- `baseline_cover_examples.jsonl`: ordinary greedy cover text baselines.
- `cover_text_features.csv`: lightweight plausibility/detectability features for RankCloak and baseline text.
- `MANIFEST.json`: reproducibility metadata with package versions, git state, profile, and model metadata.
- `summary.json`: machine-readable run summary.
- `SUMMARY.md`: human-readable run summary.
- `PROMPT_COMPARISON.md`: prompt-oriented manual inspection report for strong prompt runs.
- `DIALOGUE_PROMPT_COMPARISON.md`: prompt-oriented manual inspection report for dialogue pilot runs.
- `payload_granularity_comparison.csv`: payload-side representation comparison for the payload granularity pilot.

Small CSV, JSON, JSONL, Markdown, and PNG results are intentionally committable. Large local model files and heavyweight binary artifacts are ignored.

## Paper-Readiness Roadmap

- Full payload sweep across more cover genres.
- Detector AUC experiments after feature extraction stabilizes.
- Distribution-matched rank coding.
- Model comparison across Phi, Llama, Mistral, Qwen, and Gemma.
- Edit robustness tests for whitespace, punctuation, platform copying, and paraphrase.
- Human or LLM plausibility study.

## Sources

The notebook and code use the Calgacus paper at `https://arxiv.org/pdf/2510.20075` and implementation ideas inspected from `https://github.com/mantzaris/LlmStenoExplore`. The prior repository is not vendored here.
