# Codebase Overview

## Package Modules

`rankcloak/model_io.py`

- Defines preferred and fallback Llama 3 8B Instruct GGUF model specs.
- Loads `llama-cpp-python` with CPU-only assumptions, `logits_all=True`, and a thread count that leaves one CPU free when possible.
- Provides tokenizer and detokenizer compatibility wrappers for local GGUF experiments.
- Provides `download_llama3_gguf()` using `huggingface_hub.hf_hub_download()`.

`rankcloak/synthetic_payloads.py`

- Generates deterministic synthetic payloads from a fixed seed.
- Includes SHA-256 of a public test string, random 128-bit and 256-bit hex strings, a nonce-like hex string, UUIDv4-like value, fake bearer-token-like string, synthetic HMAC-like tag, invalid JWT-like string, and base64 ciphertext-like block.
- Marks every payload as synthetic.

`rankcloak/paper_payloads.py`

- Generates the deterministic paper payload suite.
- Provides pilot and full paper-main payload counts, payload classes, payload metadata,
  and `paper_payloads.csv` rows.
- Uses synthetic SHA-256 hex, random hex, nonce-like hex, UUIDv4-like, synthetic
  HMAC-like hex, and ciphertext-like base64 payload classes.

`rankcloak/rank_codec.py`

- Implements stable rank ordering and token/rank conversion.
- Implements direct subword rank traces for payload text.
- Implements bounded-rank byte encoding for B in `[2, 4, 8, 16, 32, 64]`.
- Implements hex-nibble rank coding for hex-like payload text, with ranks in `1..16`.
- Implements Calgacus-style cover generation and rank recovery:
  - evaluate prompt/context logits;
  - select token at requested rank;
  - append token;
  - recover ranks by replaying generated token ids.

`rankcloak/token_filters.py`

- Implements deterministic `safe_text_filter_v1`.
- Filters obvious markup, code-fence, URL, LaTeX, bracket-placeholder, and non-printable
  token fragments for selected segmented paper variants.
- Provides filtered rank selection and recovery support through shared allowed-token
  masks.

`rankcloak/tokenization_audit.py`

- Builds payload tokenization audit rows when a model is available.
- Records character length, byte length, token count, density ratios, first token ids, and decoded token pieces.

`rankcloak/metrics.py`

- Summarizes rank sequences.
- Extracts lightweight cover features:
  - character count;
  - token count;
  - line count;
  - whitespace, punctuation, digit, and alphabetic fractions;
  - unique and repeated token fractions;
  - mean and median token log probability when logits are available;
  - generated-rank summaries when ranks are available.
- Records simple artifact flags used by paper detector and quality-control outputs.

`rankcloak/prompts.py`

- Stores original, non-copyrighted cover prompts.
- Includes short prompts, long specific prompts, dialogue prompts, and family labels.
- Current prompt families include recipe, dialogue, fiction, biology, car-buying, forum, technical, code review, meeting, recipe dialogue, recipe forum exchange, car-buying dialogue, and biology dialogue.

`rankcloak/baselines.py`

- Generates ordinary greedy baseline cover text for comparison against RankCloak cover text.

`rankcloak/segmented_protocol.py`

- Implements the two-stage segmented protocol pilot.
- Implements the segmented quality-controls pilot with sentence tails, forced-prefix and
  full-message metrics, control-tail rows, and optional token filtering.

`rankcloak/paper_suite.py`

- Implements the staged paper-main suite:
  - `paper-smoke`;
  - `paper-diagnostics`;
  - `paper-nonseg-generation`;
  - `paper-segmented-generation`;
  - `paper-baselines`;
  - `paper-detector`;
  - `paper-statistics`;
  - `paper-main-pilot-resume`.
- Provides stable trial IDs, resume/skip-existing behavior, paper Markdown outputs,
  paper figures, detector/statistics stages, and artifact summaries.

`rankcloak/detection.py`

- Builds feature-only detector datasets from `paper_cover_text_features.csv`.
- Runs lightweight detector baselines without training on raw generated text.

`rankcloak/bootstrap_statistics.py`

- Provides deterministic bootstrap confidence intervals and effect-size helpers for
  paper-suite summaries.

`rankcloak/plotting.py`

- Generates matplotlib-only figures for token counts, direct rank summaries, cover length, recovery by prompt/alphabet, feature comparisons, strong prompt metrics, dialogue prompt metrics, and payload representation rank counts.

`rankcloak/reproducibility.py`

- Writes `MANIFEST.json`.
- Captures git commit and dirty state when available, Python/platform metadata, package versions, CPU/thread counts, model metadata, model file size, and model SHA-256 when feasible.

`rankcloak/schemas.py`

- Defines expected output columns for codec roundtrip, stegotext recovery, segmented
  protocol outputs, paper-suite outputs, detector outputs, and statistical summaries.

`rankcloak/experiments.py`

- Central experiment runner and profile registry.
- Writes CSV, JSONL, JSON, Markdown summaries, figures, and reproducibility manifests.
- Implements all current profiles:
  - `codec-only`;
  - `audit-only`;
  - `smoke`;
  - `small`;
  - `strong-prompts-pilot`;
  - `strong-prompts`;
  - `dialogue-key-pilot`;
  - `payload-granularity-pilot`.
  - `segmented-protocol-pilot`;
  - `segmented-quality-controls`;
  - `paper-smoke`;
  - `paper-diagnostics`;
  - `paper-nonseg-generation`;
  - `paper-segmented-generation`;
  - `paper-baselines`;
  - `paper-detector`;
  - `paper-statistics`;
  - `paper-main-pilot-resume`;
  - `paper-main-pilot`;
  - `paper-main`;
  - `paper-analysis`.

`rankcloak/cli.py`

- Exposes `rankcloak run --profile ...` through the project entry point.

## Scripts

`scripts/run_experiment.py`

- Thin wrapper around `rankcloak.experiments.main`.
- Preferred command-line entry for experiments.

`scripts/run_smoke.py`

- Legacy smoke-test entry point kept for quick early checks.

`scripts/build_notebook.py`

- Builds or updates the explanatory research notebook.

## Notebook

`notebooks/01_rankcloak_crypto_artifact_exploration.ipynb`

- Research notebook with motivation, Calgacus background, synthetic payloads, audit tables, rank statistics, bounded-rank explanation, limitations, and next experiments.
- The notebook now reads generated tables where possible; the scripts are the source of truth for repeatable runs.

## Tests

Current tests cover:

- Stable 1-indexed rank ordering and tie-breaking.
- Bounded-rank codec roundtrips for B=2, 4, 8, 16, 32, 64.
- Synthetic payload determinism and expected names.
- Result schema constants.
- Prompt registry expectations.
- Hex-nibble payload-side codec behavior.
- Paper payload determinism and schema constants.
- Resume/skip-existing behavior for staged paper outputs.
- Detector baseline dataset construction.
- Bootstrap statistics.
- Segmented protocol helpers, lead-in helpers, token filters, and sentence-tail logic.

Run:

```bash
python3 -m compileall rankcloak scripts
python3 -m pytest
```

## Git And Result Conventions

`.gitignore` excludes virtual environments, caches, build artifacts, local model files, and heavyweight result binaries. It intentionally does not ignore the full `results/` tree, so small CSV, JSON, JSONL, Markdown, and PNG outputs can be committed.

Large files such as `models/**/*.gguf` are ignored and should not be committed.
