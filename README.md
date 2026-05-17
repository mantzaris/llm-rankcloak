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
```

If installed with the project entry point:

```bash
rankcloak run --profile codec-only --overwrite
rankcloak run --profile smoke --overwrite
rankcloak run --profile small --overwrite
```

Profiles:

- `codec-only`: all payloads and all bounded alphabets; no model required.
- `smoke`: first 8 bytes of the SHA-256 payload, two cover prompts, alphabets 16 and 32.
- `small`: full selected payloads, four cover prompts, alphabets 8, 16, 32, and 64. This can be CPU-expensive.
- `audit-only`: tokenization audit and direct subword rank statistics when the model is available.

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

