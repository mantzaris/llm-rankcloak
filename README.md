# RankCloak

RankCloak is a research notebook project for exploring LLM rank-transcoding steganography on deterministic, synthetic cryptographic-artifact-like payloads. The first notebook compares direct raw subword-token rank encoding against bounded-rank byte/bit encodings for payloads such as fake hashes, nonces, UUID-like values, fake bearer-token-like strings, invalid JWT-like strings, HMAC-like tags, and ciphertext-like blocks.

This is a concealment experiment under exact-copy conditions, not encryption, key exchange, credential handling, or a claim of cryptographic security. All examples are synthetic and deterministic.

## Setup

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[llama]"
python3 -m ipykernel install --user --name rankcloak --display-name "Python (rankcloak)"
```

If building `llama-cpp-python` is not practical on your machine, install the base package first:

```bash
pip install -e .
```

The notebook will still run the synthetic payload, bounded-rank codec, plotting, and summary sections without model logits.

## Model

The preferred local model path is:

```text
models/llama3_8b/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
```

The fallback path is:

```text
models/llama3_8b/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

From the repository root, download with:

```bash
python3 - <<'PY'
from rankcloak.model_io import download_llama3_gguf
print(download_llama3_gguf())
PY
```

If Hugging Face requires authentication or license acceptance, run `huggingface-cli login`, accept any required terms in the browser, or manually place one of the GGUF files in `models/llama3_8b/`.

## Run The Notebook

```bash
jupyter lab notebooks/01_rankcloak_crypto_artifact_exploration.ipynb
```

For a non-interactive smoke execution:

```bash
python3 scripts/run_smoke.py
```

If your Jupyter/nbconvert installation is healthy, you can also execute the notebook directly:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_rankcloak_crypto_artifact_exploration.ipynb --ExecutePreprocessor.timeout=900
```

Outputs are written under:

```text
results/rankcloak_crypto_artifact_exploration/
```

Small CSV, JSON, JSONL, Markdown, and PNG outputs are intentionally not ignored by git. Large local model files are ignored.

## Sources

The notebook uses the Calgacus paper at `https://arxiv.org/pdf/2510.20075` and design patterns inspected from `https://github.com/mantzaris/LlmStenoExplore`. The prior repository is not vendored here.
