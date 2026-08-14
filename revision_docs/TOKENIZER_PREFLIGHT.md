# Tokenizer payload-fidelity preflight

The revision study must pass an immutable tokenizer-only preflight before any
`smoke_v3` or confirmatory generation. The preflight addresses the direct
subword incident documented in
`revision_docs/DIRECT_SUBWORD_PAYLOAD_FIDELITY_INCIDENT.md`. It is an input
gate, not an experimental outcome and is never pooled with cover-text results.

## Frozen scope

`scripts/run_revision_tokenizer_preflight.py` loads the embedded tokenizer from
each of the three content-pinned GGUF files with `llama_cpp.Llama(vocab_only=True,
n_gpu_layers=0)`. It does not evaluate a context, allocate a GPU model, compute
logits, or generate text. For every model it audits:

- all 480 public deterministic cryptographic payload strings;
- all 18 English prompt templates;
- all six Spanish secondary prompts; and
- all six Simplified-Chinese (`zh_hans`) secondary prompts.

This gives 1,440 payload checks and 90 prompt checks (1,530 total). Model paths,
sizes, and complete SHA-256 hashes are verified against `models.json`. The full
configuration manifest, selected configuration files, generated corpus hash,
payload-record hash, runtime source files, and llama-cpp-python version are
also recorded.

## Pass criteria

Payloads use the
`literal_utf8_no_special_tokens_reversible_space_prefix_v2` contract:

1. tokenize the original nonempty UTF-8 bytes with both special-token and BOS
   insertion disabled;
2. never delete a token by position;
3. detokenize the complete token sequence;
4. require the original payload bytes to be an exact suffix; and
5. permit only an explicitly recorded, reversible prefix of zero or more ASCII
   space bytes (`0x20`).

Each record stores the token IDs, input and serialized hashes, exact prefix
bytes in Base64, prefix length and hash, and the SHA-256 of the recovered
original bytes. Non-space prefixes, suffix changes, dropped payload bytes, and
tokenizer exceptions fail closed.

Prompt contexts use the
`actual_bos_only_removal_first_real_token_retention_v2` contract. A leading
token is removed only when its ID equals the tokenizer's actual BOS ID. The
result must equal an independent no-special-token tokenization, must retain its
first real token, and must recover the original prompt bytes under the same
explicit leading-space framing rule. This distinguishes tokenizers such as
Qwen, which expose a BOS ID but do not automatically prepend it, from
tokenizers that actually insert BOS.

## Immutable execution

From the repository root, run:

```bash
.venv/bin/python scripts/run_revision_tokenizer_preflight.py
```

The command publishes exactly once to
`results/revision_v1/tokenizer_preflight_v2/` using a completed temporary
directory and one same-filesystem rename. It refuses to replace any existing
destination. The directory contains:

- `records.jsonl`: every payload and prompt check in deterministic order;
- `failures.jsonl`: the exact ordered subset that failed (empty on success);
- `TOKENIZER_PREFLIGHT_MANIFEST.json`: counts, summaries, all input hashes,
  output-file hashes, and the canonical self-hash
  `preflight_manifest_sha256`.

The manifest deliberately contains no timestamp so identical inputs and
tokenizer behavior produce identical scientific bytes. Verify an existing
bundle without loading a model or tokenizer:

```bash
.venv/bin/python scripts/run_revision_tokenizer_preflight.py \
  --verify-existing
```

Both commands exit with status 2 when the scientific preflight status is not
`pass` or integrity verification fails. A failed preflight is preserved for
diagnosis; generation must not proceed until a new, explicitly versioned
contract and result namespace are approved.

## Regression tests

`tests/test_revision_tokenizer_preflight.py` includes Qwen-like no-auto-BOS,
actual-BOS, one-character fusion, two-character fusion, one- and two-space
prefix, non-space prefix, and suffix-mutation fixtures. It also verifies the
18/6/6 prompt registry, self-hashing, tamper detection, atomic publication,
and no-overwrite behavior.

