# Revision-v1 model and license provenance

This record distinguishes upstream model licenses and revisions from the
content-addressed GGUF artifacts used for the confirmatory study.  The model
weights are local execution inputs and are excluded from the source/data release.
Their byte sizes and SHA-256 digests are verified before every model-backed run.

| Study ID | Upstream source and immutable revision | Local execution artifact | License and use note |
| --- | --- | --- | --- |
| `llama3_8b_instruct_q4_k_m` | `meta-llama/Meta-Llama-3-8B-Instruct`; the inherited local QuantFactory artifact is pinned by content hash because its original download revision was not preserved | `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf`; 4,920,734,272 bytes; SHA-256 `86c8ea6c8b755687d0b723176fcd0b2411ef80533d23e2a5030f845d13ab2db7` | Meta Llama 3 Community License. The upstream model card permits research use but identifies non-English use as outside its intended scope. Spanish and Mandarin results are therefore secondary computational stress tests, not supported-language or naturalness claims. |
| `qwen2_5_7b_instruct_q4_k_m` | `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`; quantization package `bartowski/Qwen2.5-7B-Instruct-GGUF` revision `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031` | `Qwen2.5-7B-Instruct-Q4_K_M.gguf`; 4,683,074,240 bytes; SHA-256 `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423` | Apache License 2.0. |
| `mistral_7b_instruct_v0_3_q4_k_m` | `mistralai/Mistral-7B-Instruct-v0.3` revision `c170c708c41dac9275d15a8fff4eca08d52bab71`; quantization package `bartowski/Mistral-7B-Instruct-v0.3-GGUF` revision `61fd4167fff3ab01ee1cfe0da183fa27a944db48` | `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf`; 4,372,812,000 bytes; SHA-256 `1270d22c0fbb3d092fb725d4d96c457b7b687a5f5a715abe1e818da303e562b6` | Apache License 2.0. |

License sources were checked on 8 August 2026 against the official upstream
model repositories. The study does not redistribute the weights. A future DOI
package must include these identifiers, hashes, license links, and acquisition
instructions, but not the local artifacts themselves.

## Execution backend

- Backend: `llama-cpp-python` 0.3.23, with the tokenizer embedded in each GGUF.
- Exact replay policy: one model at a time; serial rank evaluation
  (`n_batch=1`, `n_ubatch=1` for replay-sensitive operations).
- Primary GPU: NVIDIA RTX 5000 Ada Generation, UUID
  `GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf`, 32,760 MiB; driver 590.48.01.
- Every run manifest records the physical GPU UUID, library versions, frozen
  configuration hashes, model artifact hash, source-tree state, invocation, and
  output checksums.

## Authoritative license links

- Meta Llama 3: <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/LICENSE>
- Qwen2.5-7B-Instruct: <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE>
- Mistral-7B-Instruct-v0.3: <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3>
