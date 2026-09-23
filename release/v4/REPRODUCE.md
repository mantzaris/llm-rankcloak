# Reproducing the V4 author-review archive

The archive contains the committed source tree selected by assembly_policy.json and generated ARCHIVE_SOURCE.json. PACKAGE_MANIFEST.json and PACKAGE_MANIFEST.csv are external companions listing every ZIP member with size and SHA256. SHA256SUMS records the outer archive and manifest checksums. The manifest itself and the ZIP are not embedded in the hashed payload.

## Offline scientific reconstruction

Extract the archive into a new directory. Use Python 3.10 with NumPy 2.2.6. No model weights, CUDA, network access, detector training or downloads are needed. From the extracted root run the following command, choosing an output directory outside the extraction.

```bash
python -m scripts.audit_revision_v4_stage3 --root . --output /tmp/rankcloak-v4-reconstructed
```

The script independently reads the original primary and robustness records, frozen boundary inventory, per-token scoring caches, saved semantic vectors, original Q4/Q8 shared-history records and codec metadata, and original entropy trajectories. It does not call the Stage 2 analysis, join, bootstrap or decoder helpers. It reconstructs recovery tables, the four-class selection chain, both pilot and all full-study effects with 2,000 stratified recipient-payload bootstrap draws, quantization bytes and failure runs. Compare reconstructed_effects.csv and the JSON scientific outputs with results/revision_v4/stage3/audit. Output ordering is deterministic. The output never rewrites packaged evidence.

The positive context-gain result is a fixed evaluator-token-path comparison. Standalone tail tokenization differs in 3,057 units. Both passes share target IDs and denominator. The intervals condition on the fixed control corpus. The two semantic pilot intervals cross zero. These limitations are part of the scientific result.

## Documents

The four PDF sources reside in paperV4/scientific_reports, paperV4/response and paperV4/cover_letter. TeX Live with pdflatex and bibtex builds them using the included class, bibliography, figures and tables. The build command below writes only Stage 3 logs, but rebuild in a disposable copy if preserving the extracted checksums.

```bash
python -m scripts.build_revision_v4_stage3_documents
```

main4_submission.tex is a generated single-file journal source including the compiled bibliography. Main figures are separate vector PDFs. The author-review PDFs retain one candidate label. PDF timestamps and TeX metadata can change a rebuilt file's hash. The archived PDF bytes are checked exactly, while rebuilt scientific text, references and page layout are checked separately. This distinction does not waive any data checksum.

## Model-dependent reproduction

Fresh inference is separate from offline reconstruction. It requires the exact GGUF hashes and backend libraries in configs/revision_v1, configs/revision_v3, environment and the execution manifests. See revision_docs/MODEL_AND_LICENSE_PROVENANCE.md and the model manifest paths in results/revision_v4/stage2/provenance. Qwen and Mistral have source revisions. The historical Llama quantization download revision was not retained, so its content hash is the identity and another similarly named file is not a verified replacement. Q8 is pinned in the V3 quantization configuration and generation manifests. The MiniLM revision, file hashes, license and official pooling contract are in configs/revision_v4/stage2_semantic_encoder.json. No detector checkpoint is a semantic evaluator.

Absolute /home/meow paths in historical manifests identify the original machine. They are provenance, not requirements for the offline command above. Model execution requires explicitly mapping the pinned content to the receiving machine and preserving the recorded backend/configuration contract. Do not assume CPU or other-batch replay equivalence.

## Regenerating and checking the package

From a Git checkout that contains the named source snapshot, use the full source SHA from ARCHIVE_SOURCE.json. Choose new empty paths.

```bash
python -m rankcloak.revision_v4_archive build --commit SOURCE_SHA --output /tmp/rankcloak-v4-candidate
python -m rankcloak.revision_v4_archive verify --manifest /tmp/rankcloak-v4-candidate/PACKAGE_MANIFEST.json --extract /tmp/rankcloak-v4-fresh-extraction
```

Assembly reads committed bytes through git archive. The mutable worktree and untracked files cannot enter. Verification independently checks scope against the committed policy and tree, member paths and modes, missing and extra files, all hashes and Git blob correspondence, then re-reads fresh extracted files. Stable tree order, fixed ZIP timestamps and compression parameters make output reproducible within the recorded Python/zlib runtime. Run the offline scientific command on that extraction as a separate content check.

No command here uploads files, creates a public release or submits to a journal. Follow UPLOAD_CHECKLIST.md only after author approval in a later authorized stage.
