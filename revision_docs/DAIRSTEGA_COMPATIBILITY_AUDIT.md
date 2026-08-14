# DAIRstega comparator compatibility audit

Audit date: 2026-08-08

## Decision

DAIRstega is scientifically relevant as a peer-reviewed linguistic
steganography comparator and its repository contains useful balanced example
data plus a raw-text neural steganalysis example. It is not currently suitable
for vendoring or claiming an exact official-code reproduction in RankCloak:
the audited repository has no license file, the published detector split is
row-random rather than payload-grouped, and the generation stack requires
external multi-gigabyte Llama 2 weights.

This does not block the revision. RankCloak now contains an independently
written, raw-text CNN-family equivalent and a pretrained-transformer pipeline
with stronger leakage controls. Any DAIRstega result should be labelled either
as an external-dataset evaluation or as an independent architectural
adaptation, never as an official-code reproduction.

## Sources and frozen audit target

- Peer-reviewed article: Wang et al., "Dynamically allocated interval-based
  generative linguistic steganography with roulette wheel," Applied Soft
  Computing 176 (2025), 113101,
  https://doi.org/10.1016/j.asoc.2025.113101.
- Official repository linked by the article:
  https://github.com/WangYH-BUPT/DAIRstega.
- Audited commit:
  8d85edf98d48c3efa827a125b6d4e90f88141ea2.
- Commit date and subject: 2025-07-04, "Delete DAIRstega.py".
- Audit method: shallow clone to a temporary directory; no external or
  multi-gigabyte base-model weights were downloaded, and no official source
  was copied into RankCloak.

The audit clone was 73 MB because the repository itself includes a LoRA adapter,
example data, images, a manuscript PDF, and a fine-tuning-data archive. It did
not include the Llama 2 7B/13B base weights.

## License audit

No LICENSE, COPYING, or NOTICE file exists at the audited commit. The README
does not state a software or dataset license. Consequently:

1. Do not vendor, modify, or redistribute DAIRstega source in the RankCloak
   repository or DOI archive without written permission or a subsequently
   published license.
2. Do not redistribute the six DAIRstega CSV files in the DOI archive until
   their dataset terms are clarified. A script may accept a user-supplied,
   separately obtained copy and verify its hash.
3. Cite and link the article and repository when discussing compatibility.
4. Recheck the upstream commit and license immediately before the final
   archival release.

This is a conservative reproducibility decision, not legal advice.

## Official repository contents relevant to comparison

### Generation

The README states Python 3.8, PyTorch 2.2.2, and Llama 2 7B/13B. The remaining
generation implementation is DAIRstega_t.py. The README still lists a
MindSpore DAIRstega.py file, but that file was deleted at the audited HEAD.
Generation also relies on:

- externally obtained Llama 2 base weights;
- an included LoRA adapter;
- an included one-million-bit pseudo-random stream;
- Alpaca-style prompt templates;
- optional external BERT-base weights for BERTScore.

The present audit did not execute generation because doing so would require
large model downloads and additional model-license review.

### Stego data

Six CSV files are included. Each contains 4,000 rows, balanced as 2,000 covers
and 2,000 stegos, with only text and label fields. Conditions vary alpha
(8, 16, or 32), beta (0.5 or 1.0), and reported bits per word. They do not
include payload identifiers, prompt-template IDs, source-model IDs, random
seeds, or a pairing key.

| File | SHA-256 | Duplicate text rows |
|---|---|---:|
| a8_b10_bpw110.csv | ec0d0a286ff6a3bb25a0074b0ac3636c8dbe1fdedcb4bca2830e23e75d1aa098 | 4 |
| a8_b05_bpw189.csv | 44fe11996074bea5da5229196f79d4ff452129839b7fc4766652b2c9ed1783cb | 1 |
| a16_b10_bpw113.csv | 939289ceae186326b0fbc37668a1262e6b1ed50ed36991761333bae9fc69e230 | 4 |
| a16_b05_bpw244.csv | 69ec40a8d1fc47ea976227a893c199759e58e0c0b6c5f12b2742af0aae36e6c0 | 1 |
| a32_b10_bpw111.csv | a9f7301e1e94d1bab0957471fedd514756b007d6edb6be45ea5688e3e40979c0 | 5 |
| a32_b05_bpw256.csv | 0249ca5cf923c18cc9f5156565a7c3e9e55eb77e4ed47dd067527f152ff51114 | 2 |

The duplicate counts are within-file counts relative to unique text strings.
There were no empty texts in the six files.

### Steganalysis example

The repository includes a word-level TextCNN subnetwork combined with a
bidirectional GRU, group enhancement, k-max pooling, another convolution, and a
softmax head. Its run configuration uses embedding width 300, TextCNN filter
widths 3 and 5, 100 filters, dropout 0.5, and an 80/20 split.

Frozen source hashes:

| File | SHA-256 |
|---|---|
| steganalysis/Example/TextCNN.py | 22bb8835b5d4d016c97f66366c03d63622b43106e7a00d6a47411cc57406aac2 |
| steganalysis/Example/GE.py | c60733471043d94a6f321f567dacb4846f0e582755a0e0bfa4e8e6a818d88764 |
| steganalysis/Example/data.py | 7a3d9a9af78ebc26ea84844490753991678c540b17f33f46564c124177a59ba5 |
| steganalysis/Example/run.py | cd15954bf527922a420be277c42ad43191790b636d04ffd83a82beba28a817fb |
| requirements.txt | 7197444d877ac303434cb5faaba239a6a2e75a5751a13e1c7b06a2931428a277 |

## Reproducibility limitations of the official example

The following issues prevent a fair drop-in execution for the RankCloak
confirmatory study:

- DataHelper uses NumPy random shuffling, but the runner seeds only Python's
  random module. NumPy and PyTorch are not seeded.
- The word vocabulary is built on all rows before the 80/20 split.
- Splits are individual-row splits; no payload, prompt, model, codec, or
  near-duplicate grouping is available.
- Test loss is inspected every epoch and selects the best epoch, so the nominal
  test partition serves as a validation set.
- Five fits reuse one fixed row split; the report averages the best three
  results while reporting a standard deviation over all five.
- A softmax output is passed into CrossEntropyLoss, which ordinarily expects
  unnormalized logits.
- There is no committed test suite, continuous-integration configuration,
  split manifest, checkpoint manifest, or documented expected command/output.
- The pinned requirements combine old and new packages, including PyTorch
  2.2.2 with substantially older torchtext/torchaudio pins. No lock file or
  container digest is supplied.

These observations concern the audited research code, not the scientific
validity of the peer-reviewed article.

## RankCloak compatibility matrix

| Requirement | Official DAIRstega artifact | RankCloak revision handling |
|---|---|---|
| Raw text and binary labels | Available | Directly compatible |
| Payload-grouped split | No payload key | Required and asserted |
| Held-out prompt template | No template field | Requires RankCloak manifest |
| Leave-one-model-out | No per-row model field | Requires RankCloak manifest |
| Leave-one-codec-out | DAIR alpha/beta only | RankCloak codec field required |
| Exact-text leakage check | Not present | SHA-256 train/test assertion |
| Grouped confidence intervals | Not present | Payload-group bootstrap |
| CNN-family detector | CNN is part of GE | Independent TextCNN-family equivalent |
| Pretrained transformer | Not in example | Configurable DeBERTa/AutoModel classifier |
| Offline execution | Generation needs external weights | Smoke fallback; transformer local-only by default |
| Redistribution | No stated license | No DAIR code/data vendored |

## Independent CNN-equivalent baseline

RankCloak's published_textcnn_equivalent is an independent implementation of
the raw-text CNN family described in peer-reviewed linguistic steganalysis,
including Wen et al., "Convolutional Neural Network Based Text Steganalysis,"
IEEE Signal Processing Letters 26(3), 460-464 (2019),
https://doi.org/10.1109/LSP.2019.2895286.

It also matches the core TextCNN ingredients visible in the DAIRstega example:
train-only word indices, learned embeddings, convolution widths 3 and 5, ReLU,
global max pooling, dropout, and a binary head. It deliberately does not claim
to reproduce DAIRstega's full GE CNN-BiGRU architecture.

The revision implementation differs where necessary for valid evaluation:

- vocabulary construction uses training rows only;
- all randomness is seeded;
- deterministic PyTorch algorithms are requested;
- hyperparameters and epochs are fixed before test evaluation;
- raw logits are used with cross-entropy;
- payload groups and exact text are disjoint;
- output retains requested versus actual implementation identity.

## Fair comparison options

Recommended order:

1. Primary: train the independent CNN-equivalent and pretrained transformer on
   the same RankCloak training rows and evaluate identical grouped splits.
2. Secondary external-domain test: if dataset permission is clarified, test a
   frozen RankCloak-trained detector on DAIRstega text. Report this as
   cross-source generalization and discuss cover-source mismatch.
3. Optional architecture replication: request upstream permission/license and
   a frozen split/checkpoint recipe before adapting the complete GE model.
4. Do not compare headline accuracy from the DAIRstega README directly with
   grouped RankCloak AUC; datasets, split units, tuning, and metrics differ.

## Remaining actions before confirmatory execution

- Install a reviewed PyTorch build and Transformers in the frozen revision
  environment.
- Pin a transformer model revision and hash locally cached files.
- Run the detector CLI without smoke mode and verify that every primary row has
  implementation_status equal to complete.
- Save trained checkpoints or deterministic model-state hashes for each split.
- Obtain clarification from DAIRstega authors if their source/data will be
  executed, modified, or redistributed.
- Preserve this audit commit and upstream file hashes in the DOI release.
