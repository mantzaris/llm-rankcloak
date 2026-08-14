# Published patient-Huffman comparator audit

## Decision

An end-to-end numeric comparison with the official patient-Huffman
implementation is **not currently defensible**. The isolated Huffman
mathematical kernel is reproducible, but the official historical cover
generator is not runnable from the pinned checkout and a run in its native
configuration would not be a fair RankCloak comparison.

This result does not block the primary RankCloak experiment matrix. Dai and
Cai (ACL 2019), *Towards Near-imperceptible Steganographic Text*, should be
discussed as peer-reviewed prior work. The successful kernel check must not be
reported as a cover-text, recovery, quality, or steganalysis baseline.

## Frozen provenance

The read-only checkout at external_sources/lm-steganography was audited with
scripts/audit_published_comparator.py.

| Item | Audited value |
|---|---|
| Upstream | https://github.com/falcondai/lm-steganography.git |
| Historical tag | acl-2019 |
| Annotated tag object | 5c44fbd55a10ffd638075f0fec221d67236bcae2 |
| Historical commit | 0f0c41061242688e1830e16de5718f04662b10b2 |
| Historical commit date | 2019-03-09 |
| Fetched origin/master | f0ee4b0097a90ffe368c52adf61f2601970c0435 |
| Later commit date | 2019-08-07 |
| GPT-2 gitlink | c5b9c8924be4543e7aadd4b050cedfea9c910e3d |
| External checkout before/after | Clean and unchanged |

Selected historical source fingerprints:

| File | Git blob | SHA-256 |
|---|---|---|
| core.py | 0c9e5f10e3ccdd3c477afe1487e3bcef8fcaae76 | 9a1f1b13f27951039361d0a4b05c3f87a533fcd15999bdaec57f9290cde93365 |
| huffman.py | 9dcbb6b14ed6f040d2a79f07a3e5d4985a20c053 | c9befbf9cd47391634a9f2f71fb75300cf1d639f88dd2a81b423938f61cee9e7 |
| gptlm.py | 0d057bf2a5f3ff0f5cc63c610a9029dcdeb77eaf | 5affb7b424013417fa44525b68f75c0f5bae730a2db27c80acffaff187ee90ab |
| encoder.py | 7605e3ed1da0110be9b55c133d4d2e70e889afca | ecde08deb69ffe1c1aba295f6c34f54e657081e3e721c4653f622d935407436b |

The historical tree contains .gitmodules and a GPT-2 gitlink; it does not
contain the GPT-2 source checkout itself. The submodule status begins with a
minus sign, meaning the submodule is uninitialized. The expected
external/gpt-2/src/model.py and GPT-2 117M model files are absent.

## What the official code implements

At each autoregressive step, the historical sender constructs a Huffman tree
from the language model's conditional token distribution. It consumes
payload bits only if the total variation between the model distribution and
the Huffman-induced distribution is below a threshold; otherwise it samples
from the language model normally. The receiver reconstructs the tree from the
same prefix and converts the observed token path back to bits.

The historical source locations are:

- sender and patient-Huffman control: core.py, lines 52--92;
- total-variation acceptance test: core.py, line 69;
- receiver reconstruction and recovery: core.py, lines 126--156;
- Huffman total-variation calculation: huffman.py, lines 37--72;
- TensorFlow and GPT-2 loader: gptlm.py, lines 6 and 13--44.

The historical example uses GPT-2 117M, an end-of-text/SOS prefix, 32 payload
bits, total-variation threshold 0.08, sender seed 123, and a default 80-token
maximum. It does not expose the balanced, topic-conditioned prompt protocol
planned for RankCloak.

## Reproduced checks

### Standalone Huffman kernel

Using the official build_min_heap, huffman_tree, and tv_huffman functions with
p = [0.5, 0.25, 0.125, 0.125] produced:

    {
      "code_tree": [0, [1, [2, 3]]],
      "cross_entropy_gap": 0.0,
      "total_variation": 0.0
    }

This is the expected result for the dyadic distribution. It establishes that
the small, dependency-light Huffman/TV code path executes correctly. It does
not instantiate GPT-2, generate a cover, serialize or retokenize text, recover
a payload, or measure naturalness and detectability.

### Official historical entry point

Running core.py with bytecode writing disabled exits with status 1:

    ModuleNotFoundError: No module named 'tensorflow'

The failure occurs when core.py imports GptLanguageModel from gptlm.py. The
audit did not install TensorFlow, initialize the GPT-2 submodule, or download
model assets. The later requirements file pins tensorflow 1.12.0, numpy
1.16.1, and torch 1.0.1.post2, which describes a 2019-era environment. Even
if the import were satisfied, the pinned historical code subsequently
requires the missing GPT-2 source and 117M checkpoint/model files.

## License and code-provenance interpretation

The acl-2019 tree itself contains neither a LICENSE nor a requirements.txt
file. The single later origin/master commit adds an MIT license and dependency
pins, along with documentation, samples, and code changes. The later license
text is identifiable as MIT and names Falcon Dai; its SHA-256 is
fa25f1f6faea89568194b6c65518831079c9b863d7e191b0a0050f2fc5652ebc.

The best repository-provenance interpretation is that this later MIT license
likely covers repository code distributed on that branch, including code
derived from the historical implementation. The historical tag archive
nevertheless omits the license notice. This is a provenance observation, not
a legal opinion; redistribution should preserve the later MIT notice, and
the third-party GPT-2 code and model assets require separate review.

The license timing is **not** the decisive reason to omit a numeric
comparison. The decisive problems are the incomplete historical runtime and
assets, followed by the model, tokenizer, prompting, and study-design
mismatch.

## Prespecified fair-comparison gates

| Gate | Required design | Current status |
|---|---|---|
| Prompts | Identical prompt instances or prospectively counterbalanced conditioning | Not met: historical core begins from GPT-2 SOS |
| Model and tokenizer | Same model where possible, otherwise comparable scale with an explicit sensitivity design | Not met: GPT-2 117M versus planned modern 7--8B instruction models |
| Payloads | Same independent payload bitstrings and balanced assignments | Not met: only a historical 32-bit example is locally defined |
| Controls and lengths | Prompt-, model-, length-, and stopping-matched ordinary-LM controls | Not met |
| Recovery | Encoding, serialization, decoding, and exact recovery under declared assumptions | Not met |
| Evaluation | Same blinded human, automated-quality, and steganalysis pipeline | Not met |
| Implementation validity | Official runnable implementation or a port validated against official outputs | Not met |

Zero of seven required gates are currently met. Running the official code
unchanged, even after reconstructing its environment, would conflate the
algorithm with a much smaller model, an older tokenizer, unprompted
generation, and different stopping behavior. Adapting patient-Huffman to the
RankCloak models and prompts would be a new implementation. It could become a
defensible comparator only after validation against official test vectors or
outputs and should be labeled a faithful reimplementation, not the untouched
official method.

The later branch's 20 controlled and 20 uncontrolled example files do not
solve this problem: they are not matched to RankCloak prompts, models,
payload corpus, lengths, recovery trials, or blinded evaluation.

## Machine-readable audit

Run:

    .venv/bin/python scripts/audit_published_comparator.py \
      --output /tmp/rankcloak-published-comparator-audit.json

The script also prints the complete JSON report to stdout. It:

- verifies the expected tag, commits, origin, hashes, and clean checkout;
- inventories historical and later license/dependency files;
- records the GPT-2 gitlink, initialization state, and model-asset state;
- runs the no-bytecode Huffman and official-entry probes locally;
- records all seven fair-comparison gates and the decision;
- refuses to overwrite an existing report unless --overwrite is supplied;
- exits nonzero for a provenance mismatch or checkout mutation.

No probe performs network access, dependency installation, submodule
initialization, model download, or modification of the official method.

## Response-letter-ready wording

> We audited the authors' ACL 2019 patient-Huffman implementation at its
> tagged commit. Its standalone Huffman total-variation calculation reproduced
> the expected zero value for a dyadic test distribution, but the historical
> end-to-end generator could not be executed because its TensorFlow 1-era
> environment, initialized GPT-2 submodule, and 117M model assets were absent.
> Moreover, its GPT-2 117M SOS-conditioned protocol is not matched to our
> models or prompts. We therefore compare the methods conceptually and do not
> report a potentially misleading numeric baseline.

This wording must not imply that patient-Huffman recovery, cover quality, or
detectability was empirically measured in the RankCloak revision.
