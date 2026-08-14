# Human-control source and license decision

Status date: 2026-08-08. This is a research provenance record, not legal
advice. Only official project, dataset, or license pages were used for the
licensing decision.

## Selected for candidate screening

### Databricks Dolly 15k, revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`

The pinned official dataset card says that thousands of Databricks employees
created the instruction/response records and were explicitly instructed not to
use generative AI. It identifies Databricks, Inc. as copyright holder and CC
BY-SA 3.0 as the dataset license. The exact registered JSONL contains 15,011
records, 13,085,339 bytes, and has SHA-256
`2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`.

Authoritative records:

- Pinned dataset card and attribution:
  <https://huggingface.co/datasets/databricks/databricks-dolly-15k/blob/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/README.md>
- Exact pinned source file:
  <https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/databricks-dolly-15k.jsonl>
- CC BY-SA 3.0 Unported deed and legal code:
  <https://creativecommons.org/licenses/by-sa/3.0/> and
  <https://creativecommons.org/licenses/by-sa/3.0/legalcode.en>

The import excludes context-bearing examples and Dolly categories that depend
on Wikipedia or are poor prose/style matches. It admits only `creative_writing`,
`general_qa`, and `open_qa` records to automated screening. Each retained
candidate carries dataset and record identifiers, source and text hashes,
attribution, license, acquisition date, and a precise normalization notice.

CC BY-SA permits reuse subject to its terms, including appropriate credit and
ShareAlike for adaptations. The official deed also warns that privacy,
publicity, and moral rights may still matter. Accordingly, open licensing is
only the first gate: automated screening and manual review remain mandatory.

## Coverage result

The no-text aggregate audit found 540 candidates eligible for manual review,
but eligibility is concentrated in forum/Q&A, procedural, and two explanatory
topics. With a minimum pool of eight candidates per template, coverage was:

| Prompt category | Eligible counts across its three templates | Decision |
|---|---:|---|
| Casual conversation | 1, 0, 1 | insufficient |
| Professional communication | 1, 1, 1 | insufficient |
| Forum/question-answer | 69, 112, 219 | sufficient pending review |
| Procedural instructions | 23, 11, 8 | sufficient pending review |
| Personal narrative/blog | 0, 2, 2 | insufficient |
| Factual explanatory prose | 53, 3, 33 | one template insufficient |

Ten of 18 templates are therefore below threshold. Category-level totals are
not allowed to mask a missing template. No final 72-control set can be selected
from this source alone without relaxing the frozen matching rules, so the gate
is `BLOCKED_INSUFFICIENT_AUTOMATED_COVERAGE_AND_PENDING_MANUAL_REVIEW`.

## Sources considered but not admitted

| Source | Authoritative licensing evidence | Decision and reason |
|---|---|---|
| OpenAssistant OASST1 | Official dataset and Apache-2.0 license: <https://huggingface.co/datasets/OpenAssistant/oasst1> and <https://huggingface.co/datasets/OpenAssistant/oasst1/blob/main/LICENSE> | Deferred. Human volunteer dialogue is promising for conversational styles, but thread context, many languages, private-person material, and broad unsafe-content exposure require a heavier record-level audit and adapter than completed here. |
| Stack Exchange data | Official licensing help: <https://stackoverflow.com/help/licensing> | Deferred. A fair ingest must retain post-level author, URL, contribution date/revision, and the license version applicable to that contribution. A blanket dump-level label is inadequate. |
| Project Gutenberg | Official permission and trademark terms: <https://www.gutenberg.org/policy/permission> and <https://www.gutenberg.org/policy/license> | Deferred. Public-domain status is work- and jurisdiction-specific, Project Gutenberg adds distribution/trademark terms, and historical literary prose is a weak match for contemporary messages and forum answers. |
| Enron email corpus | No authoritative blanket open-content redistribution license was located | Rejected. Research availability is not itself a redistribution license, and real employee email creates avoidable privacy and personal-data risk. |

These decisions are intentionally conservative. `source_registry.json` is the
machine-readable authority; a source cannot enter the importer merely because
it appears in this research note.

## Preconditions for adding another source

A new source entry must pin immutable bytes and a version/revision, document
human authorship, provide an authoritative license and requested attribution,
identify record-level authors when supplied, state redistribution and change
marking requirements, define an explicit adapter and category exclusions, and
pass the same no-network hash, deduplication, privacy, safety, quality, topic,
manual-review, and length-matching gates. Adding a source is a prospective
protocol change and must be recorded before its results are used.
