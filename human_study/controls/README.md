# Licensed human-written controls

This package prepares provenance-preserving human-written controls for the
draft RankCloak human study. It performs no recruitment, participant contact,
payment, survey exposure, or network access. It also does not authorize any of
those actions.

## Current gate

**Blocked before recruitment and stimulus exposure.** The pinned
Databricks Dolly 15k source is suitable for candidate screening, but its
automatically eligible pool does not cover all 18 frozen prompt templates. At
the prespecified minimum of eight candidates per template, 10 templates are
short. Every candidate also still requires manual topic, privacy, and safety
review. See `dolly_coverage_audit.json` for the aggregate, text-free result.

The source is therefore not being represented as a complete or final set of
fair human controls. A second clearly licensed, style-matched source or newly
authored study material would be required to fill the sparse casual,
professional-message, and first-person narrative strata. Newly authored
material would require a separate provenance and ethics decision; this package
does not solicit it.

## Frozen design

Selection follows `../config/design.json`: six English prompt categories,
three templates per category, and four eligible hexadecimal payload classes.
That gives 72 human-control strata. Exactly one unique control is selected per
stratum; each category receives 12 controls, each template four, and each
payload class 18. Segments are never treated as independent stimuli.

The control text is independent of payload content. Payload class only defines
the length-matching stratum, so it balances cover lengths without asking a
human source to imitate an encoded artifact.

## Files

- `source_registry.json` pins accepted source bytes, revision, attribution,
  license, human-authorship evidence, and exclusions.
- `category_profiles.json` maps the 18 frozen prompt templates to transparent
  topic/style screens. These scores only route candidates to manual review.
- `control_pipeline.py` implements offline verification, canonical
  deduplication, automated flags, review validation, and constrained matching.
- `prepare_controls.py` is the command-line entry point.
- `review_schema.json` defines the mandatory manual-review record.
- `target_schema.json` defines length targets produced from completed generated
  stimuli without using human ratings.
- `dolly_coverage_audit.json` is the checked-in aggregate audit. It contains no
  corpus text.
- `PROVENANCE_AND_LICENSE.md` records source research and the selection
  decision.

`raw/`, generated candidate text, reviews, targets, selected controls, and
attribution output are ignored by Git. This prevents licensed text or review
working files from being committed accidentally; it is not a security control.

## Offline workflow

1. Acquire the exact registered file outside this pipeline from the pinned
   source URL. Record the acquisition date. Do not substitute a moving branch
   or an auto-converted dataset file.
2. Put the file in an access-controlled local directory such as `raw/`.
3. Reproduce the aggregate coverage audit without writing text:

   ```bash
   .venv/bin/python human_study/controls/prepare_controls.py import \
     --input human_study/controls/raw/databricks-dolly-15k.jsonl \
     --source-id databricks_dolly_15k_v1_pinned \
     --acquisition-date 2026-08-08 \
     --output-dir human_study/controls/generated/dolly-audit \
     --audit-only
   ```

4. Only after a source set covers every template, repeat without
   `--audit-only` to create the local candidate file. This does not approve any
   candidate.
5. Review every proposed candidate against `review_schema.json`. Do not edit a
   source text to force eligibility; reject it instead.
6. Generate `length_targets.csv` from the frozen completed computational
   stimulus frame. It must contain exactly the 72 design strata.
7. Run selection:

   ```bash
   .venv/bin/python human_study/controls/prepare_controls.py select \
     --candidates human_study/controls/generated/human_control_candidates.jsonl \
     --reviews human_study/controls/generated/reviews.jsonl \
     --targets human_study/controls/generated/length_targets.csv \
     --output-dir human_study/controls/generated/final-selection
   ```

The selector fails closed for missing strata, insufficient reviewed candidates,
duplicate text, topic mismatches, automated flags, incomplete provenance, or a
relative word-count difference above 0.35. Its manifest always states that
recruitment and human exposure are unauthorized.

## Required attribution if controls are eventually used

The selector writes an `ATTRIBUTION.txt` derived from record provenance. For
the registered Dolly source, retain the Databricks copyright notice, dataset
and license links, CC BY-SA 3.0 identifier, and the normalization notice. Do not
publish selected text until the study team has confirmed that the planned
distribution and downstream paper/repository packaging satisfy the license and
institutional requirements.

Automated PII, safety, and quality expressions are deliberately conservative
screening aids. They are not a claim that unflagged text is anonymous, safe,
accurate, representative, or ethically approved.
