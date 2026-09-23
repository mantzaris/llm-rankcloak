# Stage 4 publication handoff

The author has approved this exact scientific interpretation, author details, assistance disclosure and public package scope. Publication of one new version is authorized. No repeated scope approval is needed. The current session has no Zenodo connector, authenticated browser or configured API credential. No draft has been created or published by Stage 4. Private drafts could not be inspected. The live latest record still identifies published version 2.0.0, DOI 10.5281/zenodo.22555497, under concept DOI 10.5281/zenodo.21987449.

## Files ready to upload

All four files are in `/home/meow/Documents/repos/llm-rankcloak/release_artifacts/v4/7f48ae2ef2fb/`. Their bytes were checked again in Stage 4. Upload exactly these four files. Do not rebuild the ZIP or edit its internal candidate label.

| Filename | Bytes | SHA256 |
| --- | --- | --- |
| llm-rankcloak-v4-7f48ae2ef2fb.zip | 367904213 | 1aca84b5dd45208f14b93ff88cc4c18ac7af2c8c485468b5373a38ba1620b625 |
| PACKAGE_MANIFEST.json | 2960592 | a0f45ecd26de211e3108e7ba739ace4251bb019a2ba1844e12af823a84a7665d |
| PACKAGE_MANIFEST.csv | 2010833 | 5224380ae240725a152efc44dce22934857706e863d59c7cec71f8938a8467aa |
| SHA256SUMS | 188 | 5dc483baca52aa91aa0a3225513dbed27440f0f7d8ad55af178e718d819541dc |

The scientific source is commit `7f48ae2ef2fb11201ffe7a5a14c1568a9811b4e0`, tree `bcea80173eea5c98bcf588642f58e70edd42c573`. The ZIP contains 8,856 files. The JSON and CSV manifests include every payload file. SHA256SUMS covers the ZIP and JSON manifest. The Stage 4 publication manifest additionally records all four companion hashes.

## Publish once under the existing concept

1. Sign in to Zenodo in your own browser. Open [the current record](https://zenodo.org/records/22555497) and check its latest version and your uploads for an existing new-version draft. If an equivalent V4 release is already public, send its URL for verification and do not create another version. If a different later release exists, return its URL for reconciliation before publication.
2. Use **New version** on this record, or resume its existing new-version draft. Confirm that the draft belongs to concept DOI `10.5281/zenodo.21987449`. Do not use a new unrelated upload. Do not import the obsolete V2 ZIP. The completed draft must contain only the four filenames above.
3. Enter the metadata below. Use the full HTML description in `release/v4/STAGE4_PUBLIC_DESCRIPTION.html`. The API-shaped preparation file is `results/revision_v4/stage4/publication/prepared_metadata.json`. It is prepared input, not a submission receipt. Use the actual publication date, changing the prepared date if publication occurs later.
4. Review the draft against `results/revision_v4/stage4/publication/publication_manifest.json`, including the four filenames, sizes, description, source identity, creator, license exceptions and concept relationship. Keep public version **3.0.0** unless a later published version requires reconciliation. Internal **3.0.0-rc1** remains in the unchanged ZIP.
5. Publish this one version. If the interface times out, inspect the record and your uploads before retrying. A reserved DOI or saved draft does not establish public coverage. Do not publish a duplicate to work around a transient delay.
6. Send only the published record URL or version DOI to this Codex session. Do not send a token. Stage 4 will independently download and verify the public ZIP and companions before changing availability statements or finalizing journal files.

Zenodo documents the [new-version workflow](https://help.zenodo.org/docs/deposit/manage-versions/) and distinguishes [DOI reservation from publication](https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/). The prepared API fields follow its [official Deposit API](https://developers.zenodo.org/). No API mutation was attempted here.

## Public metadata

| Field | Value |
| --- | --- |
| Title | RankCloak Conceals the Surface Form of Synthetic Cryptographic Artifacts in Language Model Generated Text. Research code and data |
| Resource type | Software |
| Version | 3.0.0 |
| Publication date | Actual publication date, not the date of an unpublished draft |
| Creator | Mantzaris, Alexander V. |
| Affiliation | University of Central Florida |
| ORCID | 0000-0002-0026-5725, retained from the existing public record |
| Access | Open |
| License | MIT for repository code, with the third-party exceptions explicitly described in the HTML and included notice |
| Keywords | language models; rank transcoding; synthetic cryptographic artifacts; reproducible research |
| Repository link | https://github.com/mantzaris/llm-rankcloak, relation isSupplementTo |
| Source link | https://github.com/mantzaris/llm-rankcloak/tree/7f48ae2ef2fb11201ffe7a5a14c1568a9811b4e0, relation isDerivedFrom |
| Concept relationship | Inherited by New version under 10.5281/zenodo.21987449, never a newly invented concept |

Copy the complete description from the HTML file. It explicitly includes manuscripts, historical versions, reviews and cover letters. It distinguishes the unchanged author-review snapshot from later DOI-updated submission files and makes no claim of semantic validation, reader validation, journal submission or acceptance.

Copy the prepared metadata's notes field as well. The MIT designation does not relicense the Dolly subset, templates or manuscript/review material. Do not reuse the old manuscript-exclusion description.

## Verification and final document sequence

After the real record exists, the read-only verifier can be run from the repository with the supplied record URL and a new local download directory.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.verify_revision_v4_stage4_publication \
  --record ACTUAL_PUBLISHED_RECORD_URL \
  --downloads release_artifacts/v4/stage4/public_verification
```

It checks the public state, concept, version, metadata, exact four-file membership, all downloaded hashes, every ZIP member against the committed source, required licenses and DOI resolution. It writes new attempt records and never publishes or overwrites a historical receipt. If a public page or DOI is not yet retrievable, retain the pending condition and retry verification later with a fresh directory. The independently reproduced Stage 3 scientific results apply to the identical ZIP without repeating model work.

Only after successful public verification should Codex update CITATION.cff, README, availability text, archive/editor responses, cover letter and the Stage 4 status ledger, then remove candidate labels, rebuild all four documents and regenerate main4_submission.tex. Refresh response locations from that edition, rebuild the isolated upload directory and record final hashes against the scientific source snapshot. The archive remains unchanged. Journal submission and editor contact remain outside this authorization.
