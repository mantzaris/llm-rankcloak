# RankCloak reference audit

Audit date: 9 August 2026. This audit covers every entry in the submitted
`paperV1/scientific_reports/references.bib` and the citation order compiled into
`main.bbl`. The submitted manuscript, Supplementary Information, inline
bibliography, and `references.bib` were not edited. The staged revision is
`paperV2/scientific_reports/references.bib`.

## Outcome

- The submitted bibliography contains 43 entries, of which 40 occur in the
  compiled main-paper reference list. The staged bibliography contains 48
  entries.
- Ref. 1, Calgacus, is upgraded from arXiv v6 to its peer-reviewed ICLR 2026
  conference paper, OpenReview `tmFQWuIheV`.
- Ref. 12 (DAIRstega) is replaced by its peer-reviewed *Applied Soft Computing*
  version, volume 176, article 113101 (2025), DOI
  `10.1016/j.asoc.2025.113101`.
- Ref. 28 (Bai et al.) and Ref. 32 (Roger and Greenblatt) remain arXiv-only and
  are removed as nonessential to the revised claims. The stronger detector and
  AI-agent claims are supported by peer-reviewed sources instead.
- Ref. 34 (Zolkowski et al.) is replaced by the ICLR 2026 conference version.
- Ref. 31 (Sadasivan et al.), although not singled out in the reviewer list, is
  upgraded from an arXiv record to the January 2025 *Transactions on Machine
  Learning Research* article and its complete title.
- The valid but non-peer-reviewed Bennett CERIAS report (submitted Ref. 19) is
  removed from the revised bibliography. Historical claims remain supported by
  peer-reviewed primary sources.
- The submitted entry for the Llama 3 report had the wrong lead author; Aaron
  Grattafiori is now listed first. The submitted `llama-cpp-python` entry named
  “Andrei Kanikotan”; PyPI and the official repository identify the author as
  Andrei Betlen. Both records are corrected.
- The official title of Wayner's 1995 article is misspelled “Stegnography” in
  the version-of-record metadata. The revised bibliography preserves that
  official title rather than silently normalizing it.
- The peer-reviewed patient-Huffman prior work is already present as
  `DaiCai2019NearImperceptible` (ACL 2019, DOI `10.18653/v1/P19-1422`) and is
  retained for the comparator discussion.

The revised list therefore limits academic preprints to three model-family
technical reports (Llama 3, Qwen2.5, and Mistral 7B). Model reports support
provenance only, not the paper's scientific claims. Standards, RFCs, and software records are authoritative primary
technical sources and should not be described as peer-reviewed research.

## Reviewer-highlighted records

| Submitted number | Submitted key | Decision | Verified replacement or reason |
| --- | --- | --- | --- |
| 1 | `NorelliBronstein2025Calgacus` | Replace | Peer-reviewed ICLR 2026 conference paper, OpenReview `tmFQWuIheV`. |
| 12 | `wang2025dynamicallyallocatedintervalbasedgenerative` | Replace | *Applied Soft Computing* 176, 113101 (2025), DOI `10.1016/j.asoc.2025.113101`. |
| 28 | `Bai2024NextGenerationSteganalysis` | Remove | No peer-reviewed version located; revised neural-detector claims rely on peer-reviewed CNN/GNN/few-shot work and the actual RankCloak detector evaluation. |
| 32 | `RogerGreenblatt2023HidingReasoning` | Remove | No peer-reviewed version located; peripheral AI-safety paragraph is compressed and supported by peer-reviewed Motwani and Zolkowski papers. |
| 34 | `Zolkowski2025EarlySignsSteganographicCapabilities` | Replace | ICLR 2026 conference paper, OpenReview `q4qxtaKVAU`. |

Also upgraded: submitted Ref. 31, `Sadasivan2023AIDetection`, to the TMLR 2025
version. Submitted Ref. 19, the Bennett technical report, is removed. Ref. 35
(NIST FIPS 180-4) and Refs. 36--37 (RFCs) are retained because they are the
normative primary sources for the specified artifact formats, not substitutes
for peer-reviewed evidence.

## Added references and intended use

- `NIST2007GCM`, `KrawczykBellareCanetti1997HMAC`,
  `NirLangley2018ChaCha20Poly1305`, and `JosefssonLiusvaara2017EdDSA` document
  the real cryptographic algorithms used to create the confirmatory corpus.
- `Yang2024Qwen25` and `Jiang2023Mistral7B` document model-family provenance.
  The Mistral paper documents the original 7B family, not the exact v0.3
  checkpoint; the exact v0.3 model-card revision and artifact hash must remain
  in Methods/Supplementary Methods.
- `HeGaoChen2023DeBERTaV3` is the peer-reviewed architecture citation for the
  transformer steganalysis baseline.
- `DingWangTao2020CrossLingualPosition` is the exact peer-reviewed ACL 2020
  paper suggested by Reviewer 2. It concerns bilingual position representation
  for machine translation, not steganography. If cited, it should appear only
  as broader multilingual representation context; it is not a fair prior-method
  comparator and does not support recovery, quality, or security claims.

## Verification sources

DOI-bearing records were checked against the Crossref registry and the
publisher or proceedings page where available. arXiv records were checked on
arXiv; conference records without DOI were checked against PMLR, NeurIPS,
OpenReview, or ACL Anthology. Normative records were checked against NIST and
the RFC Editor.

Primary records for the material changes:

- Calgacus ICLR 2026 paper: <https://openreview.net/forum?id=tmFQWuIheV>
- DAIRstega journal article: <https://doi.org/10.1016/j.asoc.2025.113101>
- TMLR detector-robustness article:
  <https://openreview.net/forum?id=OOgsAZdFOt>
- ICLR 2026 steganographic-capabilities paper:
  <https://openreview.net/forum?id=q4qxtaKVAU>
- Patient-Huffman ACL paper: <https://aclanthology.org/P19-1422/>
- Reviewer-suggested cross-lingual paper:
  <https://aclanthology.org/2020.acl-main.153/>
- NeurIPS secret-collusion paper:
  <https://proceedings.neurips.cc/paper_files/paper/2024/hash/861f7dad098aec1c3560fb7add468d41-Abstract-Conference.html>
- NIST GCM recommendation: <https://csrc.nist.gov/pubs/sp/800/38/d/final>
- llama-cpp-python package: <https://pypi.org/project/llama-cpp-python/>

## Machine-readable submitted-entry mapping

The following JSON array is normative for the old-key audit. `submitted_number`
is `null` for entries present in `references.bib` but absent from the compiled
submitted reference list. `replace` includes metadata correction as well as a
change from preprint to version of record.

<!-- BEGIN REFERENCE_MAPPING_JSON -->
```json
[
  {"old_key":"NorelliBronstein2025Calgacus","submitted_number":1,"action":"replace","new_key":"NorelliBronstein2025Calgacus","publication_status":"peer-reviewed ICLR 2026 conference paper","rationale":"Replace arXiv v6 with the accepted ICLR 2026 version.","claims_affected":["Calgacus source method","rank transcoding"],"evidence":"https://openreview.net/forum?id=tmFQWuIheV"},
  {"old_key":"Simmons1984Prisoners","submitted_number":2,"action":"replace","new_key":"Simmons1984Prisoners","publication_status":"peer-reviewed book chapter","rationale":"Retain and add the verified chapter DOI and publisher metadata.","claims_affected":["classical steganography"],"evidence":"https://doi.org/10.1007/978-1-4684-4730-9_5"},
  {"old_key":"Cachin1998InformationTheoretic","submitted_number":3,"action":"keep","new_key":"Cachin1998InformationTheoretic","publication_status":"peer-reviewed conference chapter","rationale":"Title, authors, year, pages, and DOI verified.","claims_affected":["information-theoretic steganography"],"evidence":"https://doi.org/10.1007/3-540-49380-8_21"},
  {"old_key":"Hopper2002ProvablySecureSteganography","submitted_number":4,"action":"keep","new_key":"Hopper2002ProvablySecureSteganography","publication_status":"peer-reviewed conference chapter","rationale":"Title, authors, year, pages, and DOI verified.","claims_affected":["provably secure steganography"],"evidence":"https://doi.org/10.1007/3-540-45708-9_6"},
  {"old_key":"Ziegler2019NeuralLinguisticSteganography","submitted_number":8,"action":"keep","new_key":"Ziegler2019NeuralLinguisticSteganography","publication_status":"peer-reviewed EMNLP-IJCNLP paper","rationale":"ACL Anthology record and DOI verified.","claims_affected":["neural linguistic steganography"],"evidence":"https://aclanthology.org/D19-1115/"},
  {"old_key":"Dubey2024Llama3","submitted_number":null,"action":"replace","new_key":"Dubey2024Llama3","publication_status":"model technical report","rationale":"Correct lead author from Abhimanyu Dubey to Aaron Grattafiori; retain only for model provenance.","claims_affected":["Llama model provenance"],"evidence":"https://arxiv.org/abs/2407.21783"},
  {"old_key":"llamacpp","submitted_number":null,"action":"replace","new_key":"llamacpp","publication_status":"software","rationale":"Correct release year and identify version/source revision through the execution manifest.","claims_affected":["inference backend provenance"],"evidence":"https://github.com/ggml-org/llama.cpp"},
  {"old_key":"llamacpppython","submitted_number":null,"action":"replace","new_key":"llamacpppython","publication_status":"software","rationale":"Correct invented author surname to Andrei Betlen and pin the used version, 0.3.23.","claims_affected":["Python inference binding provenance"],"evidence":"https://pypi.org/project/llama-cpp-python/"},
  {"old_key":"Petitcolas1999InformationHidingSurvey","submitted_number":5,"action":"keep","new_key":"Petitcolas1999InformationHidingSurvey","publication_status":"peer-reviewed journal article","rationale":"Crossref title, authors, volume, pages, year, and DOI verified.","claims_affected":["information-hiding background"],"evidence":"https://doi.org/10.1109/5.771065"},
  {"old_key":"Fang2017GeneratingSteganographicTextLSTMs","submitted_number":6,"action":"keep","new_key":"Fang2017GeneratingSteganographicTextLSTMs","publication_status":"peer-reviewed ACL workshop paper","rationale":"ACL Anthology record and DOI verified.","claims_affected":["neural linguistic steganography"],"evidence":"https://aclanthology.org/P17-3017/"},
  {"old_key":"Yang2019RNNStega","submitted_number":7,"action":"keep","new_key":"Yang2019RNNStega","publication_status":"peer-reviewed journal article","rationale":"IEEE/Crossref metadata and DOI verified.","claims_affected":["recurrent neural steganography"],"evidence":"https://doi.org/10.1109/TIFS.2018.2871746"},
  {"old_key":"DaiCai2019NearImperceptible","submitted_number":9,"action":"keep","new_key":"DaiCai2019NearImperceptible","publication_status":"peer-reviewed ACL paper","rationale":"Retain as the patient-Huffman prior method and published-method comparator citation.","claims_affected":["patient-Huffman prior work","published comparator"],"evidence":"https://aclanthology.org/P19-1422/"},
  {"old_key":"Shen2020SelfAdjustingArithmetic","submitted_number":10,"action":"keep","new_key":"Shen2020SelfAdjustingArithmetic","publication_status":"peer-reviewed EMNLP paper","rationale":"ACL Anthology record and DOI verified.","claims_affected":["arithmetic-coded neural steganography"],"evidence":"https://aclanthology.org/2020.emnlp-main.22/"},
  {"old_key":"Zhang2021ProvablySecureGenerative","submitted_number":11,"action":"keep","new_key":"Zhang2021ProvablySecureGenerative","publication_status":"peer-reviewed ACL Findings paper","rationale":"ACL Anthology record and DOI verified.","claims_affected":["generative linguistic steganography security"],"evidence":"https://aclanthology.org/2021.findings-acl.268/"},
  {"old_key":"wang2025dynamicallyallocatedintervalbasedgenerative","submitted_number":12,"action":"replace","new_key":"wang2025dynamicallyallocatedintervalbasedgenerative","publication_status":"peer-reviewed journal article","rationale":"Replace arXiv metadata with the Applied Soft Computing version of record.","claims_affected":["DAIRstega","prior-method comparison"],"evidence":"https://doi.org/10.1016/j.asoc.2025.113101"},
  {"old_key":"Wu2024GenerativeTextSteganographyLLM","submitted_number":13,"action":"keep","new_key":"Wu2024GenerativeTextSteganographyLLM","publication_status":"peer-reviewed ACM Multimedia paper","rationale":"ACM/Crossref title, authors, pages, year, and DOI verified.","claims_affected":["LLM linguistic steganography"],"evidence":"https://doi.org/10.1145/3664647.3680562"},
  {"old_key":"Kirchenbauer2023WatermarkLLM","submitted_number":15,"action":"keep","new_key":"Kirchenbauer2023WatermarkLLM","publication_status":"peer-reviewed ICML paper","rationale":"PMLR title, authors, volume, pages, and year verified.","claims_affected":["LLM watermarking distinction"],"evidence":"https://proceedings.mlr.press/v202/kirchenbauer23a.html"},
  {"old_key":"YanMurawaki2025TokenizationInconsistency","submitted_number":40,"action":"keep","new_key":"YanMurawaki2025TokenizationInconsistency","publication_status":"peer-reviewed EMNLP paper","rationale":"ACL Anthology title, authors, pages, year, and DOI verified.","claims_affected":["retokenization fragility"],"evidence":"https://aclanthology.org/2025.emnlp-main.361/"},
  {"old_key":"Sennrich2016SubwordUnits","submitted_number":38,"action":"keep","new_key":"Sennrich2016SubwordUnits","publication_status":"peer-reviewed ACL paper","rationale":"ACL Anthology record and DOI verified.","claims_affected":["subword tokenization"],"evidence":"https://aclanthology.org/P16-1162/"},
  {"old_key":"KudoRichardson2018SentencePiece","submitted_number":39,"action":"keep","new_key":"KudoRichardson2018SentencePiece","publication_status":"peer-reviewed EMNLP system paper","rationale":"ACL Anthology record and DOI verified.","claims_affected":["tokenization"],"evidence":"https://aclanthology.org/D18-2012/"},
  {"old_key":"NIST2015SecureHashStandard","submitted_number":35,"action":"keep","new_key":"NIST2015SecureHashStandard","publication_status":"normative government standard","rationale":"Authoritative primary specification for SHA-256 artifacts; DOI verified.","claims_affected":["SHA-256 corpus"],"evidence":"https://doi.org/10.6028/NIST.FIPS.180-4"},
  {"old_key":"Josefsson2006RFC4648","submitted_number":36,"action":"keep","new_key":"Josefsson2006RFC4648","publication_status":"RFC","rationale":"Authoritative primary specification for Base64 encoding; DOI verified.","claims_affected":["Base64 artifact representation"],"evidence":"https://www.rfc-editor.org/rfc/rfc4648.html"},
  {"old_key":"DavisPeabodyLeach2024RFC9562","submitted_number":37,"action":"keep","new_key":"DavisPeabodyLeach2024RFC9562","publication_status":"RFC","rationale":"Current authoritative UUID specification; DOI verified.","claims_affected":["UUIDv4 corpus"],"evidence":"https://www.rfc-editor.org/rfc/rfc9562.html"},
  {"old_key":"Wayner1992MimicFunctions","submitted_number":16,"action":"keep","new_key":"Wayner1992MimicFunctions","publication_status":"peer-reviewed journal article","rationale":"Publisher/Crossref metadata and DOI verified.","claims_affected":["mimic-function history"],"evidence":"https://doi.org/10.1080/0161-119291866883"},
  {"old_key":"ChapmanDavida1997HidingHidden","submitted_number":18,"action":"keep","new_key":"ChapmanDavida1997HidingHidden","publication_status":"peer-reviewed conference chapter","rationale":"Springer/Crossref metadata and DOI verified.","claims_affected":["ciphertext-to-text history"],"evidence":"https://doi.org/10.1007/BFb0028489"},
  {"old_key":"Bennett2004LinguisticSteganography","submitted_number":19,"action":"remove","new_key":null,"publication_status":"unrefereed CERIAS research paper","rationale":"Valid institutional report but nonessential; remove to reduce reliance on technical reports.","claims_affected":["linguistic-steganography survey background"],"evidence":"https://www.cerias.purdue.edu/apps/reports_and_papers/view/2697"},
  {"old_key":"ChangClark2014PracticalLinguisticSteganography","submitted_number":20,"action":"keep","new_key":"ChangClark2014PracticalLinguisticSteganography","publication_status":"peer-reviewed journal article","rationale":"Computational Linguistics record and DOI verified.","claims_affected":["synonym-substitution steganography"],"evidence":"https://aclanthology.org/J14-2006/"},
  {"old_key":"Lin2024ZeroShotGLS","submitted_number":14,"action":"keep","new_key":"Lin2024ZeroShotGLS","publication_status":"peer-reviewed NAACL paper","rationale":"ACL Anthology authors, title, pages, year, and DOI verified.","claims_affected":["zero-shot generative linguistic steganography"],"evidence":"https://aclanthology.org/2024.naacl-long.289/"},
  {"old_key":"Gehrmann2019GLTR","submitted_number":29,"action":"keep","new_key":"Gehrmann2019GLTR","publication_status":"peer-reviewed ACL system paper","rationale":"ACL Anthology record and DOI verified.","claims_affected":["probability-based generated-text diagnostics"],"evidence":"https://aclanthology.org/P19-3019/"},
  {"old_key":"Mitchell2023DetectGPT","submitted_number":30,"action":"keep","new_key":"Mitchell2023DetectGPT","publication_status":"peer-reviewed ICML paper","rationale":"PMLR title, authors, volume, pages, and year verified.","claims_affected":["generated-text detection"],"evidence":"https://proceedings.mlr.press/v202/mitchell23a.html"},
  {"old_key":"Sadasivan2023AIDetection","submitted_number":31,"action":"replace","new_key":"Sadasivan2023AIDetection","publication_status":"peer-reviewed TMLR article","rationale":"Replace arXiv record with January 2025 TMLR publication and complete title.","claims_affected":["detector robustness under paraphrase"],"evidence":"https://openreview.net/forum?id=OOgsAZdFOt"},
  {"old_key":"Wen2019CNNTextSteganalysis","submitted_number":25,"action":"keep","new_key":"Wen2019CNNTextSteganalysis","publication_status":"peer-reviewed journal article","rationale":"IEEE/Crossref metadata and DOI verified; supports the CNN baseline family.","claims_affected":["CNN text steganalysis"],"evidence":"https://doi.org/10.1109/LSP.2019.2895286"},
  {"old_key":"Wu2021GNNLinguisticSteganalysis","submitted_number":26,"action":"keep","new_key":"Wu2021GNNLinguisticSteganalysis","publication_status":"peer-reviewed journal article","rationale":"IEEE/Crossref metadata and DOI verified.","claims_affected":["graph neural linguistic steganalysis"],"evidence":"https://doi.org/10.1109/LSP.2021.3062233"},
  {"old_key":"Wang2023FewShotLinguisticSteganalysis","submitted_number":27,"action":"keep","new_key":"Wang2023FewShotLinguisticSteganalysis","publication_status":"peer-reviewed journal article","rationale":"IEEE/Crossref metadata and DOI verified.","claims_affected":["few-shot linguistic steganalysis"],"evidence":"https://doi.org/10.1109/TIFS.2023.3298210"},
  {"old_key":"Bai2024NextGenerationSteganalysis","submitted_number":28,"action":"remove","new_key":null,"publication_status":"arXiv only","rationale":"No peer-reviewed version located; remove from central claims rather than cite a nonexecuted LLM detector.","claims_affected":["LLM-based steganalysis background"],"evidence":"https://arxiv.org/abs/2405.09090"},
  {"old_key":"Zander2007CovertChannelsCountermeasures","submitted_number":23,"action":"keep","new_key":"Zander2007CovertChannelsCountermeasures","publication_status":"peer-reviewed survey article","rationale":"IEEE/Crossref metadata and DOI verified.","claims_affected":["covert-channel countermeasures"],"evidence":"https://doi.org/10.1109/COMST.2007.4317620"},
  {"old_key":"Badar2025StegomalwareSurvey","submitted_number":24,"action":"keep","new_key":"Badar2025StegomalwareSurvey","publication_status":"peer-reviewed journal article","rationale":"Elsevier/Crossref metadata and DOI verified.","claims_affected":["stegomalware detection background"],"evidence":"https://doi.org/10.1016/j.sigpro.2025.109888"},
  {"old_key":"Wayner1995StrongTheoreticalSteganography","submitted_number":17,"action":"replace","new_key":"Wayner1995StrongTheoreticalSteganography","publication_status":"peer-reviewed journal article","rationale":"Use the version-of-record title, which spells the word 'Stegnography'; all other metadata and DOI verified.","claims_affected":["theoretical mimicry history"],"evidence":"https://www.tandfonline.com/toc/ucry20/19/3"},
  {"old_key":"Weinberg2012StegoTorus","submitted_number":21,"action":"keep","new_key":"Weinberg2012StegoTorus","publication_status":"peer-reviewed ACM CCS paper","rationale":"ACM/Crossref metadata and DOI verified.","claims_affected":["network camouflage"],"evidence":"https://doi.org/10.1145/2382196.2382211"},
  {"old_key":"Dyer2013FormatTransformingEncryption","submitted_number":22,"action":"keep","new_key":"Dyer2013FormatTransformingEncryption","publication_status":"peer-reviewed ACM CCS paper","rationale":"ACM/Crossref metadata and DOI verified.","claims_affected":["format-transforming encryption distinction"],"evidence":"https://doi.org/10.1145/2508859.2516657"},
  {"old_key":"RogerGreenblatt2023HidingReasoning","submitted_number":32,"action":"remove","new_key":null,"publication_status":"arXiv only","rationale":"No peer-reviewed version located; peripheral claim can be supported by peer-reviewed AI-agent steganography papers.","claims_affected":["encoded reasoning background"],"evidence":"https://arxiv.org/abs/2310.18512"},
  {"old_key":"Motwani2024SecretCollusion","submitted_number":33,"action":"replace","new_key":"Motwani2024SecretCollusion","publication_status":"peer-reviewed NeurIPS paper","rationale":"Retain and add the proceedings DOI omitted from the submitted entry.","claims_affected":["AI-agent collusion"],"evidence":"https://proceedings.neurips.cc/paper_files/paper/2024/hash/861f7dad098aec1c3560fb7add468d41-Abstract-Conference.html"},
  {"old_key":"Zolkowski2025EarlySignsSteganographicCapabilities","submitted_number":34,"action":"replace","new_key":"Zolkowski2025EarlySignsSteganographicCapabilities","publication_status":"peer-reviewed ICLR 2026 paper","rationale":"Replace the 2025 arXiv record with the accepted ICLR 2026 conference version.","claims_affected":["frontier-model steganographic capability"],"evidence":"https://openreview.net/forum?id=q4qxtaKVAU"}
]
```
<!-- END REFERENCE_MAPPING_JSON -->

## Machine-readable additions

<!-- BEGIN REFERENCE_ADDITIONS_JSON -->
```json
[
  {"key":"Yang2024Qwen25","status":"model technical report","intended_use":"Qwen2.5 family provenance only","evidence":"https://arxiv.org/abs/2412.15115"},
  {"key":"Jiang2023Mistral7B","status":"model technical report","intended_use":"Mistral 7B family provenance only; exact v0.3 model card also required","evidence":"https://arxiv.org/abs/2310.06825"},
  {"key":"DingWangTao2020CrossLingualPosition","status":"peer-reviewed ACL paper","intended_use":"Reviewer-suggested multilingual representation context, not a steganography comparator","evidence":"https://aclanthology.org/2020.acl-main.153/"},
  {"key":"NIST2007GCM","status":"normative government recommendation","intended_use":"AES-256-GCM corpus procedure","evidence":"https://csrc.nist.gov/pubs/sp/800/38/d/final"},
  {"key":"KrawczykBellareCanetti1997HMAC","status":"RFC","intended_use":"HMAC-SHA-256 corpus procedure","evidence":"https://www.rfc-editor.org/rfc/rfc2104.html"},
  {"key":"NirLangley2018ChaCha20Poly1305","status":"RFC","intended_use":"ChaCha20-Poly1305 corpus procedure","evidence":"https://www.rfc-editor.org/rfc/rfc8439.html"},
  {"key":"JosefssonLiusvaara2017EdDSA","status":"RFC","intended_use":"Ed25519 signature corpus procedure","evidence":"https://www.rfc-editor.org/rfc/rfc8032.html"},
  {"key":"HeGaoChen2023DeBERTaV3","status":"peer-reviewed ICLR paper","intended_use":"pretrained transformer detector architecture","evidence":"https://openreview.net/forum?id=sE7-XhLxHA"}
]
```
<!-- END REFERENCE_ADDITIONS_JSON -->

## Resolved and deliberately limited items

1. Calgacus now cites its peer-reviewed ICLR 2026 conference version; the arXiv
   record remains useful only as version provenance and is not the staged citation.
2. No peer-reviewed version of Bai et al. or Roger and Greenblatt was found.
   Both are removed rather than relabeled.
3. The Llama 3, Qwen2.5, and Mistral 7B technical reports are not peer reviewed.
   They are necessary provenance records and must be cited only where the model
   families are described.
4. The Mistral 7B paper predates and does not document the exact
   Mistral-7B-Instruct-v0.3 artifact. The immutable model-card revision and
   SHA-256 digest remain the primary reproducibility record.
5. The two software projects do not supply version-specific scholarly DOIs for
   the exact runtime used. Exact package/backend versions and source hashes must
   remain in the execution manifest and DOI release package.
6. OpenReview conference/TMLR records generally do not carry Crossref DOIs.
   Their official forum identifiers are used and must not be replaced with the
   obsolete arXiv DOI.

## Response-letter wording

**Reviewer 1, Comment 13 (reference quality).** We audited every reference and
replaced preprints with peer-reviewed versions wherever one was available.
Specifically, Calgacus (submitted Ref. 1) now cites its ICLR 2026 paper; the
former DAIRstega preprint (submitted Ref. 12) now cites the 2025 *Applied Soft
Computing* version; the former “Early Signs” preprint (submitted Ref. 34) now
cites the ICLR 2026 paper; and the AI-text-detection preprint now cites its 2025
TMLR article. We removed two nonessential arXiv
citations (submitted Refs. 28 and 32) and the non-peer-reviewed Bennett report,
added missing DOIs to the Simmons and Motwani papers, and corrected software and
model metadata. The only academic preprints retained are model-family technical
reports used solely for reproducibility; standards, RFCs, and software records are cited as
primary technical sources rather than scholarly evidence.

**Reviewer 2, suggested Ding et al. paper.** We added the exact ACL 2020 citation
for Ding, Wang, and Tao and clarified its limited connection. That work concerns
cross-lingual positional representations for machine translation; it does not
encode payloads, define a steganographic channel, or provide a recovery or
detectability comparator. We therefore cite it only as broader representation
context for the secondary multilingual panel and do not present it as a
linguistic-steganography baseline.

