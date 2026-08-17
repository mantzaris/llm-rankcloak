# Citation audit for the major revision

Audit date: 2026-08-16. The original numbering below follows the V1 manuscript. Publisher pages, DOI records, conference proceedings, standards bodies, and authoritative bibliographic records were preferred over search-result metadata. The revised bibliography removes tangential references and cites version-of-record metadata where it can be verified.

## References singled out by Reviewer 1

| V1 no. | V1 item | Verification evidence | V2 action |
|---:|---|---|---|
| 1 | Norelli and Bronstein, “LLMs can hide text in other text of the same length,” arXiv:2510.20075 | The official ICLR 2026 program and OpenReview camera-ready PDF identify the paper as an ICLR 2026 conference paper by Antonio Norelli and Michael M. Bronstein: <https://iclr.cc/virtual/current/papers.html> and <https://openreview.net/pdf/d2a94e061dd040d3e72b9d5e3d679f1276bae01f.pdf>. | Replaced the arXiv record with the ICLR 2026 conference version. |
| 12 | Wang et al., “Dynamically allocated interval-based generative linguistic steganography with roulette wheel,” arXiv:2401.15656 | Elsevier’s version of record gives *Applied Soft Computing* **176**, 113101 (2025), DOI 10.1016/j.asoc.2025.113101: <https://doi.org/10.1016/j.asoc.2025.113101>. DBLP independently reports the same authors, volume, article number, year, and DOI: <https://dblp.org/rec/journals/asc/WangSLZL25>. | Replaced with the peer-reviewed journal article. |
| 28 | Bai et al., “Towards next-generation steganalysis: LLMs unleash the power of detecting steganography,” arXiv:2405.09090 | Searches of publisher and proceedings sources found only the arXiv/CoRR record. The authoritative arXiv page is <https://arxiv.org/abs/2405.09090>; no version of record was verified. | Removed. The revised discussion of neural steganalysis relies on peer-reviewed CNN, graph-neural, and few-shot linguistic-steganalysis papers, and reports the new RankCloak detector experiments directly. |
| 32 | Roger and Greenblatt, “Preventing language models from hiding their reasoning,” arXiv:2310.18512 | The authoritative arXiv record remains <https://arxiv.org/abs/2310.18512>. No peer-reviewed version of the same work was verified in publisher, PMLR, or OpenReview searches. | Removed because encoded chain-of-thought is tangential to the RankCloak method and not needed to support a revised claim. |
| 34 | Zolkowski et al., “Early signs of steganographic capabilities in frontier LLMs,” arXiv:2507.02737 | The ICLR 2026 program/OpenReview record identifies an accepted ICLR 2026 conference paper with the same five authors: <https://openreview.net/pdf/01e47868a99b856ea5d75e0b736b0cd03e555f2d.pdf>. | Replaced the preprint with the ICLR 2026 conference version. |

## Other replacements and checks

- Sadasivan et al., previously entered as an arXiv item with a publication note, is now cited as the 2025 *Transactions on Machine Learning Research* article, ISSN 2835-8856, verified at <https://openreview.net/forum?id=OOgsAZdFOt>.
- The revised bibliography retains peer-reviewed linguistic-steganography papers from ACL/EMNLP/NAACL, ACM Multimedia, IEEE journals, Springer proceedings, and *Computational Linguistics*. DOI, author order, year, venue, and pagination were checked against the existing publisher-identified records in V1 and, where revised, against the source linked in `references.bib`.
- The Ding, Wang, and Tao paper suggested by Reviewer 2 is a peer-reviewed ACL 2020 paper on cross-lingual positional representations for machine translation (pages 1679--1685; DOI 10.18653/v1/2020.acl-main.153), verified at <https://aclanthology.org/2020.acl-main.153/>. It is not a linguistic-steganography method or detector baseline. It is therefore not cited as one. The response letter explains this scope decision; the multilingual RankCloak results are framed as exact-copy recovery tests, not as cross-lingual representation learning.

## Necessary non-journal sources

The revised bibliography retains a small number of authoritative standards rather than replacing them with secondary papers: NIST FIPS 180-4 for SHA-2, NIST SP 800-38D for AES-GCM, and IETF RFCs 2104, 4648, 8032, 8439, and 9562 for HMAC, base encodings, Ed25519, ChaCha20--Poly1305, and UUIDs. These are normative algorithm/format specifications and are the appropriate sources for the deterministic test-vector procedures. Software/model identifiers, exact revisions, hashes, and configuration paths are reported as reproducibility metadata in Supplementary Note S2 rather than presented as scholarly evidence.

## Residual unrefereed-source assessment

No research claim in the V2 main text depends on an unverified preprint. The revised reference list has no arXiv-only research entry. ICLR and TMLR records without conventional page ranges or DOIs are cited by their official proceedings/OpenReview records. Normative standards and software metadata remain necessarily non-journal sources and are clearly used only for algorithm or implementation identity.
