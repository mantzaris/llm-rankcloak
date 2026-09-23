"""Replace answer sets while leaving all thirteen exact review blocks untouched."""
import argparse,re
from rankcloak.revision_v4_stage2_common import ROOT,OUT,read_json,atomic_json
from scripts.prepare_revision_v4_stage1 import escape
p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');args=p.parse_args()
base='results/revision_v4/stage2/'
rows=[]
def answer(key,label,status,direct,evidence,locations,paths,anchors,pending=None):
 rows.append({'id':key,'old_label':label,'status':status,'direct_answer':direct,'evidence_change':evidence,'location_text':locations,'evidence_paths':paths,'manuscript_anchors':anchors,'pending_work':pending})
answer('editor_accuracy','editor requirements','complete',
 'We have revised the claims to distinguish surface-form concealment, exact configured recovery, visible-text transport, detectability and reader-perceived naturalness. Adverse and inconclusive outcomes remain visible.',
 'The V4 text corrects the historical raw and limited unmodified references to saved-token recovery, documents the unfiltered sampled-skip entropy experiment, and separates the directional fixed-message quantization endpoint from free generation. The final numerical check links statements to retained tables.',
 'Abstract, main Results under Boundary validation and targeted diagnostics, Discussion, and Supplementary Notes S16 through S19.',
 [base+'quantization/summary.csv',base+'transport/cover_summary.csv','results/revision_v4/source_tables/entropy_six_failures.csv'],['sec:v4-results','sec:v4-boundaries','si:v4-theory'])
answer('archive','code deposit and archive coverage','pending',
 'The underlying code has a published DOI. Version 2.0.0 is 10.5281/zenodo.22555497 and covers V3, not the V4 work. We retain that correct published citation. A new V4 deposit has not been published.',
 'The archive audit verified the published contents and found that its description incorrectly excludes manuscript files. We prepared a local inventory and a corrected description for a future version, including the V4 source, evidence and document files.',
 'Main Data and Code availability. Local release specification and prospective inventory.',
 ['results/revision_v4/provenance/archive_v2_coverage.json','revision_docs/REVISION_V4_RELEASE_SPEC.md',base+'release/prospective_inventory.csv'],['sec:data-code'],
 'INTERNAL PENDING WORK. Assemble from the exact finalized commit, verify the archive inventory, publish an authorized new version, and insert its real DOI only after coverage is checked. No deposit or submission was made in this stage.')
answer('editor_technical','editor technical assessment','partial',
 'The revision adds targeted transport and quantization endpoints, the full filter specification, all six entropy-failure diagnostics, a configuration compatibility check and source-based theoretical analysis. Semantic validation remains incomplete.',
 'Both blinded pilots support conditional context gain but do not validate the semantic-relatedness metric. We retain that limitation and label the subsequent likelihood study exploratory. We do not claim that automated scores prove reader judgments.',
 'Main Methods under Boundary instrumentation and prospective control validation, targeted Results, and Supplementary Notes S16 through S19.',
 [base+'analysis/pilot_initial/summary.json',base+'analysis/pilot_amendment1/summary.json','revision_docs/REVISION_V4_STAGE2_THEORY.md'],['sec:v4-boundary-method','si:v4-coherence'],
 'INTERNAL PENDING WORK. R1.4 remains partly unresolved because the prospective two-metric control-validation requirement was not met. Final archive coverage is also pending.')
answer('reviewer2','reviewer2','complete',
 'We thank Reviewer 2 for acknowledging the earlier revisions.',
 'The acknowledgment is retained in full with the current reviewer label. This revision responds separately to the new editor and Reviewer 1 requests.',
 'This response, Reviewer 2 acknowledgment. No manuscript change is needed for this acknowledgment.',
 ['paperV4/response/requests.txt','paperV4/response/review_comments.json'],[])
answer('general','general assessment','complete',
 'We agree that recovery requires a synchronized configuration and that the observed transport and detection failures substantially restrict practical claims. The revision presents these as limitations, not as problems solved by a new robustness method.',
 'The discussion states that a fully configured attacker may decode, distinguishes trace-informed detection from visible-text scoring, and labels the block/ECC fallback as unimplemented. The unsuccessful semantic pilots remain visible. The additional transport diagnostics are illustrative rather than population estimates.',
 'Abstract, main Discussion under Configuration sensitivity, detection and a proposed fallback, limitations, and Supplementary Notes S17 through S19.',
 ['revision_docs/REVISION_V4_STAGE2_THEORY.md',base+'transport/summary.json',base+'analysis/pilot_amendment1/summary.json'],['sec:v4-boundaries','sec:limitations','si:v4-coherence'])
answer('wording','verbatim wording preservation','complete',
 'We have retained the complete review wording without simplification or paraphrase, including both general paragraphs and this instruction.',
 'The thirteen exact quote blocks reconstruct the original request file. The validator checks each escaped block against the unchanged source and checks the original file hash. Answers are separate from quoted comments.',
 'All review-comment blocks in this response. The original requests file and exact-block ledger remain unchanged.',
 ['paperV4/response/requests.txt','paperV4/response/review_comments.json'],[])
answer('R1.1','R1.1','complete',
 'The tested Markdown operation is a specific composite transform. It adds a blockquote prefix to each line and removes whitespace. The receiver then applies saved token offsets after retokenization. This can shift the forced span, change tokens and alter contextual ranks. It does not demonstrate failure for every Markdown workflow.',
 'Five hash-selected sources give 25 decomposed arms and 135 segment instances. Exact reuse leaves 26 missing segment replays. We also checked all 53 reused segment inputs at the new context allocation, with identical rank and error outcomes. These checks add no logical cover arms. Prefix-only insertion recovers none of the five. Declared wrapper removal recovers one originally failing Mistral case, but cannot restore removed whitespace. Byte edits, extracted spans, first rank differences, errors and recovered bytes are retained. Original composite recovery remains 0/144.',
 'Main Methods under Decoder compatibility and bounded transport diagnostics, targeted Results, and Supplementary Note S18 with its illustrative recovery table.',
 [base+'transport/freeze.json',base+'transport/cover_summary.csv',base+'transport/segment_diagnostics.jsonl',base+'gpu/resource_summary.json'],['sec:v4-compatibility','sec:v4-results','si:v4-transport','tab:v4-transport'])
answer('R1.2','R1.2','complete',
 'A minor version change has no universal recovery-rate implication. It may preserve the relevant tokenization or change token identities, offsets and conditioning. Model or gate differences introduce other disagreement mechanisms. We state the required shared contract and conditional agreement argument explicitly.',
 'A new canonical configuration fingerprint includes eleven required contract components. Tests accept matching configurations, verify deterministic serialization and reject changes to every component. It is a compatibility API, not authentication or a robustness guarantee, and was not retroactively applied to the historical messages.',
 'Main Methods under Decoder compatibility and bounded transport diagnostics, main Discussion under Configuration sensitivity, detection and a proposed fallback, and Supplementary Note S19.',
 ['rankcloak/revision_v4_compatibility.py',base+'compatibility/example_header.json',base+'compatibility/validation.json','tests/test_revision_v4_stage2.py'],['sec:v4-compatibility','sec:v4-boundaries','si:v4-theory'])
answer('R1.3','R1.3','complete',
 'Zero cross-model recovery remains a severe limitation. We propose configuration identification before decoding, independently framed blocks with context resets, integrity checks, bounded outer error/erasure correction and retransmission. This is a conceptual fallback, not an implemented improvement.',
 'The argument states the minimum-distance condition 2e+s<d and its assumptions. ECC cannot reconstruct an arbitrary wrong model codebook or repair persistent synchronization loss. Reliable framing, a supported configuration and overhead for redundancy and retries remain necessary. Retransmission under an unresolved mismatch is ineffective.',
 'Main Discussion under Configuration sensitivity, detection and a proposed fallback, and Supplementary Note S19.',
 ['revision_docs/REVISION_V4_STAGE2_THEORY.md'],['sec:v4-boundaries','si:v4-theory'])
if args.preview:
 coherence_evidence='The frozen exploratory context-gain sample contains 92 payloads and 2,750 eligible boundaries. Scoring is in progress and no partial processed set is reported as complete.'
else:
 s=read_json(OUT/'analysis/coherence_study/summary.json');a=next(r for r in s['effects'] if r['model_id']=='all' and r['metric']=='context_gain' and r['contrast']=='actual minus ordinary');b=next(r for r in s['effects'] if r['model_id']=='all' and r['metric']=='context_gain' and r['contrast']=='actual minus shuffled')
 coherence_evidence=f"The frozen exploratory study completed {s['boundaries']:,} boundaries across {s['payload_clusters']} payloads. Actual minus ordinary context gain was {a['effect']:.4f} nats per tail token, with a 95% interval [{a['ci_low']:.4f}, {a['ci_high']:.4f}]. Actual minus shuffled gain was {b['effect']:.4f} [{b['ci_low']:.4f}, {b['ci_high']:.4f}]. Eligibility, fixed-corpus intervals, donor reuse and influence, and predeclared weak, typical and strong examples are reported. The ordinary comparison also differs in continuation policy, sampled controls versus greedy RankCloak tails. It does not isolate a causal boundary effect. This is not a completed two-metric coherence evaluation."
answer('R1.4','R1.4','partial',
 'We tested complementary semantic-relatedness and independent-model conditional-compatibility metrics. The semantic metric did not pass the prospective control-separation requirement in either of two blinded pilots. Context gain did pass. We therefore cannot claim that the requested transitions are proven natural or logically connected to readers.',
 'The pilots each cover 36 identities. Initial semantic separation was 0.0078 with interval [-0.0416, 0.0572]. The one predeclared 16-token amendment on disjoint recipient and donor payloads gave -0.0106 [-0.0728, 0.0519]. Both context-gain intervals were positive. '+coherence_evidence,
 'Main Methods under Boundary instrumentation and prospective control validation, targeted Results and its context-gain table, and Supplementary Note S17.',
 [base+'analysis/pilot_initial/summary.json',base+'analysis/pilot_amendment1/summary.json',base+'plans/coherence_study/freeze.json']+([] if args.preview else [base+'analysis/coherence_study/summary.json']),['sec:v4-boundary-method','sec:v4-results','si:v4-coherence'],
 'INTERNAL PENDING WORK. Semantic and perceptual support remains unresolved. No further metric search was undertaken after the allowed amendment. Any future design needs a new prospective rationale or reader study, not relabeling this negative validation as success.')
answer('R1.5','R1.5','complete',
 'The evaluated model-aware attack is trace informed and exposes a strong statistical limitation. Its features come from saved token likelihoods under the generating model. Reproducing them requires the original token path, model/backend, prompts, boundaries, resets and probability convention. A visible-text-only scorer is a distinct condition.',
 'The added discussion explains that payload-rank distributions need not match ordinary sampling. It distinguishes untempered full-vocabulary scores from the temperature-0.8 top-p-0.95 sampler and states the support conditions for a sequence-KL argument. AUC is not converted to KL, and no unique causal feature attribution is claimed. An attacker with the complete receiver configuration may decode. No confidentiality or informed-attacker indistinguishability is asserted.',
 'Main steganalysis Methods and Discussion under Configuration sensitivity, detection and a proposed fallback. The source-based argument is in the Stage 2 theory audit.',
 ['rankcloak/revision_v3_analysis.py','revision_docs/REVISION_V4_STAGE2_THEORY.md'],['sec:detector-method','sec:v4-boundaries'])
answer('R1.6','R1.6','complete',
 'All six strict-gate failures are retained and analyzed at their original budgets. Five are Qwen and one Mistral, all ASCII B16. They consume 110/128, 116/128, 54/64, 106/128, 59/128 and 116/128 ranks. Strict completion remains 114/120.',
 'The supplement reports exact pinned-tokenizer windows, longest below-threshold runs of 50, 163, 39, 35, 267 and 51 positions, rolling progress, margins and rank pressure. The longest runs contain 30 to 157 distinct token IDs, at most three identical IDs consecutively, and rank-one frequencies of 85.7 to 98.8 percent. All 605 positions remain ordinary sampled skips with no payload progress. Twelve matched ungated/moderate trajectories and three hash-selected strict successes are retained. Observed contexts include formatted public-service and project-update passages, but low-entropy runs also occur in successes. These are descriptive associations, not established linguistic causes. The trajectories used no token filter and sampled ordinary skips, not greedy skips.',
 'Main targeted Results and Supplementary Notes S13 and S16, including the six-case table, trajectory figure and exact local windows.',
 ['results/revision_v4/source_tables/entropy_six_failures.csv','results/revision_v4/source_tables/entropy_positions.csv','results/revision_v4/source_tables/entropy_text_windows.csv',base+'entropy/failure_comparators.csv',base+'entropy/longest_run_characteristics.json'],['sec:v4-results','si:s13','si:v4-stage1','tab:v4-entropy-runs'])
answer('R1.7','R1.7','complete',
 'The new comparison separates model work from sorting, rank counting, probability passes and coding operations. Rank selection sorts all allowed candidates, whereas recovery of a known token counts greater logits and lower-ID ties.',
 'The analysis inspects the primary arithmetic-coding and SAAC implementations and ADG grouping algorithm. It includes repeated diagnostic sorts, finite-precision cumulative tables, grouping search/removal, prompt prefills, KV state, logits_all storage, tails, sampled skips, segmentation and payload-side direct-subword passes. Bounds are implementation-specific and imply no cross-backend wall-time superiority.',
 'Main Computational overhead and generalizability and Supplementary Note S19, implementation-specific operation-count table.',
 ['revision_docs/REVISION_V4_STAGE2_THEORY.md',base+'provenance/primary_sources/manifest.json'],['sec:results-overhead','si:v4-theory','tab:v4-complexity'])
answer('R1.8','R1.8','complete',
 'The bibliography key Cai2025EntropyGuidedWatermarking refers to the previous reference 16, arXiv:2504.12108. The inspected primary record lists no journal reference. We have replaced its role in the V4 argument with the peer-reviewed Lee and colleagues ACL 2024 paper on entropy-threshold selective watermarking.',
 'The V4 prose now cites Lee2024SWEET for the limited thresholding precedent. It distinguishes detecting a watermark from recovering a chosen payload. Neither a watermark theorem nor the unverified preprint is used as a RankCloak recovery guarantee. The historical experiment itself is still documented by its original configuration and data.',
 'Main Introduction, Supplementary Note S13 and the V4 bibliography.',
 ['paperV4/scientific_reports/references.bib','revision_docs/REVISION_V4_STAGE2_THEORY.md'],['sec:introduction','si:s13'])
answer('R1.9','R1.9','complete',
 'The main Methods now list every literal blocked substring and every other safe_text_filter_v1 exclusion criterion from the actual unchanged code.',
 'The specification covers empty and replacement pieces, ASCII controls and exceptions, whitespace/start rules, backslashes, individual-token decoding and exceptions, mask application, special-token distinctions and stable tie-breaking. It distinguishes full selection sorting from counted rank recovery and notes limitations for concatenated markup and nonfinite logits. The versioned source link and source hash are retained in the audit, with focused edge-case tests.',
 'Main Methods under Complete deterministic token filter specification and its complete rule table.',
 ['rankcloak/token_filters.py','results/revision_v4/provenance/filter_contract.json','results/revision_v4/source_tables/filter_rules.csv','paperV4/scientific_reports/v4_filter_methods.tex'],['sec:v4-filter','tab:v4-filter'])
answer('R1.10','R1.10','complete',
 'We separate same-history logit perturbations from subsequent feedback when independently generated token paths split. For a fixed history and allowed set, pairwise gaps above twice a uniform absolute perturbation bound preserve order. Smaller gaps and ties have no such protection. This is a conditional bound, not a measured quantization-error distribution.',
 'The audit reconciles 69,528 rank changes across 244,440 positions in 1,920 pairs. Encoded messages contribute 56,321/122,220 and ordinary controls 13,207/122,220. The distinct offline fixed-message endpoint gives zero exact Q4-to-Q8 payload recoveries in 960 supported rows, with 766 invalid-rank trials and 194 valid but wrong decoded payloads. It supports no reverse-direction claim. Observed-token replay feeds received cover tokens, not decoded payload symbols, back into the model. Existing evidence supports these claims without new neighboring-logit replay.',
 'Main targeted Results and Supplementary Notes S14, S18 under Directional cross-quantization endpoint, and S19.',
 [base+'quantization/manifest.json',base+'quantization/q4_to_q8_fixed_message_decode.jsonl'],['sec:v4-results','si:s14','si:v4-fixed-quant','si:v4-theory'])
path=ROOT/'paperV4/response/response_to_reviewers_v4.tex';source=path.read_text()
for r in rows:
 block='% BEGIN ANSWER '+r['id']+'\n'+r'\textbf{'+escape(r['id']+' response, '+r['status'])+r'.}'+ '\n\n'+r'\textbf{Direct answer.} '+escape(r['direct_answer'])+'\n\n'+r'\textbf{Evidence and change.} '+escape(r['evidence_change'])+'\n\n'+r'\textbf{Locations.} '+escape(r['location_text'])
 if r['pending_work']:block+='\n\n'+r'\textcolor{red}{'+escape(r['pending_work'])+'}'
 block+='\n% END ANSWER '+r['id']
 if '% BEGIN ANSWER '+r['id'] in source:pattern=r'% BEGIN ANSWER '+re.escape(r['id'])+r'\n.*?% END ANSWER '+re.escape(r['id'])
 else:pattern=r'\\textbf\{Internal pending work for '+re.escape(r['old_label'])+r'\.\}\n\n\\textbf\{Direct answer pending\.\}[^\n]*\n\n\\textbf\{Evidence and change pending\.\}[^\n]*\n\n\\textbf\{Location pending\.\}[^\n]*'
 source,n=re.subn(pattern,lambda _:block,source,count=1,flags=re.S)
 if n!=1:raise ValueError('answer set not found '+r['id'])
source=source.replace('This response is a Stage 1 scaffold for submission','This response is a Stage 2 working revision for submission').replace('Pending responses are internal drafting instructions and must be resolved before submission.','Genuine pending items remain marked. In particular, semantic validation and final V4 archive coverage are unresolved.')
path.write_text(source)
atomic_json(OUT/'response_evidence.json',{'stage':2,'preview_execution_pending':args.preview,'response_sets':rows,'quote_blocks':13})
print('Updated sixteen answer sets with '+str(sum(r['status']=='complete' for r in rows))+' complete responses')
