"""Render V4 result inserts from completed tables. Preview keeps execution pending."""
from pathlib import Path
import argparse,json,shutil,csv
from collections import defaultdict
from rankcloak.revision_v4_stage2_common import ROOT,OUT,read_json,read_jsonl,atomic_json,file_hash
from scripts.prepare_revision_v4_stage1 import escape

p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');args=p.parse_args()
paper=ROOT/'paperV4/scientific_reports';pilots=[read_json(OUT/'analysis'/name/'summary.json') for name in ['pilot_initial','pilot_amendment1']]
lines=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{llrrr}\toprule',r'Pilot & Metric & Effect & Lower 95\% & Upper 95\% \\ \midrule']
for label,pilot in zip(['Initial','Fresh amended'],pilots):
 for r in pilot['effects']:
  if r['model_id']=='all' and r['metric'] in ['semantic_cosine','context_gain']:
   metric='Semantic cosine' if r['metric']=='semantic_cosine' else 'Context gain'
   lines.append(f"{label} & {metric} & {r['effect']:.4f} & {r['ci_low']:.4f} & {r['ci_high']:.4f}"+r' \\')
lines += [r'\bottomrule\end{tabular}',r'\caption{Blinded ordinary-minus-shuffled ordinary effects. Each pilot uses 36 identities and 2,000 payload-cluster draws. Only context gain has a positive lower interval in both pilots. Semantic validation remains unsupported.}\label{tab:v4-pilot}',r'\end{table}']
(paper/'supplementary_tables/v4_pilots.tex').write_text('\n'.join(lines)+'\n')
# Eligibility is determined before scores and keeps all selected structural units.
strata=list(csv.DictReader((OUT/'plans/coherence_study/eligibility_by_stratum.csv').open()))
eligibility=defaultdict(lambda:[0,0,0])
for row in strata:
 if row['selected_payload']=='True':
  values=eligibility[(row['model_id'],row['schedule'])]
  for i,key in enumerate(['structural_boundaries','eligible_boundaries','excluded_boundaries']):values[i]+=int(row[key])
assert [sum(v[i] for v in eligibility.values()) for i in range(3)]==[3174,2750,424]
elines=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{llrrr}\toprule',r'Generator & Schedule & Structural & Eligible & Excluded \\ \midrule']
eligibility_rows=[]
for (model,schedule),values in sorted(eligibility.items()):
 label={'llama3_8b_instruct_q4_k_m':'Llama','qwen2_5_7b_instruct_q4_k_m':'Qwen','mistral_7b_instruct_v0_3_q4_k_m':'Mistral'}[model]
 schedule_label='Multi-topic' if 'multi' in schedule else 'Single-topic'
 elines.append(f'{label} & {schedule_label} & {values[0]} & {values[1]} & {values[2]}'+r' \\')
 eligibility_rows.append({'model_id':model,'schedule':schedule,'structural':values[0],'eligible':values[1],'excluded':values[2]})
elines += [r'\bottomrule\end{tabular}',r'\caption{Selected structural boundaries before scoring. All models and both schedules are retained for every selected payload. Joint three-arm eligibility removes 424 boundaries, including two trials with no scorable boundary. Detailed artifact-class and overlapping-reason counts remain in the frozen eligibility table.}\label{tab:v4-boundary-eligibility}',r'\end{table}']
(paper/'supplementary_tables/v4_boundary_eligibility.tex').write_text('\n'.join(elines)+'\n')
atomic_json(OUT/'analysis/selected_eligibility_summary.json',eligibility_rows)
main=[r'\subsection*{Boundary validation and targeted diagnostics}\label{sec:v4-results}',r'The initial control-only pilot gave an ordinary-minus-shuffled semantic cosine difference of 0.0078 (95\% interval [\(-0.0416\), 0.0572]) and context gain of 2.2200 nats per tail token [1.7784, 2.6777]. The sole amendment shortened the right cap to 16 tokens and used controls disjoint in recipient and donor payloads. Semantic cosine remained inconclusive at \(-0.0106\) [\(-0.0728\), 0.0519], while context gain was 2.0612 [1.7378, 2.4109]. Thus the planned two-metric validation was not achieved. The following boundary study is exploratory, using the validated context-gain metric.']
if args.preview:
 main += [r'\textbf{Internal pending execution.} The frozen sample contains 92 payloads and 2,750 eligible boundaries. Scoring is in progress and no partial processed sample is reported as completed.']
 supplement=r'\textbf{Internal pending execution.} Full planned-sample scoring, effect tables, donor influence and examples are awaiting the checkpointed jobs.'
else:
 summary=read_json(OUT/'analysis/coherence_study/summary.json');freeze=read_json(OUT/'plans/coherence_study/freeze.json');examples=read_json(OUT/'analysis/coherence_study/examples.json')
 assert summary['boundaries']==freeze['selected_eligible_boundaries']==2750 and summary['missing_planned_requests']==0
 effects=[r for r in summary['effects'] if r['model_id']=='all' and r['metric']=='context_gain']
 compact=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{lrrr}\toprule',r'Contrast & Difference & Lower 95\% & Upper 95\% \\ \midrule']
 for r in effects:compact.append(f"{escape(r['contrast'])} & {r['effect']:.4f} & {r['ci_low']:.4f} & {r['ci_high']:.4f}"+r' \\')
 compact += [r'\bottomrule\end{tabular}',r'\caption{Exploratory paired context-gain differences in nats per evaluator tail token for the completed frozen sample. Segments are averaged within trials and trials within payloads. Intervals use 2,000 payload-cluster draws within artifact class and condition on the fixed donor corpus.}\label{tab:v4-coherence}',r'\end{table}']
 (paper/'supplementary_tables/v4_coherence.tex').write_text('\n'.join(compact)+'\n')
 model_labels={'llama3_8b_instruct_q4_k_m':'Llama','qwen2_5_7b_instruct_q4_k_m':'Qwen','mistral_7b_instruct_v0_3_q4_k_m':'Mistral'}
 model_table=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{llrr}\toprule',r'Generator & Contrast & Context gain [95\% interval] & Conditional [95\% interval] \\ \midrule']
 for model,label in model_labels.items():
  for contrast in ['actual minus ordinary','actual minus shuffled']:
   vals=[next(r for r in summary['effects'] if r['model_id']==model and r['metric']==metric and r['contrast']==contrast) for metric in ['context_gain','conditional_logp']]
   cells=[f"{r['effect']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]" for r in vals]
   model_table.append(label+' & '+escape(contrast)+' & '+' & '.join(cells)+r' \\')
 model_table += [r'\bottomrule\end{tabular}',r'\caption{Generator-specific paired effects under the fixed cross-family evaluator map. Values are in nats per evaluator tail token. Context gain is exploratory and raw conditional likelihood is secondary. Each interval uses the same 2,000 stratified payload draws as the pooled analysis. The evaluator and tokenization vary with generator, limiting interpretation of differences across rows.}\label{tab:v4-coherence-models}',r'\end{table}']
 (paper/'supplementary_tables/v4_coherence_models.tex').write_text('\n'.join(model_table)+'\n')
 a=next(r for r in effects if r['contrast']=='actual minus ordinary');b=next(r for r in effects if r['contrast']=='actual minus shuffled')
 main += [f"The completed frozen sample contains {summary['boundaries']:,} boundaries from {summary['trials']} scored trials and {summary['payload_clusters']} payloads. Actual minus ordinary context gain was {a['effect']:.4f} nats per tail token (95\\% interval [{a['ci_low']:.4f}, {a['ci_high']:.4f}]). Actual minus shuffled gain was {b['effect']:.4f} [{b['ci_low']:.4f}, {b['ci_high']:.4f}]. These contrasts concern short within-message transitions under the selected evaluator map. They do not validate sentence-level semantic coherence, reader judgments or transitions between deliberately rotated prompts.",r'\input{supplementary_tables/v4_coherence.tex}']
 conditional=next(r for r in summary['effects'] if r['model_id']=='all' and r['metric']=='conditional_logp' and r['contrast']=='actual minus ordinary')
 null_models=[model_labels[r['model_id']] for r in summary['effects'] if r['model_id']!='all' and r['metric']=='conditional_logp' and r['contrast']=='actual minus ordinary' and r['ci_low']<=0<=r['ci_high']]
 main.append(f"The secondary actual-minus-ordinary conditional-likelihood difference was {conditional['effect']:.4f} nats per tail token [{conditional['ci_low']:.4f}, {conditional['ci_high']:.4f}]. Generator-specific conditional intervals included zero for {escape(' and '.join(null_models))}. RankCloak tails use greedy continuation, while ordinary controls are sampled. The comparison therefore includes a continuation-policy difference and does not isolate a causal boundary effect.")
 shutil.copy2(OUT/'analysis/coherence_study/boundary_coherence.pdf',paper/'figures/v4_boundary_coherence.pdf')
 supp=[f"The measured-cost admission selected {freeze['selected_payloads']} of 240 payloads, with {freeze['payloads_per_class']} per class, preserving all models and schedules. The selected structural inventory contains {freeze['selected_structural_boundaries']:,} boundaries in {freeze['selected_planned_trials']} trials. The completed analysis contains {summary['boundaries']:,} eligible boundaries, {summary['scoring_units']:,} arm-specific units and {summary['trials']} trials. Every planned request completed. Selection used no RankCloak metric scores. The sample supports the reported precision only, not a guaranteed effect or a claim about excluded boundaries.",
 r'\begin{figure}[H]\centering\includegraphics[width=\textwidth]{figures/v4_boundary_coherence.pdf}\caption{Paired boundary metric differences with payload-cluster 95\% intervals. Context gain is the validated exploratory metric. Raw conditional likelihood is secondary. These automated results do not establish perceived naturalness.}\label{fig:v4-coherence}\end{figure}']
 supp.append(r'\input{supplementary_tables/v4_coherence_models.tex}')
 for arm in ['ordinary','shuffled']:
  d=summary[arm+'_donor_support'];supp.append(f"The {arm} arm uses {d['payloads']} donor payloads across {d['source_identities']} source identities. Maximum donor-payload reuse is {d['max_payload_reuse']} and maximum source-identity reuse is {d['max_source_reuse']}.")
 for r in summary['donor_sensitivity']:
  if r['metric']=='context_gain':supp.append(f"For {r['contrast']}, deleting each reference donor payload in turn gives point estimates from {r['minimum_leave_one_donor_out_effect']:.4f} to {r['maximum_leave_one_donor_out_effect']:.4f}. This is a donor-influence diagnostic, not a population interval incorporating two-way donor sampling.")
 supp.append('All scored windows have eight generator tokens on the left and eight to sixteen on the right. These short contexts limit interpretation beyond local transitions.')
 supp.append(f"Standalone tail tokenization differed from the retained joint-tail path in {summary['standalone_right_tokenization_differs_units']} units. The conditional and reference passes nevertheless scored the same IDs and denominator for each comparison. All exact source mappings and per-model effects remain in the joined tables.")
 supp.append(r'\subsection*{Predeclared illustrative boundaries}')
 supp.append('The weak, typical and strong examples use the minimum, upper-middle and maximum actual context gains, with identity-hash tie breaks. The marker shows the exact boundary. Literal newline markers retain formatting.')
 for ex in examples:
  assert ex['left_source_id']==ex['right_source_id']==ex['boundary_id']
  supp += [r'\noindent\begin{minipage}{\linewidth}',r'\paragraph{'+escape(ex['label'].capitalize())+r' context gain.}',f"The score is {ex['context_gain']:.4f} nats per tail token. Both windows use the following source.",
   r'\begin{flushleft}\small Source \code{'+ex['boundary_id']+r'}\\ Evaluator \code{'+ex['evaluator_model_id']+r'}.\end{flushleft}',
   f"Left source bytes are [{ex['left_source_byte_start']}, {ex['left_source_byte_stop']}). Right source bytes are [{ex['right_source_byte_start']}, {ex['right_source_byte_stop']}).",
   r'\begin{quote}\small '+escape(ex['left'].replace('\n',r'\n'))+r'\textbf{ [BOUNDARY] }'+escape(ex['right'].replace('\n',r'\n'))+r'\end{quote}',r'\end{minipage}\par\medskip']
 supp.append(r'The highest context-gain example contains the word-internal join \code{over[BOUNDARY]much} and retains malformed wording. The labels refer only to the predeclared metric ordering. They are not ratings of grammaticality or meaning.')
 supplement='\n\n'.join(supp)
 atomic_json(OUT/'analysis/coherence_study/manuscript_claims.json',{'boundaries':summary['boundaries'],'scoring_units':summary['scoring_units'],'trials':summary['trials'],'payloads':summary['payload_clusters'],'primary_effects':effects,'source_summary_sha256':file_hash(OUT/'analysis/coherence_study/summary.json')})
main += [r'The 25 decomposed transport arms across five identity-selected covers required 26 missing segment replays and 53 context-capacity checks of reused historical inputs. All reused ranks and error outcomes agreed at the new allocation capacity. These 79 distinct segment inputs remain within the same 25 logical arms. Prefix-only insertion recovered none of the five covers. Declared wrapper removal recovered one originally failing Mistral case but did not supply a general repair. The original 0/144 composite-Markdown endpoint remains unchanged. Byte, span, token and rank evidence is retained in Supplementary Note S18.',
 r'The six strict-gate failures all use ASCII B16 without a token filter. Five are Qwen trials and one is Mistral. They consume 59 to 116 of 128 requested ranks, or 54 of 64, at their original budgets. Longest below-threshold runs range from 35 to 267 positions. Matched ungated and moderate runs complete, while successful strict runs can also contain extended low-entropy stretches. The aligned text windows and progress evidence describe associations without identifying a causal linguistic trigger (Supplementary Note S16).',
 r'The quantization audit separates 56,321 changed observed ranks at 122,220 encoded positions from 13,207 at 122,220 ordinary positions. A distinct offline Q4-to-Q8 fixed-message endpoint recovers zero of 960 original payloads. Of these trials, 766 contain invalid bounded ranks and 194 decode valid but incorrect bytes. This uses saved Q8 ranks on the historical Q4 token path and establishes no reverse-direction result. For identical histories and allowed sets, a logit gap greater than twice a uniform absolute perturbation bound preserves a pairwise ordering. This sufficient condition does not estimate the actual quantization error. Independent free-generation divergence introduces different histories, whereas observed-token replay continues to feed received cover tokens, not recovered payload symbols.']
(paper/'v4_stage2_results.tex').write_text('\n\n'.join(main)+'\n');(paper/'v4_coherence_supplement_results.tex').write_text(supplement+'\n')
print('Rendered '+('pending preview' if args.preview else 'completed planned-sample results'))
