"""Render the new primary scientific narrative from the completed retained analysis."""
import re
from rankcloak.revision_v4_contextual_coherence import ROOT,OUT
from rankcloak.revision_v4_stage2_common import read_json,atomic_json
from scripts.render_revision_v4_contextual_results import interval


def main():
    result=read_json(OUT/'analysis/summary.json')
    if result['status']!='complete':raise ValueError('The planned study must finish before manuscript integration')
    panel=next(r for r in result['summaries'] if r['group']=='panel')
    audit=read_json(OUT/'validation/score_audit.json')
    enc=interval(panel['encoded']);ordinary=interval(panel['ordinary']);difference=interval(panel['difference'])
    text=r'''\subsection*{Contextual acceptability of complete messages}\label{sec:v4-contextual-results}\label{sec:v4-results}
All three judges accepted the six clear and six minor-imperfection calibration messages. Llama and Mistral rejected five of six constructed disruptions and Qwen rejected six. The two missed contradictions remain in the calibration record. This check supports basic rubric interpretation, not human validity. No rubric amendment was made.

'''
    text+=f"The full frozen sample contains 576 encoded messages from 288 trials and 48 payloads, including 85 messages unavailable under the old window rules. Using {panel['complete_pairs']} complete pairs, the equally weighted panel accepts {enc}\\% of encoded messages and {ordinary}\\% of matched greedy ordinary messages. The paired difference is {difference} percentage points. Brackets denote 95\\% payload-cluster intervals conditional on paired availability. The Mistral-source difference interval includes zero, which does not establish equivalence. Table~\\ref{{tab:v4-contextual}} and Fig.~\\ref{{fig:v4-contextual}} show the overall and generator-specific estimates.\n\n"
    text+=r'\input{supplementary_tables/v4_contextual.tex}'+'\n\n'
    text+=r'''\begin{figure}[H]
\centering\includegraphics[width=\textwidth]{figures/contextual_acceptability.pdf}
\caption{\textbf{Automated acceptability of intact messages.} The two-judge panel applies the same lenient rubric to RankCloak and matched greedy ordinary output. Left, acceptance at disruption levels 0 or 1. Right, the paired difference in percentage points. Error bars are 95\% intervals from 2,000 class-stratified payload-cluster draws. Intervals condition on the fixed control set. Generator rows use different pairs of judges, so generator and judge effects are not fully crossed. These are automated judgments, not human acceptance or detection rates.}\label{fig:v4-contextual}
\end{figure}

'''
    text+=f"Mean logical connectedness is {interval(panel['encoded_connectedness'],1,2)}/5 for encoded messages and {interval(panel['ordinary_connectedness'],1,2)}/5 for ordinary messages. Binary judge disagreement is {100*panel['encoded_disagreement']['estimate']:.1f}\\% and {100*panel['ordinary_disagreement']['estimate']:.1f}\\%, respectively. The panel distribution over levels 0, 1, 2 and 3 is 11.5, 78.8, 8.5 and 1.2\\% for encoded messages. Full distributions for both arms, individual judges, both schedules and realized lengths appear in Supplementary Note S20. "
    text+=f"The {result['unique_judge_requests']:,} unique requests supply {result['logical_judge_units']:,} mapped assignments, with {result['invalid_unique_responses']} invalid final responses and {result['retries']} retries. "
    if result['invalid_unique_responses']:
        low=result['missing_sensitivity']['encoded_lower_missing']['estimate']*100;high=result['missing_sensitivity']['encoded_upper_missing']['estimate']*100
        text+=f"The primary panel uses {panel['complete_pairs']} complete pairs in 280 trials, retaining all 48 payload clusters. Assigning every missing encoded score as disrupted or acceptable gives an all-slot acceptance range of {low:.1f} to {high:.1f}\\%. Corresponding ordinary bounds are 94.8 to 97.0\\%, with a paired difference from -8.1 to -4.3 percentage points. These sensitivity bounds are not confidence intervals. "
    text+='No missing score is treated as acceptance.\n\n'
    text+=f"Across all valid encoded judgments, {audit['disrupted_encoded_judge_slots']} are disrupted. Of these, {audit['disrupted_slots_with_literal_quote']} contain a literal passage and {audit['disrupted_slots_with_boundary_crossing_quote']} quote across the recorded boundary. This post-scoring mapping is descriptive. Whole-message quotations may cross a boundary incidentally. It neither attributes a defect causally to the boundary nor treats an absent quotation as absence of a defect. Identity-selected full-message examples include agreement on acceptance, disagreement and agreement on disruption where available. The earlier fragment pilots and limited fixed-path likelihood findings remain in Supplementary Note S17.\n"
    paper=ROOT/'paperV4/scientific_reports'
    (paper/'v4_contextual_results.tex').write_text(text)
    old=(paper/'v4_stage2_results.tex').read_text();mark=old.index(r'\subsection*{Mechanisms of transmission failure}')
    (paper/'v4_stage2_results.tex').write_text('\\input{v4_contextual_results.tex}\n\n'+old[mark:])
    main=(paper/'main4.tex').read_text()
    abstract=("Cryptographic artifacts such as hashes, nonces and ciphertexts are conspicuous in their usual textual forms. RankCloak represents synthetic artifacts through language model token choices that conceal their original syntax. We evaluated 480 deterministic artifacts from eight classes with three model families and 18 English prompts. All 6,480 primary trials recovered the original bytes when token sequences and configurations were preserved. Visible-text retokenization recovered 88 of 144 covers. The tested blockquote and trimming operation, paraphrasing and cross-model decoding failed. Strong detector performance after deduplication, including under generating-model trace access, limits stealth. Entropy gating reduced some detection signals but lowered serialized-payload rates and left six strict-gate payloads incomplete. "
        f"In a contextual follow-up with blinded model judges, {panel['complete_pairs']} of 576 message pairs had complete ratings. Under a rubric allowing minor awkwardness, acceptance averaged {100*panel['encoded']['estimate']:.1f}\\% for RankCloak and {100*panel['ordinary']['estimate']:.1f}\\% for matched greedy ordinary output. Human perception was not measured. "
        "None of 960 fixed Q4-generated payloads recovered using saved Q8 ranks for one Qwen revision. These results demonstrate artifact-specific surface-form concealment and configured inversion, with measured acceptability, transmission and detectability limitations. An observer with the full receiver configuration may decode, so surface concealment does not establish confidentiality.")
    main=re.sub(r'(\\begin\{abstract\}\n).*?(\n\\end\{abstract\})',lambda m:m[1]+abstract+m[2],main,flags=re.S)
    previous='Cover-quality evidence remained protocol-dependent.'
    replacement=(f"The intact-message follow-up measures a practical but explicitly automated endpoint. Under the lenient rubric, the panel accepts {100*panel['encoded']['estimate']:.1f}\\% of encoded messages compared with {100*panel['ordinary']['estimate']:.1f}\\% of matched greedy controls. This quantifies acceptance despite minor awkwardness while retaining material disruptions and judge disagreements. It is neither a human reading rate nor evidence of indistinguishability. The comparison concerns complete delivered conditions, and length or content differences can contribute to the observed contrast.\n\nCover-quality evidence also remains protocol-dependent.")
    if previous in main:main=main.replace(previous,replacement)
    else:
        main=re.sub(r'The intact-message follow-up measures.*?Cover-quality evidence also remains protocol-dependent\.',lambda m:replacement,main,flags=re.S)
    (paper/'main4.tex').write_text(main)
    cover=ROOT/'paperV4/cover_letter/cover_letter_v4.tex';letter=cover.read_text()
    start=letter.index('The adverse findings remain central.') if 'The adverse findings remain central.' in letter else letter.index('The primary coherence evaluation now');end=letter.index('\\closing',start)
    letter=letter[:start]+f"The primary coherence evaluation now assesses complete messages in their authentic prompt context under a fixed rubric that allows minor awkwardness. Two blinded local model judges evaluate each item independently. The frozen sample contains 576 messages from 48 payloads, paired with ordinary output using matched greedy continuation. Complete ratings are available for 537 pairs. Panel acceptance is {100*panel['encoded']['estimate']:.1f}\\% for encoded messages and {100*panel['ordinary']['estimate']:.1f}\\% for ordinary controls, with a paired difference of {difference} percentage points (95\\% payload-cluster interval). The response reports calibration mistakes, score distributions, disagreement, missing-score bounds and exact unfavorable examples. Direct human perception remains unmeasured. The editor can assess whether this automated evidence meets Reviewer 1 comment 4.\n\n"+r'''The earlier fragment pilots and limited local-likelihood results remain in the supplement. The new study was designed after those results and frozen before its own scoring. Historical adverse findings are unchanged, including failed transmission conditions and zero recovery among 960 fixed Q4 messages decoded from saved Q8 ranks. RankCloak demonstrates surface-form concealment and configured inversion, without a confidentiality or indistinguishability guarantee.

Published version 2.0.0, DOI \url{https://doi.org/10.5281/zenodo.22555497}, covers V3. The earlier verified local candidate predates this follow-up. Updated archive coverage and a verified public version DOI remain required before resubmission. This letter is prepared for author review and records no new deposit, resubmission or acceptance.

'''+letter[end:]
    cover.write_text(letter)
    claims=[{'claim_id':key,'source':'analysis/summary.json','selected_population':{'payloads':48,'trials':288,'message_pairs':576},
        'observed_population':{'payloads':48,'trials':280,'complete_message_pairs':537},
        'population':'Complete panel pairs from the selected outside-cohort payload stratum',
        'independent_check':'validation/score_audit.json',
        'estimand':key,'estimate':panel[key]['estimate'],'lower':panel[key]['lower'],'upper':panel[key]['upper'],
        'units':'proportion' if key in ['encoded','ordinary','difference'] else 'score from 1 to 5',
        'uncertainty':'2000 paired class-stratified recipient-payload bootstrap draws, fixed controls, complete-pair availability',
        'evidence_class':'new automated contextual rubric study','limitations':'Not human perception or indistinguishability. Judge and generator assignments not fully crossed.'}
        for key in ['encoded','ordinary','difference','encoded_connectedness','ordinary_connectedness']]
    atomic_json(OUT/'manuscript/claim_evidence.json',{'claims':claims,'historical_claim_audit_retained':'results/revision_v4/stage3/claim_evidence.json',
        'historical_result_scope':'Other recovery, transformation, entropy, quantization and detectability claims remain unchanged.'})
    print('Integrated completed contextual results into manuscript and cover letter')


if __name__=='__main__':main()
