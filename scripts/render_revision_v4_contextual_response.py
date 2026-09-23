"""Current-study answers and statuses. Historical stage ledgers stay immutable."""
import copy
import re
from rankcloak.revision_v4_contextual_coherence import ROOT, OUT
from rankcloak.revision_v4_stage2_common import read_json, atomic_json
from scripts.prepare_revision_v4_stage1 import escape


def number(result, scale=100, digits=1):
    return f"{scale*result['estimate']:.{digits}f} [{scale*result['lower']:.{digits}f}, {scale*result['upper']:.{digits}f}]"


def main():
    summary=read_json(OUT/'analysis/summary.json')
    if summary['status']!='complete':raise ValueError('Incomplete planned sample')
    overall=next(r for r in summary['summaries'] if r['group']=='panel')
    rows=copy.deepcopy(read_json(ROOT/'results/revision_v4/stage4/response_status.json')['response_sets'])
    by={r['id']:r for r in rows}
    outcome=(f"Among {overall['complete_pairs']} complete message pairs in 280 trials, the equally weighted panel accepts {number(overall['encoded'])}% of encoded messages and {number(overall['ordinary'])}% of matched ordinary messages. "
        f"The paired difference is {number(overall['difference'])} percentage points. Brackets give 95% class-stratified payload-cluster intervals. "
        f"Mean logical connectedness is {overall['encoded_connectedness']['estimate']:.2f}/5 for encoded text and {overall['ordinary_connectedness']['estimate']:.2f}/5 for ordinary text. "
        f"Binary judge disagreement is {100*overall['encoded_disagreement']['estimate']:.1f}% and {100*overall['ordinary_disagreement']['estimate']:.1f}%, respectively.")
    paths=['configs/revision_v4/coherence_replacement.json','results/revision_v4/coherence_replacement/plans/final_sample/freeze.json',
           'results/revision_v4/coherence_replacement/analysis/calibration.json','results/revision_v4/coherence_replacement/analysis/summary.json',
           'results/revision_v4/coherence_replacement/analysis/disruption_quote_alignment.jsonl']
    by['R1.4'].update(
        direct_answer='We now make intact-message contextual acceptability the primary automated coherence evaluation. The question is whether the meaning can be followed in the authentic prompt context while allowing minor awkwardness. Disruption scores 0 and 1 count as acceptable, whereas 2 and 3 indicate a material breakdown. Two blinded local model judges evaluate each complete message without seeing its hidden boundary. This supplies automated evidence about contextual intelligibility and logical connection. It does not measure human perception or indistinguishability.',
        evidence_change='This follow-up was designed after the fragment results, then frozen before its own final scoring. A seeded selection retains 48 payloads, 12 per hexadecimal class, all three generators and both schedules, giving 288 trials and 576 complete messages. The selected payloads lie outside the earlier 92-payload scored cohort. Eighty-five messages excluded by the old window rules are retained. Each cover is paired with ordinary text from its generator and authentic prompt under the same filter, greedy continuation, assigned 264-token cap and heuristic stop. The 54 distinct controls are reused explicitly. Each item is independently judged by the other two model families. All three judges accept all six clear and six minor-imperfection calibration cases. Llama and Mistral reject five of six explicit disruptions and Qwen rejects six. No rubric amendment followed. '+outcome+f" The analysis retains full four-level distributions, subgroup results, exact complete examples and disruption quotations. There are {summary['invalid_unique_responses']} invalid final unique responses and {summary['retries']} retries. Nineteen final responses are truncated and one has malformed JSON. Eighteen encoded and 25 reused ordinary score slots are unavailable, leaving 39 incomplete message pairs. Extreme assignment of missing scores gives all-slot acceptance bounds of 88.9 to 90.5% for encoded and 94.8 to 97.0% for ordinary text, with a paired difference from -8.1 to -4.3 percentage points. These are sensitivity bounds rather than confidence intervals. The Qwen-source complete-pair comparison contains 47 payloads. The Mistral-source difference interval includes zero, without establishing equivalence. Messages are averaged within trial and payload. Intervals condition on the fixed control set and judge assignments are not fully crossed. This delivered-condition contrast is not an isolated causal boundary effect. The earlier semantic pilot intervals, 0.0078 [-0.0416, 0.0572] and -0.0106 [-0.0728, 0.0519], remain in Supplementary Note S17 with the limited local likelihood results. Their poor control discrimination motivated a different measurement target; it does not establish that every message is incoherent or prove the earlier test unfair. The new automated evidence leaves direct reader perception unmeasured. Its adequacy remains for the editor to assess.",
        evidence_paths=paths,
        location_text='Main Methods, Intact-message contextual acceptability. Main Results, Contextual acceptability of complete messages, with the primary table and figure. Supplementary Note S20 gives the full rubric, calibration, score distributions, judge and schedule strata, lengths and exact examples. Note S17 retains the earlier assays.',
        manuscript_anchors=['sec:v4-contextual-method','sec:v4-contextual-results','tab:v4-contextual','fig:v4-contextual','si:v4-contextual','si:v4-coherence'],
        empirical_coverage='automated_contextual_evidence_supplied_human_perception_unmeasured')
    by['editor_technical'].update(
        direct_answer='The revision retains the theoretical and implementation analyses and limits the contribution to artifact-specific surface-form concealment and configured inversion. A new intact-message study now supplies the primary automated contextual evidence. The distinction between completed written responses, automated evidence and unmeasured human perception remains explicit.',
        evidence_change='The new study applies the same lenient, fixed rubric to complete encoded messages and matched greedy ordinary controls. It uses authentic prompts, blinded cross-family judges, a separate constructed calibration set and a frozen 48-payload sample. '+outcome+' The previous fragment assays are retained as supplementary methodological history, with their failed semantic control checks and limited local likelihood findings. Historical recovery, transport, entropy, quantization and detector results are unchanged. No reader validation, confidentiality or indistinguishability is claimed. The new study requires updated archive coverage before submission.',
        evidence_paths=paths,
        location_text='Abstract, contextual Methods and Results, Discussion and limitations, Supplementary Notes S17 and S20, and the answer to Reviewer 1 comment 4.',
        manuscript_anchors=['sec:v4-contextual-method','sec:v4-contextual-results','sec:limitations','si:v4-contextual'],
        empirical_coverage='automated_contextual_evidence_supplied_human_perception_unmeasured')
    by['editor_accuracy']['evidence_change']+=' The new contextual follow-up uses matched greedy controls and reports automated rubric acceptance separately from human perception. All earlier fragment outcomes remain in the supplement.'
    by['editor_accuracy']['evidence_paths']+=paths
    by['editor_accuracy']['location_text']='Abstract, main Results under Contextual acceptability of complete messages, Discussion, and Supplementary Notes S16 through S20.'
    by['editor_accuracy']['manuscript_anchors']+=['si:v4-contextual']
    by['general']['evidence_change']=by['general']['evidence_change'].replace('The unsuccessful semantic pilots remain visible.', 'A new intact-message contextual evaluation supplies automated evidence under an explicit lenient rubric. The unsuccessful earlier semantic pilots remain in Supplementary Note S17.')
    by['general']['manuscript_anchors']+=['si:v4-contextual']
    by['general']['location_text']='Abstract, main Discussion under Configuration sensitivity, detection and a proposed fallback, limitations, and Supplementary Notes S17 through S20.'
    by['archive'].update(
        direct_answer='The published version 2.0.0, DOI 10.5281/zenodo.22555497, covers V3. It does not cover this V4 revision. The current code and retained data are in the repository. We do not represent the earlier local candidate as including the new contextual study.',
        evidence_change='The earlier verified local archive remains unchanged and predates this follow-up. The revised availability statement explicitly identifies the new evidence as outside that candidate. An updated committed snapshot must be assembled, verified and published as a new version under concept DOI 10.5281/zenodo.21987449 before its actual verified version DOI can be placed in the submission documents. This is an outstanding deposit action, not a completed publication.',
        evidence_paths=['results/revision_v4/provenance/archive_v2_coverage.json','revision_docs/REVISION_V4_STAGE3_REPORT.md','paperV4/ACTIONS_BEFORE_SUBMISSION.md'],
        external_action='prepare_verify_publish_current_scientific_snapshot_then_update_doi',
        empirical_coverage='not_applicable')
    locations={}
    for stem in ['main4','supplementary4']:
        aux=ROOT/'paperV4/scientific_reports'/(stem+'.aux')
        for key,number_label,page in re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{(\d+)\}',aux.read_text()):
            locations[key]={'document':stem,'page':int(page),'label':number_label}
    response=ROOT/'paperV4/response/response_to_reviewers_v4.tex';text=response.read_text()
    start=text.index('\\begin{center}',text.index('\\maketitle'));end=text.index('\\section*{Editor letter}',start)
    text=text[:start]+r'''\begin{center}\bfseries V4 AUTHOR-REVIEW CANDIDATE\end{center}
This response concerns submission 0170c7b2-1065-4344-a5d5-b839b51ac22c, \emph{RankCloak Conceals the Surface Form of Synthetic Cryptographic Artifacts in Language Model Generated Text}.
The original review wording is quoted in full. A new intact-message study supplies automated contextual evidence for Reviewer 1 comment 4. Human perception remains unmeasured. The current scientific evidence still requires public archive coverage. Page references identify the accompanying coherence-replacement author-review PDFs, with stable section, figure and table names for other renderings.

'''+text[end:]
    headings={'editor_accuracy':'Accuracy and scope','archive':'Code deposit','editor_technical':'Technical assessment','general':'General assessment','wording':'Preservation of the review wording'}
    for row in rows:
        row.update(written_response='complete',editor_judgment='unknown',public_deposit='current_followup_not_covered')
        if row['id'] in ['R1.4','editor_technical','editor_accuracy','general','archive']:
            row['author_decision']='current_followup_for_author_review'
        if row['id']!='archive':row['external_action']='author_review_then_editor_judgment' if row['author_decision']=='current_followup_for_author_review' else 'editor_judgment_after_author_submission'
        missing=set(row['manuscript_anchors'])-set(locations)
        if missing:raise ValueError('Missing compiled locations '+str(missing))
        row['compiled_locations']={a:locations[a] for a in row['manuscript_anchors']}
        parts=[]
        for document,label in [('main4','main4.pdf'),('supplementary4','supplementary4.pdf')]:
            pages=sorted({v['page'] for v in row['compiled_locations'].values() if v['document']==document})
            if pages:parts.append(label+' pages '+', '.join(map(str,pages)))
        location=row['location_text']+(' Coherence-replacement author-review edition, '+ ' and '.join(parts)+'.' if parts else '')
        row['rendered_location_text']=location
        body=('\\textbf{'+headings[row['id']]+'.}\n\n') if row['id'] in headings else ''
        for title,content in [('Answer',row['direct_answer']),('Change and evidence',row['evidence_change']),('Location',location)]:
            body+='\\textbf{'+title+'.} '+escape(content)+'\n\n'
        pattern=r'(% BEGIN ANSWER '+re.escape(row['id'])+r'\n).*?(% END ANSWER '+re.escape(row['id'])+r')'
        text,n=re.subn(pattern,lambda m:m[1]+body+m[2],text,flags=re.S)
        if n!=1:raise ValueError('Missing response '+row['id'])
    if r'\usepackage{needspace}' not in text:
        text=text.replace(r'\usepackage{parskip}',r'\usepackage{parskip}'+chr(10)+r'\usepackage{needspace}')
    text=re.sub(r'(?:\\Needspace\{7\\baselineskip\}\n)?(\\section\*\{Reviewer 1 comment [0-9]+\})',lambda m:r'\Needspace{7\baselineskip}'+chr(10)+m[1],text)
    response.write_text(text)
    atomic_json(OUT/'response_status.json',{'schema':'rankcloak.response.contextual.v1','response_sets':rows,'written_complete':16,
        'empirical_R1_4':'automated_contextual_evidence_supplied_human_perception_unmeasured','public_current_study_coverage':False,
        'historical_archive_unchanged':True,'journal_submission_performed':False})
    atomic_json(OUT/'validation/locations.json',locations)
    matrix=['# V4 response and evidence status','','All sixteen written responses are complete for author review. R1.4 now includes a completed automated intact-message contextual evaluation under a lenient explicit rubric. Human perception remains unmeasured, and the editor determines whether the evidence addresses the request. The earlier failed fragment assays remain in Supplementary Note S17.','','The current follow-up is outside the earlier local archive candidate and the published V3 deposit. Archive preparation and publication are separate remaining actions. The thirteen exact review blocks and original requests are unchanged. The current machine ledger is `results/revision_v4/coherence_replacement/response_status.json`.','','| Point | Written response | Evidence coverage | External action | Locations |','| --- | --- | --- | --- | --- |']
    for row in rows:matrix.append('| '+row['id']+' | complete | '+row['empirical_coverage'].replace('_',' ')+' | '+row['external_action'].replace('_',' ')+' | '+row['rendered_location_text']+' |')
    (ROOT/'paperV4/REVIEW_RESPONSE_MATRIX.md').write_text('\n'.join(matrix)+'\n')
    print('Rendered 16 answers with current compiled locations')


if __name__=='__main__':main()
