"""Render finished V4 answers while retaining exact review blocks and separate statuses."""
import json,re
from pathlib import Path
from scripts.prepare_revision_v4_stage1 import escape

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision_v4/stage3'


def main():
    prior=json.loads((ROOT/'results/revision_v4/stage2/response_evidence.json').read_text())
    rows=prior['response_sets']
    by={r['id']:r for r in rows}
    by['archive'].update(
        direct_answer='The underlying code is deposited as version 2.0.0, DOI 10.5281/zenodo.22555497. That published archive covers V3 and does not cover V4. This author-review candidate retains the correct published DOI and explicitly identifies the remaining coverage requirement.',
        evidence_change='The revision package supplies a commit-based V4 assembly specification, a matching release description and reproducibility instructions. It includes the V4 code, retained evidence, unsuccessful pilots and author-review documents. The local candidate and its independent verification receipt accompany the handoff. Local verification does not satisfy the request for a published DOI archive. A new version under concept DOI 10.5281/zenodo.21987449 must be published and its files verified before its real version DOI is inserted into the final submission documents.',
        location_text='Main Data Availability and Code Availability. The accompanying author checklist states the deposit and final-document sequence.',
        evidence_paths=['results/revision_v4/provenance/archive_v2_coverage.json','release/v4/assembly_policy.json','release/v4/REPRODUCE.md','release/v4/RELEASE_DESCRIPTION.md','paperV4/ACTIONS_BEFORE_SUBMISSION.md'])
    by['editor_technical'].update(
        direct_answer='We have expanded the theoretical and implementation analyses and narrowed the conclusions to artifact-specific surface-form concealment and configured inversion. The requested semantic and perceptual evidence remains incomplete. We report that limitation rather than treating the written response as full empirical satisfaction.',
        evidence_change='Both control-only pilots failed the prospective semantic control-separation requirement. Conditional context gain passed its specified control check, and the frozen exploratory study supports only local fixed-token-path compatibility. Sampled ordinary controls and greedy RankCloak continuations prevent a causal boundary interpretation. The revised text retains exclusions, generator-specific nulls and unfavorable examples. No reader validation is claimed. R1.4 is answered substantively below, with the adequacy of this narrower evidence left to the author and editor. Public V4 archive coverage also remains outstanding.',
        location_text='Abstract, boundary Methods and Results, Discussion and limitations, Supplementary Notes S16 through S19, and response to Reviewer 1 comment 4.')
    by['R1.4'].update(
        direct_answer='We attempted the requested complementary automated evaluations prospectively. Semantic relatedness was not established because the semantic metric failed the specified control-separation requirement in both control-only pilots. This does not imply that every RankCloak transition is incoherent. Conditional context gain passed its control-separation check, but neither that result nor a positive RankCloak likelihood contrast proves that transitions appear natural or logically connected to readers. We have removed that implication from the paper.',
        evidence_change='Each pilot selected 36 identities without inspecting RankCloak scores. Initial ordinary-minus-shuffled semantic cosine separation was 0.0078 with 95% interval [-0.0416, 0.0572]. The sole predeclared amendment shortened the right window to 16 tokens and used controls disjoint in recipient and donor payloads. It gave -0.0106 [-0.0728, 0.0519]. Context-gain differences were 2.2200 [1.7784, 2.6777] and 2.0612 [1.7378, 2.4109] nats per evaluator tail token. The frozen exploratory study selected 92 payloads and 3,174 structural boundaries. Joint eligibility left 2,750 boundaries in 550 scored trials. Actual-minus-ordinary context gain was 0.1575 [0.1098, 0.2066], and actual-minus-shuffled gain was 2.3265 [2.2705, 2.3912]. These recipient-payload bootstrap intervals condition on the fixed control corpus. Mistral excludes 341 of 1,058 selected structural boundaries. Secondary raw conditional-likelihood intervals include zero for Mistral and Qwen. Ordinary continuations are sampled while RankCloak tails are greedy, so the contrast does not isolate a causal payload-boundary effect. The strongest score-selected example retains its internal word boundary and malformed wording. We now use the operational term non-payload greedy continuations and explicitly state that semantic coherence and reader-perceived naturalness were not established. This completes our explanation for this candidate, with partial empirical coverage. The editor may judge that additional evidence is required.',
        location_text='Main Boundary instrumentation and prospective control validation, Local boundary compatibility and metric limitations, Discussion and Responsible use and limitations. Supplementary Note S17 gives both pilots, exclusions, donor support, subgroup effects and exact examples.')
    by['R1.1']['evidence_change']=by['R1.1']['evidence_change'].replace('Prefix-only insertion recovers none of the five.','Prefix-only insertion recovers none of the five. All 27 selected segment instances are rejected at a prohibited first forced token before a rank sequence is returned. These endpoints do not measure downstream rank amplification.')
    by['R1.1']['location_text']='Main Methods under Decoder compatibility and bounded transport diagnostics, Results under Mechanisms of transmission failure, and Supplementary Note S18 with its illustrative recovery table.'
    by['R1.6']['location_text']='Main Results under Strict-gate failure contexts and Supplementary Notes S13 and S16, including the six-case table, trajectory figure and exact local windows.'
    by['R1.9']['direct_answer']='The main Methods now give the complete deterministic token filter specification, including every literal blocked substring and all other exclusion criteria. The historical filter is unchanged.'
    by['R1.10']['location_text']='Main Results under Fixed-message cross-quantization decoding and Supplementary Notes S14, S18 under Directional cross-quantization endpoint, and S19.'
    by['R1.5']['location_text']='Main Neural steganalysis, statistical analysis, and computational overhead, Configuration sensitivity, detection and a proposed fallback, and Supplementary Notes S12 and S19.'
    by['wording']['evidence_change']='All thirteen original quoted blocks are retained, including the editor letter, Reviewer 2 acknowledgment, both Reviewer 1 introductory paragraphs and the instruction to preserve wording. Answers remain separate from the original comments.'
    # Stable response IDs remain in comments and this machine ledger, not reader headings.
    headings={'editor_accuracy':'Accuracy and scope','archive':'Code deposit','editor_technical':'Technical assessment','general':'General assessment','wording':'Preservation of the review wording'}
    tex=ROOT/'paperV4/response/response_to_reviewers_v4.tex'
    text=tex.read_text()
    text=text.replace('\\title{V4 response working draft}','\\title{Response to the editor and reviewers}')
    start=text.index('\\begin{center}',text.index('\\maketitle'))
    end=text.index('\\section*{Editor letter}',start)
    text=text[:start]+r'''\begin{center}\bfseries V4 AUTHOR-REVIEW CANDIDATE\end{center}
This response concerns submission 0170c7b2-1065-4344-a5d5-b839b51ac22c, \emph{RankCloak Conceals the Surface Form of Synthetic Cryptographic Artifacts in Language Model Generated Text}.
The original review wording is quoted in full. The answers distinguish completed analyses from limitations of the evidence. In particular, the semantic and perceptual part of Reviewer 1 comment 4 is only partly supported, and the public archive does not yet cover V4.

'''+text[end:]
    locations={}
    for stem in ['main4','supplementary4']:
        aux=ROOT/'paperV4/scientific_reports'/(stem+'.aux')
        if aux.exists():
            for key,page in re.findall(r'\\newlabel\{([^}]+)\}\{\{[^}]*\}\{(\d+)\}',aux.read_text()):
                locations[key]={'document':stem,'page':int(page)}
    for r in rows:
        r.pop('status',None);r.pop('pending_work',None);r.pop('old_label',None)
        r.update(written_response='complete',empirical_coverage='supported_requested_analysis',external_action='author_review')
        if r['id'] in ['R1.4','editor_technical']:
            r.update(empirical_coverage='partial',external_action='author_and_editor_judgment_required')
        if r['id']=='archive':r.update(empirical_coverage='not_applicable',external_action='publish_verify_v4_deposit_then_update_documents')
        if r['id'] in ['reviewer2','wording']:r['empirical_coverage']='not_applicable'
        if r['id'] in ['R1.2','R1.3','R1.7']:r['empirical_coverage']='theoretical_or_compatibility_answer_no_new_robustness_claim'
        r['location_text']=r['location_text'].replace('Boundary validation and targeted diagnostics','Local boundary compatibility and metric limitations').replace('targeted Results','the relevant Results subsections').replace('Main targeted Results','Main Results')
        if r['id']=='R1.1':r['manuscript_anchors']=['sec:v4-compatibility','sec:v4-transport-results','si:v4-transport','tab:v4-transport']
        if r['id']=='R1.6':r['manuscript_anchors']=['sec:v4-entropy-results','si:s13','si:v4-stage1','tab:v4-entropy-runs']
        if r['id']=='R1.10':r['manuscript_anchors']=['sec:v4-quant-results','si:s14','si:v4-fixed-quant','si:v4-theory']
        r['compiled_locations']={a:locations[a] for a in r['manuscript_anchors'] if a in locations}
        pages=default_pages(r['compiled_locations'])
        location=r['location_text']+((' Compiled locations '+pages+'.') if pages else '')
        r['rendered_location_text']=location
        body=(('\\textbf{'+headings[r['id']]+'.}\n\n') if r['id'] in headings else '')
        body+='\\textbf{Answer.} '+escape(r['direct_answer'])+'\n\n\\textbf{Change and evidence.} '+escape(r['evidence_change'])+'\n\n\\textbf{Location.} '+escape(location)+'\n'
        pattern=r'(% BEGIN ANSWER '+re.escape(r['id'])+r'\n).*?(% END ANSWER '+re.escape(r['id'])+r')'
        text,n=re.subn(pattern,lambda m:m[1]+body+m[2],text,flags=re.S)
        if n!=1:raise ValueError('missing answer '+r['id'])
    tex.write_text(text)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'response_status.json').write_text(json.dumps({'schema':'rankcloak.response.stage3.v1','response_sets':rows,'written_complete':16,'empirical_R1_4':'partial','v4_public_deposit':False},indent=2)+'\n')
    (OUT/'validation/locations.json').write_text(json.dumps(locations,indent=2)+'\n')
    matrix=['# V4 response and evidence status','','All sixteen written responses are complete for author review. R1.4 has partial empirical coverage. Its written explanation does not establish semantic or perceptual validation. The editor may require additional evidence. The published DOI covers V3 only.','','The exact thirteen review blocks and original requests are unchanged. Machine IDs, evidence links and compiled manuscript locations are in `results/revision_v4/stage3/response_status.json`.','','| Point | Written response | Evidence coverage | External action | Locations |','| --- | --- | --- | --- | --- |']
    for r in rows:matrix.append('| '+r['id']+' | complete | '+r['empirical_coverage'].replace('_',' ')+' | '+r['external_action'].replace('_',' ')+' | '+r['rendered_location_text']+' |')
    (ROOT/'paperV4/REVIEW_RESPONSE_MATRIX.md').write_text('\n'.join(matrix)+'\n')


def default_pages(locations):
    parts=[]
    for document,label in [('main4','main pages'),('supplementary4','Supplementary Information pages')]:
        pages=sorted({r['page'] for r in locations.values() if r['document']==document})
        if pages:parts.append(label+' '+', '.join(map(str,pages)))
    return ' and '.join(parts)


if __name__=='__main__':main()
