"""Strict Stage 3 checks without mutating Stage 1 or Stage 2 validation records."""
import hashlib,json,re
from pathlib import Path
from scripts.validate_revision_v4_stage2 import verify_quotes,verify_file_manifest,verify_transport_reuse
from scripts.prepare_revision_v4_stage1 import escape

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision_v4/stage3'


def validate_status(ledger):
    rows=ledger['response_sets'];expected={'editor_accuracy','archive','editor_technical','reviewer2','general','wording'}|{f'R1.{i}' for i in range(1,11)}
    if len(rows)!=16 or {r['id'] for r in rows}!=expected:raise ValueError('sixteen distinct answers required')
    if any(r['written_response']!='complete' for r in rows):raise ValueError('unfinished written response')
    by={r['id']:r for r in rows}
    for key in ['R1.4','editor_technical']:
        if by[key]['empirical_coverage']!='partial' or by[key]['external_action']!='author_and_editor_judgment_required':raise ValueError('partial empirical request concealed')
    if ledger['v4_public_deposit'] is not False or by['archive']['external_action']!='publish_verify_v4_deposit_then_update_documents':raise ValueError('unpublished archive misrepresented')


def main():
    initial=json.loads((OUT/'provenance/initial_state.json').read_text())
    for row in initial['protected']:
        raw=(ROOT/row['path']).read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if blob!=row['git_blob']:raise ValueError('historical artifact modified '+row['path'])
    letter=(ROOT/'paperV4/response/requests.txt').read_text();quote_ledger=json.loads((ROOT/'paperV4/response/review_comments.json').read_text());response=(ROOT/'paperV4/response/response_to_reviewers_v4.tex').read_text()
    if hashlib.sha256(letter.encode()).hexdigest()!=quote_ledger['source_sha256']:raise ValueError('original letter hash')
    quotes=verify_quotes(letter,quote_ledger,response)
    ledger=json.loads((OUT/'response_status.json').read_text());validate_status(ledger)
    alltex='\n'.join(p.read_text() for p in (ROOT/'paperV4/scientific_reports').rglob('*.tex') if p.name!='main4_submission.tex')
    anchors=set(re.findall(r'\\label\{([^}]+)\}',alltex))
    for row in ledger['response_sets']:
        matches=re.findall(r'% BEGIN ANSWER '+re.escape(row['id'])+r'\n(.*?)% END ANSWER '+re.escape(row['id']),response,re.S)
        if len(matches)!=1:raise ValueError('answer marker')
        for key in ['direct_answer','evidence_change','rendered_location_text']:
            if escape(row[key]) not in matches[0]:raise ValueError('answer ledger drift '+row['id'])
        if not set(row['manuscript_anchors'])<=anchors:raise ValueError('nonexistent location')
        for name in row['evidence_paths']:
            if not (ROOT/name).exists():raise ValueError('missing answer evidence '+name)
    for folder in ['initial','initial_unscored_roundtrip_preflight','amendment1','pilot_initial','pilot_amendment1','coherence_study']:
        verify_file_manifest(ROOT/'results/revision_v4/stage2/plans'/folder)
    verify_file_manifest(ROOT/'results/revision_v4/stage2/transport')
    if verify_transport_reuse(ROOT/'results/revision_v4/stage2/transport')!=53:raise ValueError('transport reuse count')
    export=(ROOT/'results/revision_v4/manuscript_tables/filter_methods.tex').read_text().replace(r'\textbf{V4 working draft.} ','')
    if export!=(ROOT/'paperV4/scientific_reports/v4_filter_methods.tex').read_text():raise ValueError('exact filter export changed')
    audit=json.loads((OUT/'audit/audit_summary.json').read_text())
    if audit['status']!='passed' or audit['gpu_execution_seconds']!=0:raise ValueError('audit incomplete')
    for path,expected in json.loads((OUT/'audit/input_hashes.json').read_text()).items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:raise ValueError('audited input changed')
    for r in json.loads((OUT/'claim_evidence.json').read_text())['claims']:
        for source in r['source_artifacts']:
            if not (ROOT/source).exists():raise ValueError('missing claim source '+source)
        if not all(r[k] for k in ['population','quantity','units','uncertainty','evidence_class','limitations','manuscript_locations']):raise ValueError('incomplete claim record')
    from scripts.prepare_revision_v4_stage3_handoff import flatten
    maintext=flatten(ROOT/'paperV4/scientific_reports/main4.tex')
    if maintext.count('\\begin{figure}')+maintext.count('\\begin{table}')!=8:raise ValueError('main display item count')
    abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',maintext,re.S)[1]
    if len(abstract.split())>200:raise ValueError('abstract word count')
    for content in [maintext,flatten(ROOT/'paperV4/scientific_reports/supplementary4.tex'),response,(ROOT/'paperV4/cover_letter/cover_letter_v4.tex').read_text()]:
        if content.count('V4 AUTHOR-REVIEW CANDIDATE')!=1:raise ValueError('candidate label count')
        if any(x in content for x in ['INTERNAL PENDING WORK','INTERNAL WORKING DRAFT','Stage 2 evidence integrated','Direct answer pending','validated context-gain metric']):raise ValueError('stale development prose')
    # Check every reconstructed boundary effect printed in the main table.
    table=(ROOT/'paperV4/scientific_reports/supplementary_tables/v4_coherence.tex').read_text()
    for r in json.loads((OUT/'audit/reconstructed_effects.json').read_text()):
        if r['plan']=='coherence_study' and r['model']=='all' and r['metric']=='context_gain':
            if not all(f'{r[k]:.4f}' in table for k in ['effect','ci_low','ci_high']):raise ValueError('main table numerical mismatch')
    build=json.loads((OUT/'validation/latex/build_status.json').read_text())
    if len(build)!=4 or any(r['status']!='compiled' or r['unresolved_references'] or any('Overfull' in w for w in r['final_layout_warnings']) for r in build):raise ValueError('document build/layout failure')
    linkage=json.loads((OUT/'audit/linkage.json').read_text())
    if linkage['status']!='passed' or linkage['boundary_source_byte_windows_checked']!=16500 or linkage['entropy_source_text_windows_checked']!=28:raise ValueError('source linkage audit incomplete')
    for path,expected in linkage['input_hashes'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:raise ValueError('linkage input changed')
    review=json.loads((OUT/'validation/page_review.json').read_text())
    if review['status']!='reviewed':raise ValueError('visual review incomplete')
    counts=json.loads((OUT/'validation/document_audit.json').read_text())
    for d in counts['documents']:
        r=next(r for r in review['documents'] if r['path']==d['document'])
        if r['pages_reviewed']!=list(range(1,d['pages']+1)) or r['sha256']!=hashlib.sha256((ROOT/d['document']).read_bytes()).hexdigest():raise ValueError('reviewed PDF changed')
    upload=json.loads((OUT/'validation/upload_source/verification.json').read_text())
    if upload['status']!='passed' or upload['source_sha256']!=hashlib.sha256((ROOT/upload['source']).read_bytes()).hexdigest():raise ValueError('upload source changed after build')
    result={'status':'passed','protected_files_unchanged':len(initial['protected']),'exact_review_blocks':quotes,'written_responses':16,'R1_4_empirical_coverage':'partial','V4_published_archive_coverage':False,'independent_effects':audit['effects_reconstructed'],'independent_input_files':audit['input_files'],'main_figures_and_tables':8,'abstract_words':len(abstract.split()),'gpu_job_seconds':0}
    (OUT/'validation/stage3_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
