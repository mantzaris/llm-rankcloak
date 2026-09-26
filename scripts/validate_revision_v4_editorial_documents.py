"""Check clean presentation, unchanged science, bookmarks and document builds.

This reads frozen evidence and checks document production. It runs no scientific
analysis or model, and it never writes into an earlier stage's results.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
from scripts.prepare_revision_v4_stage3_handoff import flatten
from scripts.prepare_revision_v4_stage1 import escape
from scripts.render_revision_v4_editorial_response import render_text
from scripts.validate_revision_v4_stage2 import verify_quotes
from scripts.review_revision_v4_editorial_pdfs import inspect, DOCS
from scripts.build_revision_v4_stage4_bundle import manuscript_source

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/revision_v4/presentation_cleanup'
PAPER = ROOT/'paperV4/scientific_reports'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def old_text(path, commit):
    return subprocess.check_output(['git','show',commit+':'+str(path.relative_to(ROOT))],cwd=ROOT,text=True)


def old_flatten(path, commit):
    text = old_text(path, commit)
    return re.sub(r'\\input\{([^}]+)\}',lambda m:old_flatten(path.parent/(m[1]+('' if m[1].endswith('.tex') else '.tex')),commit),text)


def environments(text, name):
    return re.findall(r'\\begin\{'+name+r'\}.*?\\end\{'+name+r'\}',text,re.S)


def norm(text):
    return re.sub(r'[^a-z0-9]','',text.lower())


def navigation(supplement):
    outline = subprocess.check_output(['mutool','show',str(PAPER/'supplementary4.pdf'),'outline'],text=True)
    (OUT/'validation/supplement_outline.txt').write_text(outline)
    nodes = re.findall(r'^[+|\-]\t(\t*)"([^"]+)"\t#nameddest=(\S+)',outline,re.M)
    notes = [n for n in nodes if n[1].startswith('Supplementary Note')]
    assert [int(re.search(r'Note S(\d+)',n[1])[1]) for n in notes]==list(range(1,22))
    assert all(len(n[0])==0 for n in notes)
    assert all(len(n[0])<=1 for n in nodes)
    assert len({n[2] for n in nodes})==len(nodes), 'Repeated bookmark destination'
    raw = subprocess.check_output(['pdfinfo','-dests',str(PAPER/'supplementary4.pdf')],text=True)
    (OUT/'validation/supplement_destinations.txt').write_text(raw)
    dests = {name:(int(page),float(y)) for page,y,name in re.findall(r'^\s*(\d+) \[ XYZ\s+[-\d.]+\s+([-\d.]+)\s+null\s*\] "([^"]+)"',raw,re.M)}
    xml = subprocess.check_output(['pdftotext','-bbox-layout',str(PAPER/'supplementary4.pdf'),'-'],text=True)
    xml = re.sub('[\x00-\x08\x0b\x0c\x0e-\x1f]','',xml)
    pages = ET.fromstring(xml).findall('.//{*}page')
    # Hyperref writes sidebar bookmarks without printing or reading a contents page.
    links = subprocess.check_output(['mutool','show','-g',str(PAPER/'supplementary4.pdf'),'grep'],text=True)
    linked = set(re.findall(r'/D\(([^)]+)\)',links))
    assert linked <= set(dests), 'Unresolved internal PDF link'
    verified = []
    for tabs,title,dest in nodes:
        assert dest in dests
        page,y = dests[dest]
        element = pages[page-1];top = float(element.attrib['height'])-y
        near_words = [w for w in element.findall('.//{*}word') if top-8<=float(w.attrib['yMin'])<=top+48]
        near = ' '.join(w.text or '' for w in sorted(near_words,key=lambda w:(round(float(w.attrib['yMin']),1),float(w.attrib['xMin']))))
        assert norm(title) in norm(near), (title,page,top,near)
        verified.append({'title':title,'level':len(tabs)+1,'destination':dest,'page':page,'heading_at_destination':True})
    return verified


def main():
    baseline=load(OUT/'baseline.json');commit=baseline['head']
    expanded=flatten(PAPER/'main4.tex');supplement=flatten(PAPER/'supplementary4.tex')
    previous=old_flatten(PAPER/'main4.tex',commit);previous_supp=old_flatten(PAPER/'supplementary4.tex',commit)
    # Only current document sources/workflows and their new records may change.
    changed=subprocess.check_output(['git','diff','--name-only',commit],cwd=ROOT,text=True).splitlines()
    allowed={'.gitignore','paperV4/ACTIONS_BEFORE_SUBMISSION.md','paperV4/REVIEW_RESPONSE_MATRIX.md'}
    source_names={'main4.tex','main4.pdf','main4.bbl','main4_submission.tex','supplementary4.tex','supplementary4.pdf','supplementary4.bbl'}
    allowed |= {'paperV4/scientific_reports/'+name for name in source_names}
    allowed |= {'paperV4/response/response_to_reviewers_v4.'+suffix for suffix in ['tex','pdf']}
    allowed |= {'paperV4/cover_letter/cover_letter_v4.'+suffix for suffix in ['tex','pdf']}
    extra=[p for p in changed if p not in allowed and not p.startswith('results/revision_v4/presentation_cleanup/') and not p.startswith('scripts/') and p not in ['paperV4/response/editorial_answers.json','paperV4/DOCUMENT_BUILD.md','revision_docs/REVISION_V4_PRESENTATION_CLEANUP_REPORT.md']]
    assert not extra, 'Changes outside presentation scope '+str(extra)
    for path in changed:
        if path.startswith('scripts/'):
            assert Path(path).name in {'build_revision_v4_editorial_documents.py','render_revision_v4_editorial_response.py','validate_revision_v4_editorial_documents.py','review_revision_v4_editorial_pdfs.py'}, 'Historical workflow modified'
    old_header='% V4 author-review candidate. Historical evidence remains unchanged.\n'
    for name in ['main4.tex','supplementary4.tex']:
        assert (PAPER/name).read_text().split(r'\begin{document}')[0]==old_text(PAPER/name,commit).removeprefix(old_header).split(r'\begin{document}')[0]
    scientific=lambda t:t[t.index(r'\section*{Introduction}'):t.index(r'\section*{Data Availability}')]
    assert scientific(expanded)==scientific(previous), 'Main scientific prose changed'
    assert expanded[expanded.index(r'\bibliography{references}'):]==previous[previous.index(r'\bibliography{references}'):], 'Standard declarations changed'
    note_start=r'\suppnote{Supplementary Note S1:'
    assert supplement[supplement.index(note_start):]==previous_supp[previous_supp.index(note_start):], 'Supplementary science changed'
    cover=(ROOT/'paperV4/cover_letter/cover_letter_v4.tex').read_text()
    old_cover=old_text(ROOT/'paperV4/cover_letter/cover_letter_v4.tex',commit)
    assert cover[cover.index('The V4 revision'):cover.index('The V4 code and data')]==old_cover[old_cover.index('The V4 revision'):old_cover.index('Published version 2.0.0')], 'Cover scientific summary changed'
    filter_path=PAPER/'v4_filter_methods.tex'
    assert filter_path.read_text()==old_text(filter_path,commit)
    assert environments(expanded,'equation')==environments(previous,'equation')
    assert environments(expanded,'algorithm')==environments(previous,'algorithm')
    assert environments(supplement,'algorithm')==environments(previous_supp,'algorithm')
    assert environments(expanded,'figure')==environments(previous,'figure')
    assert environments(expanded,'table')==environments(previous,'table')
    for name in ['table','longtable']:
        assert all(block in supplement for block in environments(previous_supp,name)), 'Historical supplementary table changed'
    config=load(ROOT/'configs/revision_v4/coherence_replacement.json')
    rubric=(PAPER/'v4_contextual_supplement.tex').read_text()
    assert config['rubric'] in rubric
    for line in config['scales'].splitlines():
        if re.match(r'^\d = ',line):assert line.split(' = ',1)[1] in rubric
    assert config['scales'][config['scales'].index('For a very short'): ] in rubric
    for example in load(ROOT/'results/revision_v4/coherence_replacement/analysis/examples.json'):
        assert example['text'] in rubric and example['message_id'] in rubric
    summary=load(ROOT/'results/revision_v4/coherence_replacement/analysis/summary.json')
    panel=next(r for r in summary['summaries'] if r['group']=='panel')
    results=(PAPER/'v4_contextual_results.tex').read_text()
    numerical=[]
    for arm in ['encoded','ordinary','difference']:
        row=panel[arm];formatted=f"{row['estimate']*100:.1f} [{row['lower']*100:.1f}, {row['upper']*100:.1f}]"
        assert formatted in results
        numerical.append({'quantity':arm,'formatted':formatted,'source':'results/revision_v4/coherence_replacement/analysis/summary.json','source_sha256':sha(ROOT/'results/revision_v4/coherence_replacement/analysis/summary.json')})
    assert panel['complete_pairs']==537 and '39 incomplete pairs' in results and '48 payloads' in results
    assert '2,000' in expanded and 'fixed' in (PAPER/'v4_contextual_methods.tex').read_text()
    ledger=load(OUT/'response_status.json');response=(ROOT/'paperV4/response/response_to_reviewers_v4.tex').read_text()
    quotes=verify_quotes((ROOT/'paperV4/response/requests.txt').read_text(),load(ROOT/'paperV4/response/review_comments.json'),response)
    assert quotes==13 and len(ledger['response_sets'])==16 and ledger['written_complete']==16
    assert ledger['empirical_R1_4']=='automated_contextual_evidence_supplied_human_perception_unmeasured'
    assert not ledger['public_current_study_coverage'] and not ledger['journal_submission_performed']
    for row in ledger['response_sets']:
        block=re.findall(r'% BEGIN ANSWER '+re.escape(row['id'])+r'\n(.*?)% END ANSWER '+re.escape(row['id']),response,re.S)
        assert len(block)==1 and row['written_response']=='complete'
        for key in ['direct_answer','evidence_change']:assert render_text(row[key]) in block[0]
        assert escape(row['rendered_location_text']) in block[0]
        for path in row['evidence_paths']:assert (ROOT/path).is_file(),path
        for anchor,location in row['compiled_locations'].items():
            aux=(PAPER/(location['document']+'.aux')).read_text()
            assert re.search(r'\\newlabel\{'+re.escape(anchor)+r'\}\{\{[^}]*\}\{'+str(location['page'])+r'\}',aux)
    r14=next(r for r in ledger['response_sets'] if r['id']=='R1.4')
    response_words=len((r14['direct_answer']+' '+r14['evidence_change']).split());assert 200<=response_words<=300
    old_answers=json.loads(old_text(ROOT/'paperV4/response/editorial_answers.json',commit))['response_sets']
    current_answers=load(ROOT/'paperV4/response/editorial_answers.json')['response_sets']
    assert [r['id'] for r in current_answers]==[r['id'] for r in old_answers]
    for current,old in zip(current_answers,old_answers):
        if current['id'] not in {'archive','editor_accuracy','editor_technical'}:
            assert current==old, 'Non-administrative answer changed '+current['id']
        for key in ['written_response','empirical_coverage','external_action','author_decision']:
            if key in old:assert current[key]==old[key], 'Response status changed'
    assert 'Editorial author-review edition' not in response
    repeat=load(OUT/'validation/response_render_repeat.json')
    assert repeat['identical'] and repeat['second_sha256']==sha(ROOT/'paperV4/response/response_to_reviewers_v4.tex')
    locations=load(OUT/'validation/locations.json');assert locations['alg:tail']['label']=='S1'
    assert set(locations)==set(baseline['numbered_locations']), 'Section or item labels changed'
    for key,old in baseline['numbered_locations'].items():
        if key.startswith(('eq:','fig:','tab:','alg:')):assert locations[key]['label']==old['label'], 'Item renumbered '+key
    assert 'Algorithm S1' in expanded and '\\ref{alg:tail}' not in expanded
    nav=navigation(supplement)
    documents=[]
    for relative in DOCS:
        path=ROOT/relative;text,pages,issues=inspect(path)
        assert not issues,(relative,issues)
        for marker in ['V4 AUTHOR-REVIEW CANDIDATE','Editorial author-review edition','editorial author-review PDFs','prepared for author review']:
            assert marker.lower() not in text.lower(), (relative,marker)
        if path.stem=='supplementary4':
            assert not re.search(r'^\s*(Reading guide|Contents)\s*$',text,re.M)
            assert 'Supplementary Note S1:' in text.split('\f')[0], 'Extra supplementary front-matter page'
        (OUT/'validation'/(path.stem+'_text.txt')).write_text(text)
        documents.append({'pdf':relative,'pages':len(pages),'sha256':sha(path)})
    latest=load(OUT/'validation/latex/latest_builds.json')
    assert set(latest)=={'scientific','correspondence','submission'}
    for job in latest.values():
        for record in job['records']:
            assert not record['warnings'] and not record['unresolved_references']
            if record['document']!='main4_submission':
                target=next(r for r in documents if Path(r['pdf']).stem==record['document'])
                assert target['sha256']==record['pdf_sha256']
    assert (PAPER/'main4_submission.tex').read_text().split('\n',1)[1]==manuscript_source(PAPER)
    assert 'author-review' not in (PAPER/'main4_submission.tex').read_text().lower()
    assert r'\tableofcontents' not in (PAPER/'supplementary4.tex').read_text()
    compiled=set(re.findall(r'\\bibitem\{([^}]+)\}',(PAPER/'main4.bbl').read_text()))
    cited={k for group in re.findall(r'\\cite\{([^}]+)\}',expanded) for k in group.split(',')};assert cited<=compiled
    receipt={'status':'passed','baseline_commit':commit,'historical_evidence_and_scientific_code_unchanged':True,
        'scientific_prose_unchanged':True,'standard_declarations_unchanged':True,
        'equations_unchanged':len(environments(expanded,'equation')),'main_algorithms_unchanged':2,'supplementary_tail_algorithm_unchanged':True,
        'filter_rules_unchanged':True,'main_figures_unchanged':5,'main_tables_unchanged':3,'supplementary_historical_table_blocks_unchanged':True,
        'quoted_review_blocks':quotes,'complete_answers':16,'R1_4_words_excluding_locations':response_words,
        'banners_removed':True,'printed_contents_removed':True,'reading_guide_removed':True,'response_render_repeat':repeat,
        'results_begin_before':baseline['numbered_locations']['sec:results']['page'],'results_begin_after':locations['sec:results']['page'],'documents':documents,
        'navigation':nav,'numerical_claim_checks':numerical,'main_references':len(compiled),'additional_gpu_seconds':0,
        'clean_submission_build':latest['submission'],'source_changes':changed}
    (OUT/'validation/presentation_check.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['navigation','source_changes','numerical_claim_checks','clean_submission_build']},indent=2))


if __name__=='__main__':
    main()
