"""Validate the current punctuation edit without rerunning scientific analyses."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
from scripts.prepare_revision_v4_stage3_handoff import flatten
from scripts.prepare_revision_v4_stage1 import escape
from scripts.render_revision_v4_editorial_response import render_text
from scripts.validate_revision_v4_editorial_documents import old_text, old_flatten, environments, norm
from scripts.validate_revision_v4_stage2 import verify_quotes
from scripts.review_revision_v4_editorial_pdfs import inspect, DOCS
from scripts.build_revision_v4_stage4_bundle import manuscript_source

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/revision_v4/punctuation_edit'
PAPER = ROOT/'paperV4/scientific_reports'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    manifest=load(OUT/'authorized_edits.json')
    assert commit==manifest['baseline_commit']=='dec80d2daff76d2fd3463950951b6012a04d6057'
    entries=manifest['source_edits'];answer_edits=manifest['answer_edits']
    sources={r['path'] for r in entries}
    numbers=lambda text:re.findall(r'(?<![A-Za-z0-9])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?(?![A-Za-z0-9])',text)
    for relative in sources:
        expected=old_text(ROOT/relative,commit)
        for row in [r for r in entries if r['path']==relative]:
            assert hashlib.sha256(row['old'].encode()).hexdigest()==row['old_sha256']
            assert expected.count(row['old'])==1,(relative,row['line'])
            assert numbers(row['old'])==numbers(row['new']),(relative,'numbers',row['line'])
            expected=expected.replace(row['old'],row['new'],1)
        assert (ROOT/relative).read_text()==expected,relative
    answer_path=ROOT/'paperV4/response/editorial_answers.json'
    expected_answers=json.loads(old_text(answer_path,commit))
    for row in answer_edits:
        target=next(r for r in expected_answers['response_sets'] if r['id']==row['id'])
        assert row['field'] in ['direct_answer','evidence_change']
        assert target[row['field']]==row['old']
        assert numbers(row['old'])==numbers(row['new'])
        target[row['field']]=row['new']
    assert load(answer_path)==expected_answers
    allowed=sources | set(DOCS) | {
        '.gitignore','paperV4/DOCUMENT_BUILD.md','paperV4/REVIEW_RESPONSE_MATRIX.md',
        'paperV4/scientific_reports/main4.bbl','paperV4/scientific_reports/supplementary4.bbl',
        'paperV4/scientific_reports/main4_submission.tex','paperV4/response/editorial_answers.json',
        'paperV4/response/response_to_reviewers_v4.tex','revision_docs/REVISION_V4_PUNCTUATION_EDIT_REPORT.md',
        'scripts/build_revision_v4_editorial_documents.py','scripts/render_revision_v4_editorial_response.py',
        'scripts/review_revision_v4_editorial_pdfs.py','scripts/validate_revision_v4_punctuation_edit.py'}
    changed=subprocess.check_output(['git','diff','--name-only',commit],cwd=ROOT,text=True).splitlines()
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    assert all(p in allowed or p.startswith('results/revision_v4/punctuation_edit/') for p in changed+untracked)
    for relative in ['paperV4/response/requests.txt','paperV4/response/review_comments.json','paperV4/cover_letter/cover_letter_v4.tex']:
        assert (ROOT/relative).read_text()==old_text(ROOT/relative,commit)
    assert (PAPER/'v4_filter_methods.tex').read_text()==old_text(PAPER/'v4_filter_methods.tex',commit)
    expanded=flatten(PAPER/'main4.tex');supplement=flatten(PAPER/'supplementary4.tex')
    previous=old_flatten(PAPER/'main4.tex',commit);previous_supp=old_flatten(PAPER/'supplementary4.tex',commit)
    for stem in ['main4','supplementary4']:
        assert (PAPER/(stem+'.tex')).read_text().split(r'\begin{document}')[0]==old_text(PAPER/(stem+'.tex'),commit).split(r'\begin{document}')[0]
    assert expanded[expanded.index(r'\section*{Data Availability}'):]==previous[previous.index(r'\section*{Data Availability}'):]
    for current,old in [(expanded,previous),(supplement,previous_supp)]:
        assert numbers(current)==numbers(old),'Numeric sequence changed'
        for environment in ['equation','align','algorithm','tabular','tabularx','longtable','verbatim','Verbatim','quote']:
            assert environments(current,environment)==environments(old,environment),environment
        for pattern in [r'\\\(.*?\\\)',r'\\(?:label|cite|ref|eqref|code|path|url)\{[^}]*\}',r'\\includegraphics(?:\[[^]]*\])?\{[^}]+\}']:
            assert re.findall(pattern,current,re.S)==re.findall(pattern,old,re.S),pattern
    rubric=(PAPER/'v4_contextual_supplement.tex').read_text()
    old_rubric=old_text(PAPER/'v4_contextual_supplement.tex',commit)
    start=r'\suppsubsection{Complete judging rubric}';end=r'\suppsubsection{Judge settings and calibration}'
    assert rubric[rubric.index(start):rubric.index(end)]==old_rubric[old_rubric.index(start):old_rubric.index(end)]
    examples=r'\suppsubsection{Full-message examples}'
    assert rubric[rubric.index(examples):]==old_rubric[old_rubric.index(examples):]
    assert not re.search(r'AI assistance|\b(?:Codex|ChatGPT|OpenAI)\b',expanded,re.I)
    response_path=ROOT/'paperV4/response/response_to_reviewers_v4.tex';response=response_path.read_text()
    blank_answers=lambda text:re.sub(r'(% BEGIN ANSWER [^\n]+\n).*?(% END ANSWER [^\n]+)',r'\1ANSWER\2',text,flags=re.S)
    assert blank_answers(response)==blank_answers(old_text(response_path,commit))
    quotes=verify_quotes((ROOT/'paperV4/response/requests.txt').read_text(),load(ROOT/'paperV4/response/review_comments.json'),response)
    ledger=load(OUT/'response_status.json');assert quotes==13 and len(ledger['response_sets'])==ledger['written_complete']==16
    assert ledger['empirical_R1_4']=='automated_contextual_evidence_supplied_human_perception_unmeasured'
    assert not ledger['public_current_study_coverage'] and not ledger['journal_submission_performed']
    for row in ledger['response_sets']:
        block=re.findall(r'% BEGIN ANSWER '+re.escape(row['id'])+r'\n(.*?)% END ANSWER '+re.escape(row['id']),response,re.S)
        assert len(block)==1 and row['written_response']=='complete'
        for key in ['direct_answer','evidence_change']:assert render_text(row[key]) in block[0]
        assert escape(row['rendered_location_text']) in block[0]
        for anchor,location in row['compiled_locations'].items():
            aux=(PAPER/(location['document']+'.aux')).read_text()
            assert re.search(r'\\newlabel\{'+re.escape(anchor)+r'\}\{\{[^}]*\}\{'+str(location['page'])+r'\}',aux)
    repeat=load(OUT/'validation/response_render_repeat.json')
    assert repeat['identical'] and repeat['second_sha256']==sha(response_path)
    locations=load(OUT/'validation/locations.json');assert set(locations)==set(baseline['numbered_locations'])
    for key,old in baseline['numbered_locations'].items():
        if key.startswith(('eq:','fig:','tab:','alg:')):assert locations[key]['label']==old['label']
    assert locations['alg:tail']['label']=='S1'
    nav=navigation(supplement);assert len(nav)==61
    builds=load(OUT/'validation/latex/latest_builds.json')
    assert set(builds)=={'scientific','correspondence','submission'}
    documents=[]
    for relative in DOCS:
        path=ROOT/relative;text,pages,issues=inspect(path);assert not issues,(relative,issues)
        for marker in ['V4 AUTHOR-REVIEW CANDIDATE','Editorial author-review edition','editorial author-review PDFs','prepared for author review']:
            assert marker.lower() not in text.lower()
        if path.stem=='supplementary4':assert not re.search(r'^\s*(Reading guide|Contents)\s*$',text,re.M)
        raw=subprocess.check_output(['pdfinfo','-dests',str(path)],text=True)
        destinations=set(re.findall(r'"([^"]+)"',raw))
        objects=subprocess.check_output(['mutool','show','-g',str(path),'grep'],text=True)
        assert set(re.findall(r'/D\(([^)]+)\)',objects))<=destinations
        record=next(r for job in builds.values() for r in job['records'] if r['document']==path.stem)
        assert record['pdf_sha256']==sha(path) and not record['warnings'] and not record['unresolved_references']
        documents.append({'path':relative,'pages':len(pages),'sha256':sha(path)})
    submission=(PAPER/'main4_submission.tex').read_text()
    assert submission.split('\n',1)[1]==manuscript_source(PAPER)
    assert (PAPER/'main4.bbl').read_text() in submission
    assert not re.search(r'\\(?:input|includegraphics|bibliography)\{',submission)
    assert environments(submission,'abstract')==environments(expanded,'abstract')
    assert environments(submission,'table')==environments(expanded,'table')
    clean=builds['submission']['records'][0]
    assert clean['clean_dependency_build'] and not clean['warnings'] and not clean['unresolved_references']
    for job in builds.values():
        for record in job['records']:
            number=4 if record['document'] in ['main4','supplementary4'] else 3
            log=(ROOT/job['logs']/(record['document']+f'_pass{number}.txt')).read_text()
            assert not re.search(r'Missing character|multiply.defined|duplicate.*destination|destination with the same identifier',log,re.I)
    punctuation={c:sum(r['old'].count(c)-r['new'].count(c) for r in entries+answer_edits) for c in [';',':','-','–','—']}
    assert punctuation==manifest['punctuation_removed']
    visual={'status':'separate visual review required'}
    if (OUT/'validation/pdf_review.json').exists():
        review=load(OUT/'validation/pdf_review.json')
        if review.get('visual_review_complete'):
            for row in review['documents']:
                assert sha(ROOT/row['document'])==row['sha256']
                assert set(row['text_or_geometry_changed_pages'])<=set(row['visually_reviewed_pages'])
            visual={'status':'passed','visually_reviewed_pages':review['visually_reviewed_page_count'],'automatic_pages':review['automatic_page_count']}
    receipt={'status':'passed','baseline_commit':commit,'edited_source_paragraphs':len(entries),'edited_answer_fields':len(answer_edits),
        'punctuation_removed':punctuation,'abstract_and_formatting_unchanged':True,'all_numeric_values_and_order_unchanged':True,
        'equations_algorithms_table_bodies_and_filter_unchanged':True,'rubric_and_literal_examples_unchanged':True,
        'prior_evidence_and_experiment_code_unchanged':True,'availability_and_submission_status_unchanged':True,
        'assistance_disclosure_remains_removed':True,'quoted_review_blocks':quotes,'complete_answers':16,
        'compiled_labels_checked':len(locations),'bookmarks':nav,'response_render_stable':True,'documents':documents,
        'editable_source_sha256':sha(PAPER/'main4_submission.tex'),'clean_dependency_build':clean,'visual_review':visual,'gpu_seconds':0}
    (OUT/'validation/checks.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['bookmarks','clean_dependency_build']},indent=2))


if __name__=='__main__':main()
