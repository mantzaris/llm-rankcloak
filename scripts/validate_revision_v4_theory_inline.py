"""Check the theory relocation and current document builds without analysis runs."""
import hashlib
import json
from pathlib import Path
import re
from scripts.prepare_revision_v4_stage1 import escape
from scripts.prepare_revision_v4_stage3_handoff import flatten
from scripts.validate_revision_v4_editorial_documents import old_text, environments
from scripts.validate_revision_v4_stage2 import verify_quotes
from scripts.review_revision_v4_editorial_pdfs import inspect, DOCS
from scripts.build_revision_v4_stage4_bundle import manuscript_source
from scripts.render_revision_v4_editorial_response import render_text

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision_v4/theory_inline'
PAPER=ROOT/'paperV4/scientific_reports'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    baseline=load(OUT/'baseline.json');commit=baseline['head']
    original=baseline['main4_before'];current=(PAPER/'main4.tex').read_text()
    theory_path=PAPER/'v4_stage2_theory.tex';theory=theory_path.read_text()
    assert sha(theory_path)==baseline['theory_sha256']
    assert theory==old_text(theory_path,commit)
    heading=r'\section*{Discussion}\label{sec:discussion}'
    expected=original.replace('\\input{v4_stage2_theory.tex}\n\n','',1)
    expected=expected.replace(heading,theory.rstrip()+'\n\n'+heading,1)
    expected=expected.replace(r'\section*{Responsible use and limitations}',r'\subsection*{Responsible use and limitations}',1)
    assert current==expected,'Unexpected manuscript edit'
    assert r'\input{v4_stage2_theory.tex}' not in current
    assert current.count(theory.rstrip())==1
    assert re.findall(r'\\section\*\{([^}]+)\}',current.split(r'\section*{Data Availability}')[0])[-1]=='Discussion'
    answer_path=ROOT/'paperV4/response/editorial_answers.json'
    expected_answers=json.loads(old_text(answer_path,commit))
    for row in expected_answers['response_sets']:
        row['location_text']=row['location_text'].replace('main Discussion under Configuration sensitivity','main Results under Configuration sensitivity').replace('Main Discussion under Configuration sensitivity','Main Results under Configuration sensitivity')
    row=next(r for r in expected_answers['response_sets'] if r['id']=='general')
    row['evidence_change']=row['evidence_change'].replace('The discussion states that a fully configured attacker may decode','The configuration-sensitivity analysis states that a fully configured attacker may decode',1)
    assert load(answer_path)==expected_answers
    for rel in ['paperV4/response/requests.txt','paperV4/response/review_comments.json','paperV4/scientific_reports/supplementary4.tex','paperV4/cover_letter/cover_letter_v4.tex']:
        assert (ROOT/rel).read_text()==old_text(ROOT/rel,commit)
    response_path=ROOT/'paperV4/response/response_to_reviewers_v4.tex';response=response_path.read_text()
    blank=lambda t:re.sub(r'(% BEGIN ANSWER [^\n]+\n).*?(% END ANSWER [^\n]+)',r'\1ANSWER\2',t,flags=re.S)
    assert blank(response)==blank(old_text(response_path,commit))
    quotes=verify_quotes((ROOT/'paperV4/response/requests.txt').read_text(),load(ROOT/'paperV4/response/review_comments.json'),response)
    ledger=load(OUT/'response_status.json');assert quotes==13 and ledger['written_complete']==len(ledger['response_sets'])==16
    for row in ledger['response_sets']:
        block=re.findall(r'% BEGIN ANSWER '+re.escape(row['id'])+r'\n(.*?)% END ANSWER '+re.escape(row['id']),response,re.S)
        assert len(block)==1
        for key in ['direct_answer','evidence_change']:
            assert render_text(row[key]) in block[0]
        assert escape(row['rendered_location_text']) in block[0]
    locations=load(OUT/'validation/locations.json')
    assert set(locations)==set(baseline['numbered_locations'])
    for key,location in locations.items():
        aux=(PAPER/(location['document']+'.aux')).read_text()
        assert re.search(r'\\newlabel\{'+re.escape(key)+r'\}\{\{[^}]*\}\{'+str(location['page'])+r'\}',aux)
        if key.startswith(('eq:','alg:','fig:','tab:')):
            assert location['label']==baseline['numbered_locations'][key]['label']
    repeat=load(OUT/'validation/response_render_repeat.json')
    assert repeat['identical'] and repeat['second_sha256']==sha(response_path)
    expanded=flatten(PAPER/'main4.tex');submission=(PAPER/'main4_submission.tex').read_text()
    assert submission.split('\n',1)[1]==manuscript_source(PAPER)
    assert environments(submission,'abstract')==environments(expanded,'abstract')
    assert environments(submission,'table')==environments(expanded,'table')
    assert (PAPER/'main4.bbl').read_text() in submission
    builds=load(OUT/'validation/latex/latest_builds.json');records=[]
    for job in builds.values():
        for record in job['records']:
            assert not record['warnings'] and not record['unresolved_references']
            number=4 if record['document'] in ['main4','supplementary4'] else 3
            log=(ROOT/job['logs']/(record['document']+f'_pass{number}.txt')).read_text()
            assert not re.search(r'Missing character|multiply.defined|duplicate.*destination|destination with the same identifier',log,re.I)
    for rel in DOCS:
        path=ROOT/rel;text,pages,issues=inspect(path);assert not issues,(rel,issues)
        built=next(r for job in builds.values() for r in job['records'] if r['document']==path.stem)
        assert built['pdf_sha256']==sha(path)
        records.append({'path':rel,'pages':len(pages),'sha256':sha(path)})
    clean=builds['submission']['records'][0];assert clean['clean_dependency_build']
    review=load(OUT/'validation/pdf_review.json')
    assert review['visual_review_complete']
    for row in review['documents']:
        assert sha(ROOT/row['document'])==row['sha256']
        assert set(row['text_or_geometry_changed_pages'])<=set(row['visually_reviewed_pages'])
    receipt={'status':'passed','baseline_commit':commit,'theory_inlined_verbatim':True,'discussion_last_scientific_section':True,
        'initial_author_wording_preserved':True,'new_scientific_edits':False,'original_theory_file_unchanged':True,
        'review_quotes':quotes,'complete_answers_only_location_terminology_changed':16,'compiled_labels':len(locations),'render_stable':True,
        'documents':records,'editable_source_sha256':sha(PAPER/'main4_submission.tex'),'clean_dependency_build':clean,
        'visually_reviewed_pages':review['visually_reviewed_page_count'],'gpu_seconds':0}
    (OUT/'validation/checks.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
