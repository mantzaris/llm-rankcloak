"""Check the author-requested disclosure removal without altering older receipts."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from scripts.prepare_revision_v4_stage3_handoff import flatten
from scripts.validate_revision_v4_editorial_documents import old_text, old_flatten
from scripts.validate_revision_v4_stage2 import verify_quotes
from scripts.review_revision_v4_editorial_pdfs import inspect, DOCS
from scripts.build_revision_v4_stage4_bundle import manuscript_source
from scripts.prepare_revision_v4_stage1 import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/revision_v4/disclosure_edit'
PAPER = ROOT/'paperV4/scientific_reports'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    baseline = load(OUT/'baseline.json')
    commit = baseline['head']
    block = baseline['removed_block']
    assert commit == '91cf154102652d15a40920349dd91c27d436d039'
    assert hashlib.sha256(block.encode()).hexdigest() == baseline['removed_block_sha256']
    previous = old_flatten(PAPER/'main4.tex',commit)
    current = flatten(PAPER/'main4.tex')
    assert previous.count(block) == 1
    assert current == previous.replace(block,'',1), 'Change beyond the requested disclosure removal'
    assert flatten(PAPER/'supplementary4.tex') == old_flatten(PAPER/'supplementary4.tex',commit)
    for path in ['paperV4/cover_letter/cover_letter_v4.tex','paperV4/response/editorial_answers.json',
                 'paperV4/response/requests.txt','paperV4/response/review_comments.json']:
        assert (ROOT/path).read_text() == old_text(ROOT/path,commit), path
    # All scientific data, analysis code, configuration, figures and past receipts are protected.
    allowed = set(DOCS) | {'paperV4/scientific_reports/main4.tex','paperV4/scientific_reports/main4_submission.tex',
        'paperV4/scientific_reports/main4.bbl','paperV4/scientific_reports/supplementary4.bbl',
        'paperV4/response/response_to_reviewers_v4.tex','paperV4/REVIEW_RESPONSE_MATRIX.md','paperV4/DOCUMENT_BUILD.md',
        '.gitignore','scripts/build_revision_v4_editorial_documents.py','scripts/render_revision_v4_editorial_response.py',
        'scripts/review_revision_v4_editorial_pdfs.py','scripts/validate_revision_v4_disclosure_edit.py'}
    changed = subprocess.check_output(['git','diff','--name-only',commit],cwd=ROOT,text=True).splitlines()
    untracked = subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    assert all(p in allowed or p.startswith('results/revision_v4/disclosure_edit/') for p in changed+untracked)
    response_path = ROOT/'paperV4/response/response_to_reviewers_v4.tex'
    response = response_path.read_text()
    strip_locations=lambda t:re.sub(r'(?m)^\\textbf\{Location\.\} .*$', 'LOCATION', t)
    assert strip_locations(response) == strip_locations(old_text(response_path,commit))
    quotes = verify_quotes((ROOT/'paperV4/response/requests.txt').read_text(), load(ROOT/'paperV4/response/review_comments.json'), response)
    assert quotes == 13
    ledger = load(OUT/'response_status.json')
    assert ledger['written_complete'] == 16 and len(ledger['response_sets']) == 16
    locations = load(OUT/'validation/locations.json')
    assert set(locations) == set(baseline['numbered_locations']) - {'sec:revision-assistance'}
    for key,old in baseline['numbered_locations'].items():
        if key.startswith(('eq:','fig:','tab:','alg:')):
            assert locations[key]['label'] == old['label'], key
    for row in ledger['response_sets']:
        assert escape(row['rendered_location_text']) in response
        for anchor,location in row['compiled_locations'].items():
            aux=(PAPER/(location['document']+'.aux')).read_text()
            assert re.search(r'\\newlabel\{'+re.escape(anchor)+r'\}\{\{[^}]*\}\{'+str(location['page'])+r'\}',aux)
    repeat = load(OUT/'validation/response_render_repeat.json')
    assert repeat['identical'] and repeat['second_sha256'] == sha(response_path)
    submission = (PAPER/'main4_submission.tex').read_text()
    assert submission.split('\n',1)[1] == manuscript_source(PAPER)
    assert (PAPER/'main4.bbl').read_text() in submission
    assert not re.search(r'\\(?:input|includegraphics|bibliography)\{',submission)
    forbidden = r'\b(?:Codex|ChatGPT|OpenAI)\b|AI assistance|sec:revision-assistance'
    assert not re.search(forbidden,current+'\n'+submission,re.I)
    builds = load(OUT/'validation/latex/latest_builds.json')
    assert set(builds) == {'scientific','correspondence','submission'}
    documents = []
    for relative in DOCS:
        path = ROOT/relative
        text,pages,issues = inspect(path)
        assert not issues, (relative,issues)
        assert not re.search(forbidden,text,re.I)
        source = path.with_suffix('.tex')
        assert not re.search(forbidden,source.read_text(),re.I)
        record=next(r for group in builds.values() for r in group['records'] if r['document']==path.stem)
        assert record['pdf_sha256']==sha(path) and not record['warnings'] and not record['unresolved_references']
        documents.append({'path':relative,'pages':len(pages),'sha256':sha(path)})
    clean=builds['submission']['records'][0]
    assert clean['clean_dependency_build'] and not clean['warnings'] and not clean['unresolved_references']
    receipt={'status':'passed','baseline_commit':commit,'change':'Disclosure heading and paragraph removed at author request; no replacement wording inserted.',
        'all_other_manuscript_and_supplement_text_unchanged':True,'abstract_and_scientific_evidence_unchanged':True,
        'historical_disclosure_records_unchanged':True,'quotes_preserved':quotes,'substantive_answers_preserved':16,
        'remaining_compiled_labels':len(locations),'response_render_stable':True,'documents':documents,
        'editable_source_sha256':sha(PAPER/'main4_submission.tex'),'clean_dependency_build':clean,
        'additional_gpu_seconds':0,'author_will_supply_replacement_disclosure':True}
    (OUT/'validation/checks.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
