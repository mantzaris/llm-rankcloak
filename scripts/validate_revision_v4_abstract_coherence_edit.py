"""Run retained document-preservation and build checks for this editorial pass.

The replacement ledger limits permitted source differences. It does not test
scientific outcomes or regenerate any analysis. Historical checks remain intact.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from scripts.prepare_revision_v4_stage3_handoff import flatten
from scripts.prepare_revision_v4_stage1 import escape
from scripts.render_revision_v4_editorial_response import render_text
from scripts.validate_revision_v4_editorial_documents import old_text, old_flatten, environments
from scripts.validate_revision_v4_stage2 import verify_quotes
from scripts import validate_revision_v4_punctuation_edit as retained_checks
from scripts.review_revision_v4_editorial_pdfs import inspect, DOCS
from scripts.build_revision_v4_stage4_bundle import manuscript_source

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/revision_v4/abstract_coherence_edit'
PAPER = ROOT/'paperV4/scientific_reports'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numbers(text):
    return re.findall(r'(?<![A-Za-z0-9])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?(?![A-Za-z0-9])', text)


def blank_answers(text):
    return re.sub(r'(% BEGIN ANSWER [^\n]+\n).*?(% END ANSWER [^\n]+)', r'\1ANSWER\2', text, flags=re.S)


def without_abstract(text):
    return re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}', '', text, flags=re.S)


def main():
    baseline = load(OUT/'baseline.json'); commit = baseline['head']
    manifest = load(OUT/'authorized_edits.json')
    assert commit == manifest['baseline_commit']
    edits = manifest['source_edits']; answer_edits = manifest['answer_edits']
    sources = {r['path'] for r in edits}
    for relative in sources:
        expected = old_text(ROOT/relative, commit)
        for row in manifest['initial_author_adjustments'] + edits:
            if row['path'] != relative:
                continue
            assert expected.count(row['old']) == 1, (relative, row.get('id'))
            if 'old_sha256' in row:
                assert hashlib.sha256(row['old'].encode()).hexdigest() == row['old_sha256']
            expected = expected.replace(row['old'], row['new'], 1)
        actual = (ROOT/relative).read_text()
        if relative.endswith('response_to_reviewers_v4.tex'):
            assert blank_answers(actual) == blank_answers(expected)
        else:
            assert actual == expected, relative
    answers_path = ROOT/'paperV4/response/editorial_answers.json'
    expected = json.loads(old_text(answers_path, commit))
    for row in answer_edits:
        target = next(r for r in expected['response_sets'] if r['id'] == row['id'])
        assert row['field'] in ['direct_answer', 'evidence_change']
        assert target[row['field']] == row['old']
        assert numbers(row['old']) == numbers(row['new'])
        target[row['field']] = row['new']
    assert load(answers_path) == expected
    allowed = sources | set(DOCS) | {
        '.gitignore', 'paperV4/DOCUMENT_BUILD.md', 'paperV4/REVIEW_RESPONSE_MATRIX.md',
        'paperV4/scientific_reports/main4.bbl', 'paperV4/scientific_reports/main4_submission.tex',
        'paperV4/response/editorial_answers.json', 'revision_docs/REVISION_V4_ABSTRACT_COHERENCE_EDIT_REPORT.md',
        'scripts/build_revision_v4_editorial_documents.py', 'scripts/render_revision_v4_editorial_response.py',
        'scripts/review_revision_v4_editorial_pdfs.py', 'scripts/validate_revision_v4_abstract_coherence_edit.py'}
    changed = subprocess.check_output(['git','diff','--name-only',commit],cwd=ROOT,text=True).splitlines()
    untracked = subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    assert all(p in allowed or p.startswith('results/revision_v4/abstract_coherence_edit/') for p in changed+untracked)
    for relative in ['paperV4/response/requests.txt','paperV4/response/review_comments.json','paperV4/scientific_reports/v4_filter_methods.tex']:
        assert (ROOT/relative).read_text() == old_text(ROOT/relative,commit)
    main_text = flatten(PAPER/'main4.tex'); supplement = flatten(PAPER/'supplementary4.tex')
    previous = old_flatten(PAPER/'main4.tex',commit); previous_supp = old_flatten(PAPER/'supplementary4.tex',commit)
    v3_abstract = environments((ROOT/'paperV3/scientific_reports/main3.tex').read_text(),'abstract')[0]
    assert environments(main_text,'abstract')[0] == v3_abstract.replace(manifest['abstract_transport_old'],manifest['abstract_transport_new'])
    for stem in ['main4','supplementary4']:
        current = (PAPER/(stem+'.tex')).read_text().split(r'\begin{document}')[0]
        old = old_text(PAPER/(stem+'.tex'),commit).split(r'\begin{document}')[0]
        assert without_abstract(current) == without_abstract(old)
    availability = r'\section*{Data Availability}'
    assert main_text[main_text.index(availability):] == previous[previous.index(availability):]
    # The sole removed numeral outside the abstract is the historical GPT-4 model
    # name in the explicitly replaced G-Eval sentence, not an experimental value.
    source_context = next(r for r in edits if r['id']=='7G')
    previous_supp = previous_supp.replace(source_context['old'], source_context['new'])
    for current, old in [(without_abstract(main_text),without_abstract(previous)), (supplement,previous_supp)]:
        assert numbers(current) == numbers(old), 'Scientific numeric sequence changed'
        # Only the two explicitly requested scope cells differ inside table bodies.
        for row in edits:
            if row['id'] in ['7C','7D']:
                old = old.replace(row['old'],row['new'])
        for environment in ['equation','align','algorithm','tabular','tabularx','longtable','verbatim','Verbatim','quote']:
            assert environments(current,environment) == environments(old,environment), environment
        for pattern in [r'\\\(.*?\\\)',r'\\(?:label|ref|eqref|code|path|url)\{[^}]*\}',r'\\includegraphics(?:\[[^]]*\])?\{[^}]+\}']:
            assert re.findall(pattern,current,re.S) == re.findall(pattern,old,re.S), pattern
        old_cites = re.findall(r'\\cite\{[^}]*\}',old)
        old_cites = [c for c in old_cites if c != r'\cite{VanDerLee2019BestPractices}']
        assert re.findall(r'\\cite\{[^}]*\}',current) == old_cites
    rubric = (PAPER/'v4_contextual_supplement.tex').read_text(); old_rubric = old_text(PAPER/'v4_contextual_supplement.tex',commit)
    start = r'\suppsubsection{Complete judging rubric}'; end = r'\suppsubsection{Judge settings and calibration}'
    assert rubric[rubric.index(start):rubric.index(end)] == old_rubric[old_rubric.index(start):old_rubric.index(end)]
    examples = r'\paragraph{Acceptable}'
    assert rubric[rubric.index(examples):] == old_rubric[old_rubric.index(examples):]
    assert not re.search(r'AI assistance|\b(?:Codex|ChatGPT|OpenAI)\b',main_text,re.I)
    response_path = ROOT/'paperV4/response/response_to_reviewers_v4.tex'; response = response_path.read_text()
    quotes = verify_quotes((ROOT/'paperV4/response/requests.txt').read_text(),load(ROOT/'paperV4/response/review_comments.json'),response)
    ledger = load(OUT/'response_status.json'); assert quotes == 13 and len(ledger['response_sets']) == ledger['written_complete'] == 16
    assert ledger['empirical_R1_4'] == 'automated_contextual_evidence_supplied_human_perception_unmeasured'
    assert not ledger['public_current_study_coverage'] and not ledger['journal_submission_performed']
    for row in ledger['response_sets']:
        block = re.findall(r'% BEGIN ANSWER '+re.escape(row['id'])+r'\n(.*?)% END ANSWER '+re.escape(row['id']),response,re.S)
        assert len(block) == 1 and row['written_response'] == 'complete'
        for key in ['direct_answer','evidence_change']:
            assert render_text(row[key]) in block[0]
        assert escape(row['rendered_location_text']) in block[0]
        for anchor,location in row['compiled_locations'].items():
            aux = (PAPER/(location['document']+'.aux')).read_text()
            assert re.search(r'\\newlabel\{'+re.escape(anchor)+r'\}\{\{[^}]*\}\{'+str(location['page'])+r'\}',aux)
    repeat = load(OUT/'validation/response_render_repeat.json')
    assert repeat['identical'] and repeat['second_sha256'] == sha(response_path)
    locations = load(OUT/'validation/locations.json'); assert set(locations) == set(baseline['numbered_locations'])
    for key,old in baseline['numbered_locations'].items():
        if key.startswith(('eq:','fig:','tab:','alg:')):
            assert locations[key]['label'] == old['label']
    assert locations['alg:tail']['label'] == 'S1'
    # Reuse the existing bookmark/link check, directing only its new receipts here.
    retained_checks.OUT = OUT
    navigation = retained_checks.navigation(supplement); assert len(navigation) == 61
    builds = load(OUT/'validation/latex/latest_builds.json')
    assert set(builds) == {'scientific','correspondence','submission'}
    documents = []
    for relative in DOCS:
        path = ROOT/relative; text,pages,issues = inspect(path); assert not issues,(relative,issues)
        for marker in ['V4 AUTHOR-REVIEW CANDIDATE','Editorial author-review edition','editorial author-review PDFs','prepared for author review']:
            assert marker.lower() not in text.lower()
        if path.stem == 'supplementary4':
            assert not re.search(r'^\s*(Reading guide|Contents)\s*$',text,re.M)
        raw = subprocess.check_output(['pdfinfo','-dests',str(path)],text=True)
        destinations = set(re.findall(r'"([^"]+)"',raw))
        objects = subprocess.check_output(['mutool','show','-g',str(path),'grep'],text=True)
        assert set(re.findall(r'/D\(([^)]+)\)',objects)) <= destinations
        record = next(r for job in builds.values() for r in job['records'] if r['document'] == path.stem)
        assert record['pdf_sha256'] == sha(path) and not record['warnings'] and not record['unresolved_references']
        documents.append({'path':relative,'pages':len(pages),'sha256':sha(path)})
    submission = (PAPER/'main4_submission.tex').read_text()
    assert submission.split('\n',1)[1] == manuscript_source(PAPER)
    assert (PAPER/'main4.bbl').read_text() in submission
    assert not re.search(r'\\(?:input|includegraphics|bibliography)\{',submission)
    assert environments(submission,'abstract') == environments(main_text,'abstract')
    assert environments(submission,'table') == environments(main_text,'table')
    clean = builds['submission']['records'][0]
    assert clean['clean_dependency_build'] and not clean['warnings'] and not clean['unresolved_references']
    for job in builds.values():
        for record in job['records']:
            number = 4 if record['document'] in ['main4','supplementary4'] else 3
            log = (ROOT/job['logs']/(record['document']+f'_pass{number}.txt')).read_text()
            assert not re.search(r'Missing character|multiply.defined|duplicate.*destination|destination with the same identifier',log,re.I)
    visual = {'status':'separate visual review required'}
    if (OUT/'validation/pdf_review.json').exists():
        review = load(OUT/'validation/pdf_review.json')
        if review.get('visual_review_complete'):
            for row in review['documents']:
                assert sha(ROOT/row['document']) == row['sha256']
                assert set(row['text_or_geometry_changed_pages']) <= set(row['visually_reviewed_pages'])
            visual = {'status':'passed','visually_reviewed_pages':review['visually_reviewed_page_count'],'automatic_pages':review['automatic_page_count']}
    receipt = {'status':'passed','baseline_commit':commit,'source_edits':len(edits),'response_edits':len(answer_edits),
        'v3_abstract_only_specified_transport_correction':True,'initial_author_edit_intent_preserved':True,
        'scientific_numeric_values_and_order_preserved_outside_restored_abstract':True,
        'equations_algorithms_and_table_values_unchanged':True,'only_two_authorized_table_scope_cells_changed':True,
        'filter_rubric_literal_examples_and_figure_assets_unchanged':True,'experiment_code_data_and_prior_records_unchanged':True,
        'availability_submission_status_and_disclosure_decision_unchanged':True,'omitted_citation':manifest['removed_citation'],
        'quoted_review_blocks':quotes,'complete_answers':16,'compiled_labels_checked':len(locations),'bookmarks':navigation,
        'response_render_stable':True,'documents':documents,'editable_source_sha256':sha(PAPER/'main4_submission.tex'),
        'clean_dependency_build':clean,'visual_review':visual,'gpu_seconds':0}
    (OUT/'validation/checks.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['bookmarks','clean_dependency_build']},indent=2))


if __name__ == '__main__':
    main()
