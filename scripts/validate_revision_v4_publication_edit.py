"""Validate only the explicitly authorized V4 publication prose edits and builds.

The previous presentation validator and receipts are historical. This check starts
from the reviewed source commit and permits only the retained exact replacements,
punctuation repairs and regenerated document locations. No analysis is rerun.
"""
import csv
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
from scripts.validate_revision_v4_editorial_documents import old_text, old_flatten, environments, norm
from scripts.review_revision_v4_editorial_pdfs import inspect, DOCS
from scripts.build_revision_v4_stage4_bundle import manuscript_source

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/revision_v4/publication_edit'
PAPER = ROOT / 'paperV4/scientific_reports'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


# Same heading/destination checks as the preserved presentation workflow,
# with output directed exclusively to this editorial pass.
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
    validation = OUT/'validation'
    validation.mkdir(parents=True, exist_ok=True)
    baseline = load(OUT/'baseline.json')
    commit = baseline['head']
    manifest = load(OUT/'authorized_replacements.json')
    assert manifest['baseline_commit'] == commit == '303a2c4442992867fed0adeb4b12061cf001c1b7'
    entries = manifest['replacements']
    expected_ids = {f'M{i:02}' for i in range(1,23)} | {f'S{i:02}' for i in range(1,25)} | {f'P{i:02}' for i in range(1,6)}
    assert {r['id'] for r in entries} == expected_ids and len(entries) == 51
    assert not manifest['conflicts']
    repairs = manifest['additional_punctuation_repairs']
    for r in repairs:
        assert re.sub(r'\W','',r['old']) == re.sub(r'\W','',r['new']), r['id']
    allowed_edits = entries + repairs
    for r in allowed_edits:
        assert hashlib.sha256(r['old'].encode()).hexdigest() == r['old_sha256'], r['id']
        assert old_text(ROOT/r['path'],commit).count(r['old']) == 1, r['id']
    expected_sources = {}
    for relative in {r['path'] for r in allowed_edits}:
        expected = old_text(ROOT/relative,commit)
        for r in allowed_edits:
            if r['path'] == relative:
                assert expected.count(r['old']) == 1, r['id']
                expected = expected.replace(r['old'],r['new'],1)
        assert (ROOT/relative).read_text() == expected, 'Unauthorized prose change '+relative
        expected_sources[relative] = sha(ROOT/relative)

    # Exact file-scope protection covers all retained measurements and historical evidence.
    allowed = set(expected_sources) | set(DOCS) | {
        '.gitignore','paperV4/DOCUMENT_BUILD.md','paperV4/REVIEW_RESPONSE_MATRIX.md',
        'paperV4/scientific_reports/main4.bbl','paperV4/scientific_reports/supplementary4.bbl',
        'paperV4/scientific_reports/main4_submission.tex','paperV4/response/response_to_reviewers_v4.tex',
        'revision_docs/REVISION_V4_PUBLICATION_EDIT_REPORT.md',
        'scripts/build_revision_v4_editorial_documents.py','scripts/render_revision_v4_editorial_response.py',
        'scripts/review_revision_v4_editorial_pdfs.py','scripts/validate_revision_v4_publication_edit.py'}
    changed = subprocess.check_output(['git','diff','--name-only',commit],cwd=ROOT,text=True).splitlines()
    untracked = subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    unexpected = [p for p in set(changed+untracked) if p not in allowed and not p.startswith('results/revision_v4/publication_edit/')]
    assert not unexpected, 'Outside authorized scope '+str(unexpected)
    assert (ROOT/'paperV4/cover_letter/cover_letter_v4.tex').read_text() == old_text(ROOT/'paperV4/cover_letter/cover_letter_v4.tex',commit)
    assert (ROOT/'paperV4/response/editorial_answers.json').read_text() == old_text(ROOT/'paperV4/response/editorial_answers.json',commit)
    expanded = flatten(PAPER/'main4.tex'); supplement = flatten(PAPER/'supplementary4.tex')
    previous = old_flatten(PAPER/'main4.tex',commit); previous_supp = old_flatten(PAPER/'supplementary4.tex',commit)
    abstract = lambda text: re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',text,re.S)[1]
    assert abstract(expanded) == abstract(previous)
    assert hashlib.sha256(abstract(expanded).encode()).hexdigest() == baseline['abstract_sha256']
    for name in ['main4.tex','supplementary4.tex']:
        current=(PAPER/name).read_text();old=old_text(PAPER/name,commit)
        # The abstract is part of the preamble, as are typography and identities.
        assert current.split(r'\begin{document}')[0] == old.split(r'\begin{document}')[0]
    assert expanded[expanded.index(r'\section*{Data Availability}'):] == previous[previous.index(r'\section*{Data Availability}'):]
    assert (PAPER/'v4_filter_methods.tex').read_text() == old_text(PAPER/'v4_filter_methods.tex',commit)
    for environment in ['equation','align','algorithm','verbatim','Verbatim','quote']:
        assert environments(expanded,environment) == environments(previous,environment), ('main',environment)
        assert environments(supplement,environment) == environments(previous_supp,environment), ('supplement',environment)
    # Numeric table bodies and all plotted data are unchanged. The explicit scope-table
    # labels/cells and caption wording are checked by the exact source allowlist above.
    assert environments(expanded,'table') == environments(previous,'table')
    for current,old in [(expanded,previous),(supplement,previous_supp)]:
        assert re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',current) == re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',old)
        cite_keys=lambda text:{k for group in re.findall(r'\\cite\{([^}]+)\}',text) for k in group.split(',')}
        assert cite_keys(current) == cite_keys(old)
        labels=lambda text:set(re.findall(r'\\label\{([^}]+)\}',text))
        expected_added={'sec:revision-assistance'} if current==expanded else set()
        assert labels(current) == labels(old) | expected_added
    numbers=lambda text:set(re.findall(r'(?<![A-Za-z0-9])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?(?![A-Za-z0-9])',text))
    old_values=numbers(previous+'\n'+previous_supp);new_values=numbers(expanded+'\n'+supplement)
    assert old_values == new_values, ('Numeric value set changed',old_values-new_values,new_values-old_values)
    config=load(ROOT/'configs/revision_v4/coherence_replacement.json')
    rubric=(PAPER/'v4_contextual_supplement.tex').read_text()
    assert config['rubric'] in rubric
    for line in config['scales'].splitlines():
        if re.match(r'^\d = ',line):assert line.split(' = ',1)[1] in rubric
    assert config['scales'][config['scales'].index('For a very short'):] in rubric
    examples=load(ROOT/'results/revision_v4/coherence_replacement/analysis/examples.json')
    for example in examples:assert example['text'] in rubric and example['message_id'] in rubric
    summary=load(ROOT/'results/revision_v4/coherence_replacement/analysis/summary.json')
    panel=next(r for r in summary['summaries'] if r['group']=='panel')
    results=(PAPER/'v4_contextual_results.tex').read_text()
    numerical=[]
    for arm in ['encoded','ordinary','difference']:
        r=panel[arm];formatted=f"{r['estimate']*100:.1f} [{r['lower']*100:.1f}, {r['upper']*100:.1f}]"
        assert formatted in results
        numerical.append({'quantity':arm,'formatted':formatted,'source':'results/revision_v4/coherence_replacement/analysis/summary.json'})
    assert panel['complete_pairs']==537 and all(t in results for t in ['576 encoded messages','39 incomplete pairs','48 payloads'])
    assert (PAPER/'supplementary_tables/v4_pilots.tex').read_text() == old_text(PAPER/'supplementary_tables/v4_pilots.tex',commit)

    # S24 is a terminology repair verified against the retained code and validation record.
    validation_source=ROOT/'results/revision_v3/provenance/generation_analysis_validation.json'
    assert load(validation_source)['checks']['failure_record_count']==0
    analysis_path=ROOT/'scripts/analyze_revision_v3_generation.py'
    generation_path=ROOT/'rankcloak/revision_v3_generation.py'
    analysis=analysis_path.read_text();generation=generation_path.read_text()
    assert 'failure_files = sorted((generation_root / "failures").glob("**/*.json"))' in analysis
    assert 'checks["failure_record_count"] = len(failure_files)' in analysis
    assert re.search(r'except Exception as exc:\s+failure_count \+= 1\s+written = write_failure\(\s+failure_path, phase, model_id, row, exc, smoke',generation)
    failures_path=ROOT/'results/revision_v4/source_tables/entropy_six_failures.csv'
    failures=list(csv.DictReader(failures_path.open()))
    assert len(failures)==6 and all(r['payload_completion']=='False' for r in failures)
    assert [(int(r['requested_ranks']),int(r['consumed_ranks']),int(r['token_budget'])) for r in failures] == [(128,110,768),(128,116,768),(64,54,384),(128,106,768),(128,59,768),(128,116,768)]
    assert '114/120' in expanded and 'zero recorded execution exceptions' in supplement
    dump(validation/'s24_evidence.json',{'execution_exception_count':0,'retained_strict_gate_failures':6,'strict_completion_denominator':'114/120','sources':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in [validation_source,analysis_path,generation_path,failures_path]],'interpretation':'Exception-handler records and budget-exhausted payload outcomes are distinct; no generation rerun.'})

    ledger=load(OUT/'response_status.json');response_path=ROOT/'paperV4/response/response_to_reviewers_v4.tex';response=response_path.read_text()
    quotes=verify_quotes((ROOT/'paperV4/response/requests.txt').read_text(),load(ROOT/'paperV4/response/review_comments.json'),response)
    assert quotes==13 and len(ledger['response_sets'])==16 and ledger['written_complete']==16
    assert ledger['empirical_R1_4']=='automated_contextual_evidence_supplied_human_perception_unmeasured'
    assert not ledger['public_current_study_coverage'] and not ledger['journal_submission_performed']
    without_locations=lambda t:re.sub(r'(?m)^\\textbf\{Location\.\} .*$',r'LOCATION',t)
    assert without_locations(response)==without_locations(old_text(response_path,commit)), 'Substantive correspondence changed'
    for row in ledger['response_sets']:
        block=re.findall(r'% BEGIN ANSWER '+re.escape(row['id'])+r'\n(.*?)% END ANSWER '+re.escape(row['id']),response,re.S)
        assert len(block)==1 and row['written_response']=='complete'
        for key in ['direct_answer','evidence_change']:assert render_text(row[key]) in block[0]
        assert escape(row['rendered_location_text']) in block[0]
        for path in row['evidence_paths']:assert (ROOT/path).is_file(),path
        for anchor,location in row['compiled_locations'].items():
            aux=(PAPER/(location['document']+'.aux')).read_text()
            assert re.search(r'\\newlabel\{'+re.escape(anchor)+r'\}\{\{[^}]*\}\{'+str(location['page'])+r'\}',aux)
    repeat=load(validation/'response_render_repeat.json')
    assert repeat['identical'] and repeat['second_sha256']==sha(response_path)
    locations=load(validation/'locations.json')
    assert set(locations)==set(baseline['numbered_locations']) | {'sec:revision-assistance'}
    for key,old in baseline['numbered_locations'].items():
        if key.startswith(('eq:','fig:','tab:','alg:')):assert locations[key]['label']==old['label'],key
    assert locations['alg:tail']['label']=='S1' and 'Algorithm S1' in expanded
    nav=navigation(supplement)
    assert len(nav)==61
    documents=[]
    for relative in DOCS:
        path=ROOT/relative;text,pages,issues=inspect(path)
        assert not issues,(relative,issues)
        destinations=subprocess.check_output(['pdfinfo','-dests',str(path)],text=True)
        destination_names=set(re.findall(r'"([^"]+)"',destinations))
        objects=subprocess.check_output(['mutool','show','-g',str(path),'grep'],text=True)
        internal_links=set(re.findall(r'/D\(([^)]+)\)',objects))
        assert internal_links <= destination_names, (relative,'Unresolved PDF link',internal_links-destination_names)
        for marker in ['V4 AUTHOR-REVIEW CANDIDATE','Editorial author-review edition','editorial author-review PDFs','prepared for author review']:
            assert marker.lower() not in text.lower(),(relative,marker)
        if path.stem=='supplementary4':
            assert not re.search(r'^\s*(Reading guide|Contents)\s*$',text,re.M)
            assert 'Supplementary Note S1:' in text.split('\f')[0]
            for stale in ['Worked examples and claim-boundary audit','Claim-boundary audit','provenance and remaining boundaries']:
                assert stale.lower() not in text.lower(),stale
        (validation/(path.stem+'_text.txt')).write_text(text)
        documents.append({'path':relative,'pages':len(pages),'sha256':sha(path)})
    latest=load(validation/'latex/latest_builds.json')
    assert set(latest)=={'scientific','correspondence','submission'}
    for job in latest.values():
        for record in job['records']:
            assert not record['warnings'] and not record['unresolved_references']
            final_pass=3 if record['document'] in ['main4_submission','response_to_reviewers_v4','cover_letter_v4'] else 4
            final_log=(ROOT/job['logs']/(record['document']+f'_pass{final_pass}.txt')).read_text()
            assert not re.search(r'Missing character|multiply.defined|duplicate.*destination|destination with the same identifier',final_log,re.I)
            if record['document']!='main4_submission':
                target=next(r for r in documents if Path(r['path']).stem==record['document'])
                assert target['sha256']==record['pdf_sha256']
    submission=(PAPER/'main4_submission.tex').read_text()
    assert submission.split('\n',1)[1]==manuscript_source(PAPER)
    assert manuscript_source(PAPER)==manuscript_source(PAPER), 'Unstable generated submission source'
    assert abstract(submission)==abstract(expanded)
    assert environments(submission,'table')==environments(expanded,'table')
    assert (PAPER/'main4.bbl').read_text() in submission
    assert not re.search(r'\\(?:input|includegraphics|bibliography)\{',submission)
    assert 'author-review' not in submission.lower() and r'\tableofcontents' not in supplement
    compiled=set(re.findall(r'\\bibitem\{([^}]+)\}',(PAPER/'main4.bbl').read_text()))
    cited={k for group in re.findall(r'\\cite\{([^}]+)\}',expanded) for k in group.split(',')}
    assert cited<=compiled
    visual={'status':'separate visual review required'}
    if (validation/'pdf_review.json').exists():
        reviewed=load(validation/'pdf_review.json')
        if reviewed.get('visual_review_complete'):
            for row in reviewed['documents']:
                assert sha(ROOT/row['document'])==row['sha256'], 'Stale visual review'
                assert set(row['text_or_geometry_changed_pages']) <= set(row['visually_reviewed_pages'])
            visual={'status':'passed','visually_reviewed_pages':reviewed['visually_reviewed_page_count'],
                    'automatic_pages':reviewed['automatic_page_count'],'receipt':'validation/pdf_review.json'}
    receipt={'status':'passed','baseline_commit':commit,'applied_ids':[r['id'] for r in entries],'punctuation_only_ids':[r['id'] for r in repairs],
        'conflicts':[],'exact_authorized_source_match':expected_sources,'abstract_unchanged':True,'title_author_typography_unchanged':True,
        'availability_and_declarations_unchanged':True,'historical_evidence_and_scientific_code_unchanged':True,
        'equations_unchanged':{'main':len(environments(expanded,'equation')),'supplement':len(environments(supplement,'equation'))},
        'main_algorithms_unchanged':2,'supplementary_tail_algorithm_unchanged':True,'filter_rules_unchanged':True,
        'main_figures':5,'main_numeric_tables_unchanged':3,'numeric_value_set_unchanged':len(old_values),
        'judging_rubric_and_literal_examples_unchanged':True,'historical_pilots_unchanged':True,
        'quoted_review_blocks':quotes,'complete_unchanged_answers':16,'response_render_repeat':repeat,
        'locations_and_numbering_checked':len(locations),'navigation':nav,'numerical_claim_checks':numerical,
        'documents':documents,'editable_submission_sha256':sha(PAPER/'main4_submission.tex'),'main_references':len(compiled),
        'clean_submission_build':latest['submission'],'visual_review':visual,
        'additional_gpu_seconds':0,'experimental_changes':0,'changed_paths':changed}
    dump(validation/'publication_edit_check.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['navigation','changed_paths','clean_submission_build','exact_authorized_source_match','numerical_claim_checks']},indent=2))


if __name__=='__main__':
    main()
