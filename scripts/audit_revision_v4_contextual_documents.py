"""Check current numerical prose, document counts and reproducible build identities."""
from pathlib import Path
import json
import re
import subprocess
import tempfile
from rankcloak.revision_v4_contextual_coherence import ROOT, OUT
from rankcloak.revision_v4_stage2_common import read_json, atomic_json, file_hash
from scripts.prepare_revision_v4_stage3_handoff import flatten
from scripts.render_revision_v4_contextual_results import interval, stack


def main():
    paper=ROOT/'paperV4/scientific_reports'
    summary=read_json(OUT/'analysis/summary.json')
    panel=next(r for r in summary['summaries'] if r['group']=='panel')
    expanded=flatten(paper/'main4.tex')
    supplement=flatten(paper/'supplementary4.tex')
    abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',expanded,re.S)[1]
    title=re.search(r'\\title\{([^}]+)\}',expanded)[1]
    results=(paper/'v4_contextual_results.tex').read_text()
    table=(paper/'supplementary_tables/v4_contextual.tex').read_text()
    response=(ROOT/'paperV4/response/response_to_reviewers_v4.tex').read_text()
    cover=(ROOT/'paperV4/cover_letter/cover_letter_v4.tex').read_text()
    checks=[]
    for arm in ['encoded','ordinary']:
        for name,text in [('abstract',abstract),('results',results),('cover',cover),('response',response)]:
            assert f"{100*panel[arm]['estimate']:.1f}" in text
            checks.append(name+'_'+arm+'_acceptance')
        assert interval(panel[arm]) in results
    for field in ['difference','encoded_connectedness','ordinary_connectedness']:
        assert interval(panel[field],1 if 'connectedness' in field else 100,2 if 'connectedness' in field else 1) in results
        checks.append('results_'+field+'_interval')
    for row in summary['summaries']:
        if row['group'] in ['panel','generator']:
            for arm in ['encoded','ordinary','difference']:assert stack(row[arm]) in table
    for level in range(4):assert f"{panel['encoded_score_'+str(level)]['estimate']*100:.1f}" in results
    for field in summary['missing_sensitivity']:
        assert f"{summary['missing_sensitivity'][field]['estimate']*100:.1f}" in results
    missing=read_json(OUT/'analysis/missingness_details.json')
    assert (panel['complete_pairs'],missing['complete_trials'],panel['encoded']['payloads'])==(537,280,48)
    assert f"{panel['complete_pairs']} complete pairs in {missing['complete_trials']} trials" in results
    assert summary['invalid_unique_responses']==20 and summary['invalid_logical_scores']==43 and summary['retries']==31
    assert read_json(OUT/'validation/source_audit.json')['previously_ineligible_messages_retained']==85
    calibration=read_json(OUT/'analysis/calibration.json')
    for model,row in calibration['models'].items():
        assert row['acceptable_by_construction']['clear']['acceptable']==6
        assert row['acceptable_by_construction']['minor']['acceptable']==6
        assert row['acceptable_by_construction']['substantial']['acceptable']==(0 if model.startswith('qwen') else 1)
    examples=read_json(OUT/'analysis/examples.json')
    example_source=(paper/'v4_contextual_supplement.tex').read_text()
    for example in examples:
        assert example['text'] in example_source and example['message_id'] in example_source
    assert 'Local boundary compatibility and metric limitations' not in response
    assert '0.0078 [-0.0416, 0.0572]' in response and '-0.0106 [-0.0728, 0.0519]' in response
    assert 'Supplementary Fig.~S17' in expanded
    aux=(paper/'supplementary4.aux').read_text()
    assert r'\newlabel{fig:forced-full}{{S17}' in aux
    body=expanded[expanded.index(r'\section*{Introduction}'):expanded.index(r'\section*{Methods}')]+expanded[expanded.index(r'\section*{Results}'):expanded.index(r'\section*{Data Availability}')]
    words={}
    for key,text in [('body',body),('complete',expanded)]:
        with tempfile.NamedTemporaryFile(suffix='.tex',mode='w') as f:
            f.write(text);f.flush();log=subprocess.check_output(['texcount','-utf8',f.name],text=True)
        log=re.sub(r'File: /tmp/[^\n]+','File: expanded '+key,log)
        (OUT/'validation'/('texcount_'+key+'.txt')).write_text(log)
        words[key]={k:int(v) for k,v in re.findall(r'(Words in text|Words in headers|Words outside text \(captions, etc\.\)): (\d+)',log)}
    documents=[]
    for relative in ['paperV4/scientific_reports/main4','paperV4/scientific_reports/supplementary4','paperV4/response/response_to_reviewers_v4','paperV4/cover_letter/cover_letter_v4']:
        path=ROOT/(relative+'.pdf')
        text=subprocess.check_output(['pdftotext','-layout',str(path),'-'],text=True)
        normalized=' '.join(text.split()).replace('- ','')
        assert title in normalized
        assert text.count('V4 AUTHOR-REVIEW CANDIDATE')==1
        info=subprocess.check_output(['pdfinfo',str(path)],text=True)
        documents.append({'pdf':relative+'.pdf','pages':int(re.search(r'^Pages:\s+(\d+)',info,re.M)[1]),'bytes':path.stat().st_size,
            'pdf_sha256':file_hash(path),'source_sha256':file_hash(ROOT/(relative+'.tex'))})
    bibliography=(paper/'main4.bbl').read_text()
    cited={k for group in re.findall(r'\\cite\{([^}]+)\}',expanded) for k in group.split(',')}
    compiled=set(re.findall(r'\\bibitem\{([^}]+)\}',bibliography))
    assert cited<=compiled
    assert {'Liu2023GEval','VanDerLee2019BestPractices'}<=compiled
    assert len(abstract.split())<=200 and len(title.split())<=20
    assert expanded.count(r'\begin{figure}')==5 and expanded.count(r'\begin{table}')==3
    assert file_hash(paper/'figures/contextual_acceptability.pdf')==file_hash(OUT/'manuscript/contextual_acceptability.pdf')
    receipt={'status':'passed','documents':documents,'numerical_prose_checks':checks,
        'table_cells':'All acceptance and difference intervals for panel and generator rows match retained summaries',
        'additional_checks':['four disruption levels','all-slot missing bounds','complete and selected denominators','calibration class counts','exact example texts','historical semantic intervals retained','moved supporting figure reference'],
        'title_words':len(title.split()),'abstract_words_whitespace':len(abstract.split()),'word_counts_texcount':words,
        'body_word_definition':'Introduction, Results, Discussion and limitations. TeXcount text words exclude Abstract, Methods, availability, references, headings and captions. Mathematics is counted separately.',
        'main_figures':5,'main_tables':3,'main_algorithms_separate':expanded.count(r'\begin{algorithm}'),'main_references':len(compiled),
        'supplement_figures':supplement.count(r'\begin{figure}'),'supplement_tables':sum(supplement.count('\\begin{'+name+'}') for name in ['table','longtable']),
        'main_submission_source_sha256':file_hash(paper/'main4_submission.tex'),
        'independent_primary_estimate_check':'validation/score_audit.json','archived_prior_claim_check':'results/revision_v4/stage3/claim_evidence.json'}
    atomic_json(OUT/'validation/document_audit.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['documents','numerical_prose_checks']},indent=2))


if __name__=='__main__':main()
