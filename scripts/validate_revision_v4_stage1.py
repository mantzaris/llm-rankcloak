"""Verify preserved inputs, exact review quotes, and Stage 1 derived artifacts."""
import csv
import hashlib
import json
from pathlib import Path
import re

from rankcloak.reproducibility import sha256_file
from rankcloak.revision_v4_stage1 import write_json
from scripts.prepare_revision_v4_stage1 import escape

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision_v4'


def verify_quotes(original, ledger, response):
    if ''.join(b['text'] for b in ledger['blocks'])!=original:
        raise ValueError('review ledger does not preserve the complete original')
    expected=['editor_letter','reviewer2','reviewer1_general']+[f'R1.{i}' for i in range(1,11)]
    if [b['id'] for b in ledger['blocks']]!=expected:
        raise ValueError('missing or reordered review blocks')
    for b in ledger['blocks']:
        pattern=r'% BEGIN QUOTE '+re.escape(b['id'])+r'\n\\begin\{reviewcomment\}\n(.*?)\n\\end\{reviewcomment\}\n% END QUOTE '+re.escape(b['id'])
        found=re.findall(pattern,response,flags=re.S)
        if found!=[escape(b['text'].rstrip())]:
            raise ValueError('review quotation changed '+b['id'])
    if response.count('Direct answer pending.')!=16 or response.count('Evidence and change pending.')!=16 or response.count('Location pending.')!=16:
        raise ValueError('unresolved response placeholders missing')
    return len(expected)


def main():
    initial=json.loads((OUT/'provenance/initial_state.json').read_text())
    changed=[name for name,meta in initial['protected_files'].items() if sha256_file(ROOT/name)!=meta['sha256']]
    if changed: raise ValueError('protected inputs changed '+str(changed))
    if sha256_file(ROOT/'paperV4/RankCloak_V4_Revision_Plan.md')!=initial['author_plan_sha256']:
        raise ValueError('author plan changed')
    letter=ROOT/'paperV4/response/requests.txt'
    ledger=json.loads((ROOT/'paperV4/response/review_comments.json').read_text())
    if sha256_file(letter)!=ledger['source_sha256']: raise ValueError('request hash changed')
    count=verify_quotes(letter.read_text(),ledger,(ROOT/'paperV4/response/response_to_reviewers_v4.tex').read_text())
    manifest=json.loads((OUT/'provenance/offline_manifest.json').read_text())
    for name,digest in manifest['inputs'].items():
        if sha256_file(ROOT/name)!=digest: raise ValueError('offline input hash changed '+name)
    for name,digest in manifest['outputs'].items():
        if sha256_file(OUT/name)!=digest: raise ValueError('offline output hash changed '+name)
    failures=list(csv.DictReader((OUT/'source_tables/entropy_six_failures.csv').open()))
    positions=list(csv.DictReader((OUT/'source_tables/entropy_positions.csv').open()))
    selected=list(csv.DictReader((OUT/'source_tables/entropy_selected_runs.csv').open()))
    cfg=json.loads((ROOT/'configs/revision_v4/stage1_offline.json').read_text())
    if len(failures)!=6 or len(selected)!=21: raise ValueError('selected-run count mismatch')
    for row in failures:
        expected=cfg['expected_failure_suffixes'][row['plan_id'].split('__')[-1]]
        if [int(row[k]) for k in ['requested_ranks','consumed_ranks','tokens_used']]!=expected:
            raise ValueError('six-case table accounting mismatch')
        actual=[p for p in positions if p['plan_id']==row['plan_id']]
        if len(actual)!=int(row['tokens_used']) or sum(p['eligible']=='True' for p in actual)!=int(row['consumed_ranks']):
            raise ValueError('per-token denominator mismatch')
    docs=['paperV4/scientific_reports/main4.tex','paperV4/scientific_reports/supplementary4.tex','paperV4/response/response_to_reviewers_v4.tex','paperV4/cover_letter/cover_letter_v4.tex']
    for name in docs:
        text=(ROOT/name).read_text()
        if 'INTERNAL' not in text or 'pending' not in text.lower(): raise ValueError('draft marker missing '+name)
    source_filter=ROOT/'results/revision_v4/manuscript_tables/filter_methods.tex'
    if source_filter.read_bytes()!=(ROOT/'paperV4/scientific_reports/v4_filter_methods.tex').read_bytes(): raise ValueError('main filter specification differs from export')
    result={'status':'passed','protected_files_unchanged':len(initial['protected_files']), 'author_plan_unchanged':True,
            'review_blocks_exact':count,'pending_response_sets':16,'requests_sha256':sha256_file(letter),
            'selected_entropy_runs':len(selected),'position_rows':len(positions),'failure_count':len(failures),
            'offline_input_hashes_verified':len(manifest['inputs']),'document_draft_markers_verified':len(docs)}
    write_json(OUT/'provenance/stage1_validation.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
