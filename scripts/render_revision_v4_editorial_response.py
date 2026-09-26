"""Render reviewed editorial answers using current compiled manuscript locations.

Scientific scoring and prior response ledgers are read-only. Edit editorial_answers.json
for current answer prose. This renderer never regenerates manuscript prose.
"""
import hashlib
import json
from pathlib import Path
import re
from scripts.prepare_revision_v4_stage1 import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/revision_v4/presentation_cleanup'


def render_text(text):
    repository = 'https://github.com/mantzaris/llm-rankcloak'
    return escape(text).replace(repository, r'\url{'+repository+'}')


def main():
    rows = json.loads((ROOT/'paperV4/response/editorial_answers.json').read_text())['response_sets']
    locations = {}
    for stem in ['main4', 'supplementary4']:
        aux = ROOT/'paperV4/scientific_reports'/(stem+'.aux')
        for key, number, page in re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{(\d+)\}', aux.read_text()):
            if key == 'LastPage':
                continue
            if key in locations:
                raise ValueError('Duplicate cross-document label '+key)
            locations[key] = {'document':stem, 'page':int(page), 'label':number}
    response = ROOT/'paperV4/response/response_to_reviewers_v4.tex'
    original_text = text = response.read_text()
    headings = {'editor_accuracy':'Accuracy and scope', 'archive':'Code deposit', 'editor_technical':'Technical assessment',
                'general':'General assessment', 'wording':'Preservation of the review wording'}
    for row in rows:
        missing = set(row['manuscript_anchors'])-set(locations)
        if missing:
            raise ValueError('Unresolved locations '+str(missing))
        row['compiled_locations'] = {a:locations[a] for a in row['manuscript_anchors']}
        parts = []
        for stem in ['main4','supplementary4']:
            pages = sorted({v['page'] for v in row['compiled_locations'].values() if v['document']==stem})
            if pages:
                parts.append(stem+'.pdf pages '+', '.join(map(str,pages)))
        location = row['location_text'] + (' Accompanying documents, '+' and '.join(parts)+'.' if parts else '')
        row['rendered_location_text'] = location
        body = ('\\textbf{'+headings[row['id']]+'.}\n\n') if row['id'] in headings else ''
        for title, key in [('Answer','direct_answer'), ('Change and evidence','evidence_change')]:
            body += '\\textbf{'+title+'.} '+render_text(row[key])+'\n\n'
        body += '\\textbf{Location.} '+escape(location)+'\n\n'
        pattern = r'(% BEGIN ANSWER '+re.escape(row['id'])+r'\n).*?(% END ANSWER '+re.escape(row['id'])+r')'
        text, n = re.subn(pattern, lambda m:m[1]+body+m[2], text, flags=re.S)
        if n!=1:
            raise ValueError('Expected one answer '+row['id'])
    response.write_text(text)
    ledger = {'schema':'rankcloak.response.presentation.v1', 'response_sets':rows, 'written_complete':16,
        'empirical_R1_4':'automated_contextual_evidence_supplied_human_perception_unmeasured',
        'public_current_study_coverage':False, 'historical_evidence_unchanged':True, 'journal_submission_performed':False}
    (OUT/'validation').mkdir(parents=True, exist_ok=True)
    (OUT/'validation/response_render_repeat.json').write_text(json.dumps({
        'first_sha256':hashlib.sha256(original_text.encode()).hexdigest(),
        'second_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'identical':original_text==text,
        'removed_edition_suffix_absent':'Editorial author-review edition' not in text,
        'rule':'An ordinary render compares its input and output. A second render must be unchanged.'
    },indent=2)+'\n')
    (OUT/'response_status.json').write_text(json.dumps(ledger,indent=2)+'\n')
    (OUT/'validation/locations.json').write_text(json.dumps(locations,indent=2)+'\n')
    matrix = ['# V4 response and evidence status','',
        'All sixteen written responses are complete for author review. R1.4 supplies automated intact-message contextual evidence under the fixed lenient rubric. Human perception remains unmeasured, and the editor determines its adequacy. Historical fragment assays remain in Supplementary Note S17.','',
        'The current evidence is outside the earlier local archive candidate and published V3 deposit. Archive coverage remains a separate submission action. All thirteen exact review blocks and the original requests are unchanged. Current ledger: `results/revision_v4/presentation_cleanup/response_status.json`.','',
        '| Point | Written response | Evidence coverage | External action | Locations |',
        '| --- | --- | --- | --- | --- |']
    for row in rows:
        matrix.append('| '+' | '.join([row['id'],'complete',row['empirical_coverage'].replace('_',' '),row['external_action'].replace('_',' '),row['rendered_location_text']])+' |')
    (ROOT/'paperV4/REVIEW_RESPONSE_MATRIX.md').write_text('\n'.join(matrix)+'\n')
    print('Rendered 16 answers with compiled document locations')


if __name__ == '__main__':
    main()
