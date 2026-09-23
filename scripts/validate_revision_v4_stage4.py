"""Validate the pending Stage 4 handoff without weakening historical stage checks."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from scripts.prepare_revision_v4_stage4 import ROOT, OUT, SOURCE, sha, local_files, write_json
from scripts.prepare_revision_v4_stage1 import escape
from scripts.validate_revision_v4_stage2 import verify_quotes

# Only these existing administrative files may change before publication.
ADMINISTRATIVE = {'paperV4/ACTIONS_BEFORE_SUBMISSION.md', 'paperV4/REVIEW_RESPONSE_MATRIX.md',
                  'release/v4/UPLOAD_CHECKLIST.md', 'release/v4/version_metadata.json'}


def main():
    initial = json.loads((OUT / 'provenance/initial_state.json').read_text())
    changed = []
    preserved = 0
    for name, digest in initial['tracked_blobs'].items():
        path = ROOT / name
        if not path.is_file():
            raise ValueError('Pre-existing file removed: ' + name)
        raw = path.read_bytes()
        current = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if current == digest:
            preserved += 1
        else:
            changed.append(name)
            if name not in ADMINISTRATIVE:
                raise ValueError('Scientific, historical or unrelated file changed before publication: ' + name)
    response = (ROOT / 'paperV4/response/response_to_reviewers_v4.tex').read_text()
    quotes = verify_quotes((ROOT / 'paperV4/response/requests.txt').read_text(),
                          json.loads((ROOT / 'paperV4/response/review_comments.json').read_text()), response)
    ledger = json.loads((OUT / 'response_status.json').read_text())
    if ledger['written_complete'] != 16 or ledger['v4_public_deposit'] or ledger['empirical_R1_4'] != 'partial':
        raise ValueError('Pending publication or empirical limitation hidden')
    if ledger['author_decision'] != 'approved_by_stage4_user_instruction' or ledger['editor_judgment'] != 'unknown':
        raise ValueError('Approval status incorrect')
    rows = ledger['response_sets']
    if len(rows) != 16 or len({r['id'] for r in rows}) != 16:
        raise ValueError('Response IDs changed')
    for row in rows:
        block = re.findall(r'% BEGIN ANSWER ' + re.escape(row['id']) + r'\n(.*?)% END ANSWER ' + re.escape(row['id']), response, re.S)
        if len(block) != 1:
            raise ValueError('Missing response ' + row['id'])
        for key in ['direct_answer', 'evidence_change', 'rendered_location_text']:
            if escape(row[key]) not in block[0]:
                raise ValueError('Written answer and ledger differ: ' + row['id'])
        if row['written_response'] != 'complete':
            raise ValueError('Written response incomplete')
        if row['id'] in ['R1.4', 'editor_technical'] and row['empirical_coverage'] != 'partial':
            raise ValueError('Partial empirical coverage hidden')
        for name in row['evidence_paths']:
            if not (ROOT / name).is_file():
                raise ValueError('Evidence link missing: ' + name)
    uploads = local_files()
    manifest = json.loads((OUT / 'publication/publication_manifest.json').read_text())
    if manifest['status'] != 'prepared_not_submitted' or manifest['submitted_metadata'] is not None:
        raise ValueError('Prepared metadata presented as submitted')
    if manifest['files'] != uploads:
        raise ValueError('Publication manifest differs from approved uploads')
    if (OUT / 'publication/PUBLIC_VERIFICATION_RECEIPT.json').exists():
        raise ValueError('Public verification now exists. Finish the DOI-dependent stage instead of using pending validation.')
    mapping = json.loads((OUT / 'upload_map.json').read_text())
    if mapping['status'] != 'pending' or mapping['public_v4_verified']:
        raise ValueError('Pending bundle represented as final')
    bundle = Path(mapping['bundle_directory'])
    allowed = {'UPLOAD_MAP.md', 'UPLOAD_MAP.json'} | {r['filename'] for r in mapping['files']}
    members = {p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file()}
    if members != allowed:
        raise ValueError('Missing or extra bundle files')
    for row in mapping['files']:
        path = bundle / row['filename']
        if path.stat().st_size != row['size_bytes'] or sha(path) != row['sha256']:
            raise ValueError('Bundle identity mismatch')
        if sha(ROOT / row['source']) != row['sha256']:
            raise ValueError('Bundle copy is stale')
    builds = list((OUT / 'validation').glob('build_*/build_status.json'))
    if not builds:
        raise ValueError('No clean build receipt')
    result = json.loads(builds[-1].read_text())
    if result['status'] != 'passed' or len(result['builds']) != 5:
        raise ValueError('Incomplete build validation')
    if any(r['warnings'] for r in result['builds']):
        raise ValueError('Unresolved build warning')
    # Keep current document hashes and exact differences against the archive source.
    docs = []
    for name in initial['tracked_blobs']:
        if not name.startswith(('paperV4/scientific_reports/', 'paperV4/response/', 'paperV4/cover_letter/')):
            continue
        if Path(name).suffix not in ['.tex', '.pdf', '.bbl', '.cls', '.sty', '.bst', '.bib', '.ldf']:
            continue
        source = subprocess.check_output(['git', 'show', SOURCE + ':' + name], cwd=ROOT)
        current = ROOT / name
        docs.append({'path': name, 'size_bytes': current.stat().st_size, 'sha256': sha(current),
                     'archived_sha256': hashlib.sha256(source).hexdigest(),
                     'differs_from_archived_snapshot': source != current.read_bytes()})
    if any(r['differs_from_archived_snapshot'] for r in docs):
        raise ValueError('Document changed before public verification')
    write_json(OUT / 'validation/document_snapshot_comparison.json', {
        'source_commit': SOURCE, 'edition': 'Unchanged approved Stage 3 snapshot pending publication',
        'files': docs, 'document_byte_changes': 0, 'doi_updated_final_documents': False})
    final = {'status': 'passed_pending_external_publication', 'historical_files_unchanged': preserved,
             'existing_administrative_files_changed': changed, 'exact_review_blocks': quotes,
             'complete_written_answers': 16, 'R1_4_empirical_coverage': 'partial',
             'author_decision': 'approved', 'editor_judgment': 'unknown',
             'approved_upload_files_verified': 4, 'bundle_payload_files': len(mapping['files']),
             'clean_builds': 5, 'document_byte_changes_from_source': 0, 'gpu_jobs': 0,
             'public_v4_deposit_verified': False, 'final_upload_package_prepared': False,
             'pending_upload_bundle_prepared': True, 'journal_submission_performed': False}
    write_json(OUT / 'validation/stage4_validation.json', final)
    print(json.dumps(final, indent=2))


if __name__ == '__main__':
    main()
