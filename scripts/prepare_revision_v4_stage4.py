"""Prepare the approved upload without publishing or altering scientific documents."""
from pathlib import Path
import argparse
import copy
import datetime as dt
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/revision_v4/stage4'
SOURCE = '7f48ae2ef2fb11201ffe7a5a14c1568a9811b4e0'
TREE = 'bcea80173eea5c98bcf588642f58e70edd42c573'
CONCEPT = '10.5281/zenodo.21987449'
ARCHIVE_DIR = ROOT / 'release_artifacts/v4/7f48ae2ef2fb'
FILES = {
    'llm-rankcloak-v4-7f48ae2ef2fb.zip': (367904213, '1aca84b5dd45208f14b93ff88cc4c18ac7af2c8c485468b5373a38ba1620b625'),
    'PACKAGE_MANIFEST.json': (2960592, 'a0f45ecd26de211e3108e7ba739ace4251bb019a2ba1844e12af823a84a7665d'),
    'PACKAGE_MANIFEST.csv': (2010833, '5224380ae240725a152efc44dce22934857706e863d59c7cec71f8938a8467aa'),
    'SHA256SUMS': (188, '5dc483baca52aa91aa0a3225513dbed27440f0f7d8ad55af178e718d819541dc'),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def local_files(directory=ARCHIVE_DIR):
    records = []
    for name, (size, digest) in FILES.items():
        path = directory / name
        if path.stat().st_size != size or sha(path) != digest:
            raise ValueError('Approved upload identity mismatch: ' + name)
        records.append({'filename': name, 'size_bytes': size, 'sha256': digest,
                        'local_path': str(path.relative_to(ROOT))})
    manifest = json.loads((directory / 'PACKAGE_MANIFEST.json').read_text())
    if (manifest['source_commit'], manifest['source_tree'], manifest['payload_count']) != (SOURCE, TREE, 8856):
        raise ValueError('Approved source identity mismatch')
    return records


def response_status():
    # Retain the exact written text until verified publication permits its update.
    prior = json.loads((ROOT / 'results/revision_v4/stage3/response_status.json').read_text())
    value = copy.deepcopy(prior)
    value.update(schema='rankcloak.response.stage4.v1', v4_public_deposit=False,
                 publication_status='pending_authenticated_publication',
                 author_decision='approved_by_stage4_user_instruction',
                 editor_judgment='unknown', journal_submission='not_performed',
                 document_edition='Stage 3 author-review snapshot, unchanged pending public verification')
    for row in value['response_sets']:
        row['author_decision'] = 'approved'
        row['editor_judgment'] = 'unknown'
        row['public_deposit'] = 'pending'
        row['external_action'] = ('publish_verify_v4_then_update_documents' if row['id'] == 'archive'
                                  else 'editor_judgment_after_author_submission')
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-date', default=dt.datetime.now(dt.timezone.utc).date().isoformat())
    a = p.parse_args()
    dt.date.fromisoformat(a.prepared_date)
    publication = OUT / 'publication'
    if (publication / 'PUBLIC_VERIFICATION_RECEIPT.json').exists():
        raise ValueError('Publication already verified. Do not reset its status to prepared.')
    files = local_files()
    tree = subprocess.check_output(['git', 'rev-parse', SOURCE + '^{tree}'], cwd=ROOT, text=True).strip()
    if tree != TREE:
        raise ValueError('Source tree mismatch')
    description = (ROOT / 'release/v4/STAGE4_PUBLIC_DESCRIPTION.html').read_text().strip()
    proposal = json.loads((ROOT / 'release/v4/version_metadata.json').read_text())
    # Official Deposit API schema, distinct from the local candidate schema.
    metadata = {
        'title': proposal['title'], 'upload_type': 'software', 'version': '3.0.0',
        'publication_date': a.prepared_date, 'description': description,
        'creators': [{'name': 'Mantzaris, Alexander V.', 'affiliation': 'University of Central Florida',
                      'orcid': '0000-0002-0026-5725'}],
        'access_right': 'open', 'license': 'mit-license',
        'keywords': ['language models', 'rank transcoding', 'synthetic cryptographic artifacts', 'reproducible research'],
        'related_identifiers': [
            {'identifier': 'https://github.com/mantzaris/llm-rankcloak', 'relation': 'isSupplementTo', 'scheme': 'url'},
            {'identifier': 'https://github.com/mantzaris/llm-rankcloak/tree/' + SOURCE,
             'relation': 'isDerivedFrom', 'scheme': 'url'}],
        'notes': 'MIT applies to repository code. Third-party exceptions are documented in release/v4/THIRD_PARTY_NOTICE.md. The archived candidate is 3.0.0-rc1. Later DOI-updated submission documents are outside this immutable snapshot.',
    }
    write_json(publication / 'prepared_metadata.json', {'metadata': metadata})
    write_json(publication / 'publication_manifest.json', {
        'schema': 'rankcloak.stage4.publication-manifest.v1', 'status': 'prepared_not_submitted',
        'prepared_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'source_commit': SOURCE, 'source_tree': TREE, 'concept_doi': CONCEPT,
        'intended_version': '3.0.0', 'internal_candidate_version': '3.0.0-rc1',
        'files': files, 'prepared_metadata': metadata, 'submitted_metadata': None,
        'date_rule': 'Set publication_date to the actual publication date. Preparation is not publication.',
        'workflow': 'New version of record 22555497 under the existing concept. Check current latest and drafts first.',
    })
    write_json(OUT / 'response_status.json', response_status())
    write_json(publication / 'local_identity_verification.json', {
        'status': 'passed', 'checked_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'source_commit': SOURCE, 'source_tree': TREE, 'files': files,
        'zip_rebuilt': False, 'zip_modified': False,
        'prior_independent_verification': 'results/revision_v4/stage3/release/VERIFICATION_RECEIPT.json',
    })
    print('Verified all four approved upload files. Metadata and manifest prepared, not submitted.')


if __name__ == '__main__':
    main()
