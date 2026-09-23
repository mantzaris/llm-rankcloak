"""Read-only public verification. Never creates, edits or publishes a Zenodo record."""
import argparse
import datetime as dt
import json
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from rankcloak.revision_v4_archive import verify
from scripts.prepare_revision_v4_stage4 import ROOT, OUT, SOURCE, TREE, CONCEPT, FILES, sha, write_json


class VisibleHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
    def handle_data(self, value):
        self.parts.append(value)


def prose(value):
    p = VisibleHTML()
    p.feed(value)
    return ' '.join(' '.join(p.parts).split())


def record_id(value):
    match = re.fullmatch(r'(?:https://zenodo.org/(?:api/)?records/|https://doi.org/10\.5281/zenodo\.|10\.5281/zenodo\.)?(\d+)/?', value)
    if not match:
        raise ValueError('Use an actual Zenodo record URL, version DOI or record number')
    return match[1]


def public_url(value):
    p = urllib.parse.urlsplit(value)
    if p.scheme != 'https' or p.hostname != 'zenodo.org' or p.username or p.password or p.port:
        raise ValueError('Unexpected public file host')
    if p.query or p.fragment:
        raise ValueError('Public file URL must not carry credentials or preview parameters')
    return value


def check_record(record, manifest):
    if record.get('state') != 'done' or record.get('submitted') is not True:
        raise ValueError('Record is not publicly published')
    if record.get('conceptdoi') != CONCEPT or str(record.get('conceptrecid')) != '21987449':
        raise ValueError('Wrong concept relationship')
    doi = record.get('doi', '')
    if not re.fullmatch(r'10\.5281/zenodo\.\d+', doi) or doi in (CONCEPT, '10.5281/zenodo.22555497', '10.5281/zenodo.21987450'):
        raise ValueError('No new version DOI verified')
    metadata = record['metadata']
    if metadata.get('doi') != doi or doi.rsplit('.', 1)[-1] != str(record.get('id')):
        raise ValueError('Returned record and version DOI identities disagree')
    expected = manifest['prepared_metadata']
    for key in ('title', 'version', 'access_right'):
        if metadata.get(key) != expected[key]:
            raise ValueError('Public metadata mismatch: ' + key)
    if metadata.get('resource_type', {}).get('type') != 'software':
        raise ValueError('Resource type is not software')
    if metadata.get('license', {}).get('id') != 'mit-license':
        raise ValueError('License differs from prepared metadata')
    creators = [{k: r.get(k) for k in ('name', 'affiliation', 'orcid')} for r in metadata.get('creators', [])]
    if creators != expected['creators']:
        raise ValueError('Creator identity differs from approved metadata')
    for key in ('description', 'notes'):
        if prose(metadata.get(key, '')) != prose(expected[key]):
            raise ValueError('Public prose differs: ' + key)
    if set(metadata.get('keywords', [])) != set(expected['keywords']):
        raise ValueError('Public keywords differ')
    identifiers = {(r['identifier'], r['relation']) for r in metadata.get('related_identifiers', [])}
    if identifiers != {(r['identifier'], r['relation']) for r in expected['related_identifiers']}:
        raise ValueError('Public source links differ')
    publication_date = dt.date.fromisoformat(metadata['publication_date'])
    earliest = dt.date.fromisoformat(expected['publication_date'])
    if not earliest <= publication_date <= dt.datetime.now(dt.timezone.utc).date():
        raise ValueError('Publication date must be actual, not future or before preparation')
    files = record.get('files', [])
    if len(files) != len(FILES) or {r['key'] for r in files} != set(FILES):
        raise ValueError('Missing, extra or obsolete public uploads')
    for row in files:
        if row['size'] != FILES[row['key']][0]:
            raise ValueError('Public upload size mismatch')
        public_url(row['links']['self'])
    return doi


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(public_url(url), headers={'User-Agent': 'RankCloak-public-verification/4'}), timeout=60) as r:
        public_url(r.url)
        return json.load(r)


def download(url, target, expected_size):
    request = urllib.request.Request(public_url(url), headers={'User-Agent': 'RankCloak-public-verification/4'})
    size = 0
    with urllib.request.urlopen(request, timeout=60) as r, target.open('xb') as f:
        final_url = public_url(r.url)
        for part in iter(lambda: r.read(1048576), b''):
            size += len(part)
            if size > expected_size:
                raise ValueError('Public download exceeds approved size')
            f.write(part)
    if size != expected_size:
        raise ValueError('Public download is incomplete')
    return final_url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', required=True)
    parser.add_argument('--downloads', type=Path, required=True, help='Fresh ignored directory for public downloads and extraction')
    args = parser.parse_args()
    identity = record_id(args.record)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    attempt = OUT / 'publication' / ('verification_' + identity + '_' + stamp)
    attempt.mkdir(parents=True)
    manifest = json.loads((OUT / 'publication/publication_manifest.json').read_text())
    try:
        record = get_json('https://zenodo.org/api/records/' + identity)
        write_json(attempt / 'public_record.json', record)
        if str(record['id']) != identity:
            raise ValueError('API returned a different record identity')
        doi = check_record(record, manifest)
        latest = get_json(record['links']['latest'])
        write_json(attempt / 'latest_record.json', latest)
        if str(latest['id']) != str(record['id']):
            raise ValueError('A later release exists. Reconcile it without publishing another version.')
        if args.downloads.exists():
            raise ValueError('Use a fresh download directory. Existing receipts and bytes are preserved.')
        args.downloads.mkdir(parents=True)
        downloads = []
        for item in record['files']:
            name = item['key']
            size, digest = FILES[name]
            target = args.downloads / name
            final_url = download(item['links']['self'], target, size)
            if sha(target) != digest:
                raise ValueError('Public bytes differ from the approved upload: ' + name)
            downloads.append({'filename': name, 'url': item['links']['self'], 'final_url': final_url,
                              'size_bytes': size, 'sha256': digest})
        archive = verify(ROOT, args.downloads / 'PACKAGE_MANIFEST.json', args.downloads / 'extracted')
        write_json(attempt / 'archive_verification.json', archive)
        # A reserved identifier is insufficient. Follow the actual returned DOI publicly.
        with urllib.request.urlopen(urllib.request.Request('https://doi.org/' + doi, headers={'User-Agent': 'RankCloak-public-verification/4'}), timeout=60) as response:
            final_url = response.url
            if response.status != 200 or urllib.parse.urlsplit(final_url).hostname != 'zenodo.org' or urllib.parse.urlsplit(final_url).path.rstrip('/') != '/records/' + str(record['id']):
                raise ValueError('Version DOI has not resolved to the verified public record')
        receipt = {'schema': 'rankcloak.stage4.public-verification.v1', 'status': 'passed',
                   'verified_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                   'public_record_url': record['links']['self_html'], 'version_doi': doi,
                   'concept_doi': CONCEPT, 'version': record['metadata']['version'],
                   'publication_date': record['metadata']['publication_date'],
                   'source_commit': SOURCE, 'source_tree': TREE, 'files': downloads,
                   'public_record_path': str((attempt / 'public_record.json').relative_to(ROOT)),
                   'public_record_sha256': sha(attempt / 'public_record.json'),
                   'observed_public_metadata': record['metadata'],
                   'prepared_manifest_sha256': sha(OUT / 'publication/publication_manifest.json'),
                   'archive_verification': archive, 'doi_final_url': final_url,
                   'authentication_used_for_verification': False,
                   'scientific_reproduction_reused': 'results/revision_v4/stage3/release/VERIFICATION_RECEIPT.json',
                   'reproduction_basis': 'Downloaded ZIP and every companion exactly match the approved bytes.',
                   'gpu_jobs': 0}
        write_json(attempt / 'receipt.json', receipt)
        # Preserve previous successful receipts rather than silently replacing one.
        final = OUT / 'publication/PUBLIC_VERIFICATION_RECEIPT.json'
        if final.exists():
            raise ValueError('A verification receipt already exists. New attempt retained without overwriting it.')
        write_json(final, receipt)
        print('Public archive verified. Version DOI ' + doi)
    except Exception as error:
        write_json(attempt / 'failure.json', {'status': 'pending_verification', 'error': str(error),
                                            'published_again': False, 'gpu_jobs': 0})
        raise


if __name__ == '__main__':
    main()
