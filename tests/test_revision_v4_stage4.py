"""Publication gates and packaging invariants without network or model execution."""
import copy
import datetime as dt
import json
from pathlib import Path
import pytest
from scripts.prepare_revision_v4_stage4 import ROOT, OUT, FILES, CONCEPT, local_files, response_status
from scripts.verify_revision_v4_stage4_publication import check_record, record_id, public_url, prose
from scripts.build_revision_v4_stage4_bundle import dependency_check, manuscript_source
from scripts.validate_revision_v4_stage2 import verify_quotes


@pytest.fixture
def manifest():
    return json.loads((OUT / 'publication/publication_manifest.json').read_text())


@pytest.fixture
def published_fixture(manifest):
    # Synthetic record for schema tests only. No network call or claim of a real DOI.
    m = copy.deepcopy(manifest['prepared_metadata'])
    m['doi'] = '10.5281/zenodo.1'
    m['resource_type'] = {'type': m.pop('upload_type')}
    m['license'] = {'id': m['license']}
    return {'id': 1, 'state': 'done', 'submitted': True, 'doi': '10.5281/zenodo.1',
            'conceptdoi': CONCEPT, 'conceptrecid': '21987449', 'metadata': m,
            'files': [{'key': name, 'size': size, 'links': {'self': 'https://zenodo.org/api/records/1/files/' + name + '/content'}}
                      for name, (size, _) in FILES.items()]}


def test_published_fixture_has_required_contract(published_fixture, manifest):
    assert check_record(published_fixture, manifest) == '10.5281/zenodo.1'


@pytest.mark.parametrize('field,value', [('submitted', False), ('state', 'unsubmitted'),
                                        ('conceptdoi', '10.5281/zenodo.2'), ('conceptrecid', '2'),
                                        ('doi', CONCEPT), ('doi', '10.5281/zenodo.22555497')])
def test_reject_draft_old_or_wrong_concept(published_fixture, manifest, field, value):
    published_fixture[field] = value
    with pytest.raises(ValueError):
        check_record(published_fixture, manifest)


@pytest.mark.parametrize('field,value', [('version', '4.0.0'), ('title', 'different'),
                                        ('publication_date', '2999-01-01'),
                                        ('description', '<p>Semantic coherence established.</p>'),
                                        ('notes', ''), ('license', {'id': 'cc-by-4.0'}),
                                        ('resource_type', {'type': 'publication'}),
                                        ('creators', []), ('related_identifiers', [])])
def test_reject_changed_metadata(published_fixture, manifest, field, value):
    published_fixture['metadata'][field] = value
    with pytest.raises(ValueError):
        check_record(published_fixture, manifest)


@pytest.mark.parametrize('change', ['extra', 'missing', 'duplicate', 'size', 'host'])
def test_reject_wrong_public_files(published_fixture, manifest, change):
    rows = published_fixture['files']
    if change == 'extra': rows.append({'key': 'obsolete.zip', 'size': 1})
    elif change == 'missing': rows.pop()
    elif change == 'duplicate': rows[-1] = copy.deepcopy(rows[0])
    elif change == 'size': rows[0]['size'] += 1
    else: rows[0]['links']['self'] = 'https://example.org/file'
    with pytest.raises(ValueError):
        check_record(published_fixture, manifest)


@pytest.mark.parametrize('url', ['https://zenodo.org/records/22555497', 'https://doi.org/10.5281/zenodo.22555497',
                                 '10.5281/zenodo.22555497', '22555497'])
def test_record_identity_parser(url):
    assert record_id(url) == '22555497'


@pytest.mark.parametrize('url', ['https://example.org/records/22555497', 'https://zenodo.org/records/22555497?preview=1',
                                 'https://zenodo.org@evil.org/a', 'javascript:alert(1)'])
def test_record_identity_rejects_other_urls(url):
    with pytest.raises(ValueError): record_id(url)


@pytest.mark.parametrize('url', ['http://zenodo.org/file', 'https://zenodo.org/file?access_token=secret',
                                 'https://user@zenodo.org/file', 'https://zenodo.org:444/file'])
def test_public_download_never_uses_auth_or_other_host(url):
    with pytest.raises(ValueError): public_url(url)


def test_html_comparison_ignores_only_formatting():
    assert prose('<p>Ordinary <b>controls</b>.</p>') == prose('<div>Ordinary controls .</div>')
    assert prose('<p>not validated</p>') != prose('<p>validated</p>')


def test_archive_substitution_rejected(tmp_path):
    (tmp_path / next(iter(FILES))).write_bytes(b'wrong ZIP')
    with pytest.raises(ValueError, match='identity mismatch'): local_files(tmp_path)


def test_response_status_retains_partial_coverage_and_exact_answers():
    original = json.loads((ROOT / 'results/revision_v4/stage3/response_status.json').read_text())
    current = response_status()
    assert current['author_decision'] == 'approved_by_stage4_user_instruction'
    assert current['editor_judgment'] == 'unknown'
    assert current['written_complete'] == 16 and not current['v4_public_deposit']
    for a, b in zip(original['response_sets'], current['response_sets']):
        for key in ['direct_answer', 'evidence_change', 'rendered_location_text', 'empirical_coverage']:
            assert a[key] == b[key]
        if b['id'] in ('R1.4', 'editor_technical'): assert b['empirical_coverage'] == 'partial'


def test_exact_reviews():
    folder = ROOT / 'paperV4/response'
    assert verify_quotes((folder / 'requests.txt').read_text(), json.loads((folder / 'review_comments.json').read_text()),
                         (folder / 'response_to_reviewers_v4.tex').read_text()) == 13


def test_upload_source_matches_current_manuscript():
    folder = ROOT / 'paperV4/scientific_reports'
    assert (folder / 'main4_submission.tex').read_text().split('\n', 1)[1] == manuscript_source(folder)


def test_dependency_escape_rejected(tmp_path):
    fls = tmp_path / 'main.fls'
    fls.write_text('INPUT ../outside.sty\n')
    with pytest.raises(ValueError, match='outside bundle'): dependency_check(fls, tmp_path)


def test_dependency_local_and_system_allowed(tmp_path):
    fls = tmp_path / 'main.fls'
    fls.write_text('INPUT main.tex\nINPUT /usr/share/texlive/a.sty\n')
    assert dependency_check(fls, tmp_path) == 2


def test_inconsistent_doi_identity_rejected(published_fixture, manifest):
    published_fixture['metadata']['doi'] = '10.5281/zenodo.2'
    with pytest.raises(ValueError, match='identities disagree'):
        check_record(published_fixture, manifest)
