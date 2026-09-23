"""Scientific status and commit-based archive edge cases."""
import copy,json,subprocess,zipfile
from pathlib import Path
import pytest
from rankcloak.revision_v4_archive import build,verify,safe_path,file_sha
from scripts.audit_revision_v4_stage3 import bounded_bytes,paired_effect
from scripts.validate_revision_v4_stage3 import validate_status
from scripts.validate_revision_v4_stage2 import verify_quotes

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def repo(tmp_path):
    root=tmp_path/'repo';root.mkdir();(root/'release/v4').mkdir(parents=True);(root/'scripts').mkdir()
    policy=json.loads((ROOT/'release/v4/assembly_policy.json').read_text())
    policy['required_paths']=['LICENSE','scripts/fixture.py']
    (root/'release/v4/assembly_policy.json').write_text(json.dumps(policy))
    (root/'LICENSE').write_text('fixture license\n');(root/'scripts/fixture.py').write_text('print(1)\n')
    (root/'scripts/nested.zip').write_bytes(b'excluded bundle')
    (root/'models').mkdir();(root/'models/weight.gguf').write_bytes(b'excluded weights')
    (root/'references').mkdir();(root/'references/paper.pdf').write_bytes(b'excluded third party')
    for args in [['init','-q'],['add','.'],['-c','user.name=Archive test','-c','user.email=test@example.invalid','commit','-qm','fixture']]:
        subprocess.run(['git','-C',str(root),*args],check=True)
    return root


def test_committed_bytes_determinism_and_portable_verification(repo,tmp_path):
    (repo/'scripts/fixture.py').write_text('uncommitted change\n')
    (repo/'scripts/untracked.py').write_text('must not enter\n')
    one=build(repo,'HEAD',tmp_path/'one');two=build(repo,'HEAD',tmp_path/'two')
    assert one['archive_sha256']==two['archive_sha256']
    receipt=verify(repo,tmp_path/'one/PACKAGE_MANIFEST.json',tmp_path/'extracted')
    assert receipt['status']=='passed'
    assert (tmp_path/'extracted/scripts/fixture.py').read_text()=='print(1)\n'
    assert not (tmp_path/'extracted/scripts/untracked.py').exists()
    assert not (tmp_path/'extracted/scripts/nested.zip').exists()
    assert not (tmp_path/'extracted/models').exists()
    assert not (tmp_path/'extracted/references').exists()
    with pytest.raises(ValueError,match='fresh extraction'):verify(repo,tmp_path/'one/PACKAGE_MANIFEST.json',tmp_path/'extracted')


@pytest.mark.parametrize('name',['../evil','/absolute','a/../../evil','a\\evil','a//b','./evil','a/./b','','.','C:/escape','C:escape'])
def test_unsafe_paths(name):
    with pytest.raises(ValueError):safe_path(name)


@pytest.mark.parametrize('mutation',['extra','missing','modified','duplicate'])
def test_tampered_zip_rejected_even_with_recomputed_outer_hash(repo,tmp_path,mutation):
    info=build(repo,'HEAD',tmp_path/'candidate');manifest=tmp_path/'candidate/PACKAGE_MANIFEST.json'
    archive=manifest.parent/info['archive_filename']
    with zipfile.ZipFile(archive) as z:members=[(i,z.read(i)) for i in z.infolist()]
    if mutation=='extra':members.append((zipfile.ZipInfo('unexpected'),b'new'))
    if mutation=='missing':members=members[1:]
    if mutation=='modified':members=[(i,b'wrong' if i.filename=='scripts/fixture.py' else b) for i,b in members]
    if mutation=='duplicate':members.append(members[-1])
    with zipfile.ZipFile(archive,'w') as z:
        for i,b in members:z.writestr(i,b)
    m=json.loads(manifest.read_text());m.update(archive_sha256=file_sha(archive),archive_bytes=archive.stat().st_size);manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError):verify(repo,manifest,tmp_path/'extracted')


def test_snapshot_tree_and_manifest_scope_cannot_be_relabelled(repo,tmp_path):
    build(repo,'HEAD',tmp_path/'candidate');p=tmp_path/'candidate/PACKAGE_MANIFEST.json';m=json.loads(p.read_text());m['source_tree']='0'*40;p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='source tree'):verify(repo,p,tmp_path/'extracted')


def test_decoder_padding_semantics_and_invalid_rank():
    meta={'bits_per_symbol':3,'alphabet_size':8,'padding_bits':1,'original_byte_length':1}
    assert bounded_bytes([1,1,2],meta)==b'\x00' # discarded nonzero padding is not integrity
    with pytest.raises(ValueError,match='invalid bounded rank'):bounded_bytes([1,9,1],meta)


def test_nested_estimator_keeps_payload_weight_and_requires_pairs():
    rows=[]
    for payload,values in [('one',[0.,0.,0.]),('two',[4.])]:
        for i,v in enumerate(values):
            for arm,value in [('actual',v),('ordinary',0.)]:
                rows.append(dict(payload_name=payload,trial_id=payload,boundary_id=payload+str(i),arm=arm,score=value))
    assert paired_effect(rows,('actual','ordinary'),'score',{'one':'class','two':'class'})[0]==2
    with pytest.raises(ValueError,match='missing paired arm'):paired_effect(rows[:-1],('actual','ordinary'),'score',{'one':'class','two':'class'})


def test_written_completion_does_not_clear_empirical_or_external_requirements():
    ledger=json.loads((ROOT/'results/revision_v4/stage3/response_status.json').read_text());validate_status(ledger)
    for key,value in [('empirical_coverage','fully_satisfied'),('external_action','none')]:
        bad=copy.deepcopy(ledger);next(r for r in bad['response_sets'] if r['id']=='R1.4')[key]=value
        with pytest.raises(ValueError):validate_status(bad)
    bad=copy.deepcopy(ledger);bad['v4_public_deposit']=True
    with pytest.raises(ValueError):validate_status(bad)


def test_verbatim_review_rejects_even_small_editorial_change():
    original=(ROOT/'paperV4/response/requests.txt').read_text();ledger=json.loads((ROOT/'paperV4/response/review_comments.json').read_text());response=(ROOT/'paperV4/response/response_to_reviewers_v4.tex').read_text()
    assert verify_quotes(original,ledger,response)==13
    altered=response.replace('the distinction between payload-bearing spans and natural tails lacks perceptual validation','the distinction between payload-bearing spans and greedy tails lacks perceptual validation')
    with pytest.raises(ValueError,match='verbatim review changed'):verify_quotes(original,ledger,altered)
