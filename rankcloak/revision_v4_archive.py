"""Commit-based V4 archive assembly and independent extraction verification.

Unlike the historical release tool, provenance is not rewritten to remove local
paths. Frozen evidence bytes are preserved. This tool has no upload capability.
"""
import argparse,csv,hashlib,json,shutil,stat,subprocess,tarfile,zipfile
from pathlib import Path,PurePosixPath

POLICY='release/v4/assembly_policy.json'


def sha(data):return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def json_bytes(value):return (json.dumps(value,indent=2,sort_keys=True)+'\n').encode()


def safe_path(name):
    p=PurePosixPath(name)
    if not name or not p.parts or p.is_absolute() or '\\' in name or '\x00' in name or ':' in p.parts[0] or '..' in p.parts or '.' in p.parts or p.as_posix()!=name:
        raise ValueError('unsafe archive path '+repr(name))
    return p


def included(name,policy):
    p=safe_path(name)
    if name not in policy['include_files'] and p.parts[0] not in policy['include_roots']:return False
    if any(name.startswith(x) for x in policy['exclude_prefixes']):return False
    if set(p.parts)&set(policy['exclude_components']) or p.name in policy['exclude_names']:return False
    if any(name.lower().endswith(x) for x in policy['exclude_suffixes']):return False
    # Only the primary-source manifest may be redistributed from this audit folder.
    if '/provenance/primary_sources/' in name and p.name!='manifest.json':return False
    return True


def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args])


def snapshot(root,commit):
    commit=git(root,'rev-parse',commit+'^{commit}').decode().strip()
    tree=git(root,'rev-parse',commit+'^{tree}').decode().strip()
    policy_raw=git(root,'show',commit+':'+POLICY);policy=json.loads(policy_raw)
    entries={};excluded=[]
    for raw in git(root,'ls-tree','-rlz',commit).split(b'\0'):
        if not raw:continue
        meta,name=raw.split(b'\t',1);mode,kind,oid,size=meta.decode().split();name=name.decode()
        if included(name,policy):
            if mode not in ['100644','100755'] or kind!='blob':raise ValueError('nonregular source '+name)
            entries[name]={'git_blob':oid,'size_bytes':int(size),'mode':mode}
        else:excluded.append(name)
    missing=set(policy['required_paths'])-set(entries)
    if missing:raise ValueError('missing required committed paths '+str(sorted(missing)))
    return commit,tree,policy,sha(policy_raw),entries,excluded


def build(root,commit,directory):
    root=Path(root).resolve();directory=Path(directory).resolve()
    if directory.exists():raise ValueError('candidate directory already exists')
    commit,tree,policy,policy_hash,entries,excluded=snapshot(root,commit)
    directory.mkdir(parents=True)
    source={'schema':'rankcloak.archive-source.v4.1','source_commit':commit,'source_tree':tree,'policy_path':POLICY,
            'policy_sha256':policy_hash,'candidate_version':policy['candidate_version'],
            'published_doi':policy['published_doi'],'v4_public_deposit':False,
            'scope':'Committed files selected by the included assembly policy. Author-review snapshot, not a journal submission.'}
    metadata=json_bytes(source)
    archive=directory/('llm-rankcloak-v4-'+commit[:12]+'.zip')
    manifest=[];seen=set()
    # git archive supplies committed bytes only. Worktree and untracked files cannot enter.
    proc=subprocess.Popen(['git','-C',str(root),'archive','--format=tar',commit],stdout=subprocess.PIPE)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        # Git's tree walk is lexical within each directory; verify via named membership,
        # while metadata is a stable first member. No time or mutable path enters bytes.
        def add(name,data,mode,oid=None):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.create_system=3
            info.external_attr=(int(mode,8)<<16);info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
            manifest.append({'path':name,'size_bytes':len(data),'sha256':sha(data),'git_blob':oid,'mode':mode})
        add('ARCHIVE_SOURCE.json',metadata,'100644')
        with tarfile.open(fileobj=proc.stdout,mode='r|') as tar:
            for item in tar:
                if item.name not in entries:continue
                if not item.isfile():raise ValueError('nonregular tar member')
                data=tar.extractfile(item).read();entry=entries[item.name]
                if len(data)!=entry['size_bytes'] or hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=entry['git_blob']:
                    raise ValueError('committed blob mismatch '+item.name)
                add(item.name,data,entry['mode'],entry['git_blob']);seen.add(item.name)
        if proc.wait()!=0 or seen!=set(entries):raise ValueError('incomplete git archive')
    manifest.sort(key=lambda r:r['path'])
    result={**source,'archive_filename':archive.name,'archive_sha256':file_sha(archive),'archive_bytes':archive.stat().st_size,
            'payload_count':len(manifest),'payload_bytes':sum(r['size_bytes'] for r in manifest),'files':manifest,
            'excluded_committed_paths':excluded}
    (directory/'PACKAGE_MANIFEST.json').write_bytes(json_bytes(result))
    with (directory/'PACKAGE_MANIFEST.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','size_bytes','sha256','git_blob','mode'],lineterminator='\n');w.writeheader();w.writerows(manifest)
    (directory/'SHA256SUMS').write_text(result['archive_sha256']+'  '+archive.name+'\n'+file_sha(directory/'PACKAGE_MANIFEST.json')+'  PACKAGE_MANIFEST.json\n')
    return {k:v for k,v in result.items() if k not in ['files','excluded_committed_paths']}


def verify(root,manifest_path,extract):
    root=Path(root).resolve();manifest_path=Path(manifest_path).resolve();extract=Path(extract).resolve()
    m=json.loads(manifest_path.read_text());archive=manifest_path.parent/m['archive_filename']
    if Path(m['archive_filename']).name!=m['archive_filename']:raise ValueError('unsafe archive filename')
    if file_sha(archive)!=m['archive_sha256'] or archive.stat().st_size!=m['archive_bytes']:raise ValueError('outer archive hash mismatch')
    commit,tree,policy,policy_hash,entries,excluded=snapshot(root,m['source_commit'])
    if (tree,policy_hash)!=(m['source_tree'],m['policy_sha256']):raise ValueError('source tree mismatch')
    expected={r['path']:r for r in m['files']}
    if len(expected)!=len(m['files']) or set(expected)!=set(entries)|{'ARCHIVE_SOURCE.json'}:raise ValueError('manifest scope mismatch')
    if len(expected)!=m['payload_count'] or sum(r['size_bytes'] for r in expected.values())!=m['payload_bytes']:raise ValueError('manifest counts mismatch')
    if excluded!=m['excluded_committed_paths']:raise ValueError('exclusion list mismatch')
    for name,r in entries.items():
        if any(expected[name][k]!=r[k] for k in ['git_blob','mode','size_bytes']):raise ValueError('snapshot entry mismatch '+name)
    if extract.exists():raise ValueError('verification requires a fresh extraction directory')
    extract.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(expected):raise ValueError('missing/extra/duplicate ZIP members')
        for info in z.infolist():
            safe_path(info.filename)
            if not stat.S_ISREG(info.external_attr>>16):raise ValueError('symlink or special ZIP member')
            if (info.external_attr>>16)!=int(expected[info.filename]['mode'],8):raise ValueError('ZIP mode mismatch')
            data=z.read(info);r=expected[info.filename]
            if len(data)!=r['size_bytes'] or sha(data)!=r['sha256']:raise ValueError('ZIP member mismatch')
            if info.filename in entries and hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=r['git_blob']:raise ValueError('source byte mismatch')
            target=extract/info.filename;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(data)
    # Re-read the filesystem independently of ZIP CRCs and the assembly loop.
    actual={p.relative_to(extract).as_posix() for p in extract.rglob('*') if p.is_file()}
    if actual!=set(expected):raise ValueError('extracted membership mismatch')
    for name,r in expected.items():
        if file_sha(extract/name)!=r['sha256']:raise ValueError('extracted checksum mismatch')
    source=json.loads((extract/'ARCHIVE_SOURCE.json').read_text())
    if source['source_commit']!=commit or source['source_tree']!=tree or source['policy_sha256']!=policy_hash:raise ValueError('embedded source metadata mismatch')
    for required in policy['required_paths']:
        if not (extract/required).is_file():raise ValueError('required license/evidence missing')
    return {'status':'passed','source_commit':commit,'source_tree':tree,'archive_sha256':m['archive_sha256'],
            'archive_bytes':m['archive_bytes'],'payload_count':len(expected),'missing_files':0,'extra_files':0,
            'unsafe_paths':0,'checksum_mismatches':0,'source_blob_mismatches':0,'required_files_present':True,
            'extraction_path':str(extract),'manifest_sha256':file_sha(manifest_path)}


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','verify']);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--commit');p.add_argument('--output',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--extract',type=Path)
    a=p.parse_args()
    if a.action=='build':result=build(a.root,a.commit,a.output)
    else:result=verify(a.root,a.manifest,a.extract)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
