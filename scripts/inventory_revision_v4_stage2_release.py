"""Prepare a local inventory only. This script has no publication capability."""
from pathlib import Path
import subprocess,csv
from rankcloak.revision_v4_stage2_common import ROOT,OUT,file_hash,atomic_json
paths=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
new=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
allowed=('configs/revision_v4/stage2','paperV4/','rankcloak/revision_v4_','results/revision_v4/stage2/','revision_docs/REVISION_V4_STAGE2','revision_docs/REVISION_V4_RELEASE_SPEC','scripts/','tests/test_revision_v4_stage2.py')
paths.update(p for p in new if p.startswith(allowed))
folder=OUT/'release';folder.mkdir(parents=True,exist_ok=True)
# The inventory and its metadata are outside their own payload inventory.
paths=sorted(p for p in paths if not p.startswith('results/revision_v4/stage2/release/') and (ROOT/p).is_file())
with (folder/'prospective_inventory.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['path','size_bytes','sha256'])
 for name in paths:w.writerow([name,(ROOT/name).stat().st_size,file_hash(ROOT/name)])
atomic_json(folder/'status.json',{'status':'local prospective inventory only, no archive assembled or published','assembly_source':'current tracked files plus Stage 2 deliverables, including author Stage 2 plan','base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'file_count':len(paths),'contains_paperV3':any(p.startswith('paperV3/') for p in paths),'contains_paperV4':any(p.startswith('paperV4/') for p in paths),'published_doi':'10.5281/zenodo.22555497','published_version':'2.0.0','v4_covered_by_published_doi':False,'concept_doi':'10.5281/zenodo.21987449','final_commit_assembly_pending':True,'inventory_sha256':file_hash(folder/'prospective_inventory.csv')})
print(len(paths))
