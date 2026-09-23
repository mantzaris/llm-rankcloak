import argparse,json
from rankcloak.revision_v4_stage2_common import ROOT,read_json
from rankcloak.revision_v4_coherence import build_inventory
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/revision_v4/stage2_coherence.json');p.add_argument('--name',default='initial');p.add_argument('--exclude-payloads')
a=p.parse_args()
print(json.dumps(build_inventory(ROOT/a.config,a.name,read_json(a.exclude_payloads) if a.exclude_payloads else []),indent=2))
