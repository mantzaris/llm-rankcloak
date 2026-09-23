"""Recover continuation-policy provenance for the frozen boundary comparisons."""
from collections import Counter
from rankcloak.revision_v4_stage2_common import ROOT,OUT,read_jsonl,atomic_json,file_hash,digest

plan=OUT/'plans/coherence_study/units.jsonl'
units=read_jsonl(plan)
ordinary={r['right_source_id'] for r in units if r['arm']=='ordinary'}
actual={r['right_source_id'].split('__segment_')[0] for r in units if r['arm'] in ['actual','shuffled']}
controls=[];trials=[];hashes={}
for path in sorted((ROOT/'results/revision_v1/primary_v2').glob('*/records.jsonl')):
    hashes[str(path.relative_to(ROOT))]=file_hash(path)
    for row in read_jsonl(path):
        if row.get('control_id') in ordinary:
            g=row['generation']
            controls.append({'source_id':row['control_id'],'model_id':row['model_id'],
                'sampler':g['sampler'],'temperature':g['temperature'],'top_p':g['top_p']})
        if row.get('trial_id') in actual:
            trials.append({'source_id':row['trial_id'],'model_id':row['model_id'],'tail_policy':row['tail_policy']})
assert {r['source_id'] for r in controls}==ordinary and len(controls)==len(ordinary)
assert {r['source_id'] for r in trials}==actual and len(trials)==len(actual)
assert {(r['temperature'],r['top_p']) for r in controls}=={(.8,.95)}
result={'units_sha256':file_hash(plan),'source_hashes':hashes,
    'protocol_source_sha256':file_hash(ROOT/'rankcloak/revision_protocol.py'),
    'ordinary_source_identities':len(controls),'rankcloak_source_trials':len(trials),
    'ordinary_sampler_counts':dict(Counter(r['sampler'] for r in controls)),
    'ordinary_temperature':.8,'ordinary_top_p':.95,
    'rankcloak_tail_policies':list({digest(r['tail_policy']):r['tail_policy'] for r in trials}.values()),
    'implementation_evidence':'revision_protocol.py generate_rank_span tail loop chooses rank 1 under the declared filter.',
    'interpretation':'The ordinary comparison includes sampled continuations, while RankCloak tails are greedy. It does not isolate a causal effect of the forced boundary or establish reader-perceived coherence.',
    'control_sources':controls,'rankcloak_sources':trials}
atomic_json(OUT/'analysis/continuation_policy_audit.json',result)
print({k:result[k] for k in ['ordinary_source_identities','rankcloak_source_trials','ordinary_sampler_counts','rankcloak_tail_policies']})
