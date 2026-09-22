"""Read-only source inventory for V4 follow-up analyses and archive coverage."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

from rankcloak.reproducibility import sha256_file
from rankcloak.revision_protocol import context_sha256
from rankcloak.revision_v4_stage1 import write_csv, write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision_v4'


def main():
    inputs={}
    boundaries=[]
    controls={}
    masks={}
    for path in sorted((ROOT/'results/revision_v1/primary_v2').glob('*/records.jsonl')):
        inputs[str(path.relative_to(ROOT))]=sha256_file(path)
        with path.open() as f:
            for line in f:
                r=json.loads(line)
                if r.get('record_type')=='ordinary_control' and r['control_view']=='full_message':
                    g=r['generation']
                    key=(r['model_id'],context_sha256(g['context_token_ids']))
                    controls.setdefault(key,[]).append({'control_id':r['control_id'],'payload_name':r['payload_name'],'token_count':len(g['token_ids'])})
                if r.get('record_type')!='rankcloak_trial': continue
                if r['token_filter']=='safe_text_filter_v1':
                    masks[r['model_id']]=r['allowed_token_mask']
                if not r['segmented']: continue
                for s in r['segments']:
                    prompt=s['prompt']
                    boundaries.append({'boundary_id':r['trial_id']+'__segment_'+str(s['segment_index']),
                        'trial_id':r['trial_id'],'payload_name':r['payload_name'],'payload_class':r['payload_class'],
                        'model_id':r['model_id'],'protocol_variant':r['protocol_variant'],'segment_index':s['segment_index'],
                        'context_sha256':s['context_sha256'],'prompt_template_id':prompt.get('prompt_id',prompt.get('template_id','')) if isinstance(prompt,dict) else '',
                        'forced_token_count':len(s['forced_token_ids']),'tail_token_count':len(s['tail_token_ids']),
                        'full_token_count':len(s['full_token_ids']),'source_path':str(path.relative_to(ROOT)),
                        'source_record_sha256':hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()})
    for b in boundaries:
        count=b['forced_token_count']+min(32,b['tail_token_count'])
        pool=controls.get((b['model_id'],b['context_sha256']),[])
        b['eligible_ordinary_prefixes']=sum(r['token_count']>=count for r in pool)
        b['structurally_eligible']=bool(b['forced_token_count']>0 and b['tail_token_count']>0 and b['eligible_ordinary_prefixes'])
    write_csv(OUT/'source_tables/boundary_inventory.csv',boundaries)
    write_json(OUT/'provenance/historical_filter_masks.json',masks)
    model_files=[]
    requirements=json.loads((ROOT/'configs/revision_v3/generation_requirements.json').read_text())
    for a in requirements['artifacts']:
        path=ROOT/a['expected_path']; exists=path.exists()
        digest=sha256_file(path) if exists else None
        model_files.append({**a,'present':exists,'observed_size_bytes':path.stat().st_size if exists else None,
                            'observed_sha256':digest,'pin_verified':exists and digest==a['sha256'] and path.stat().st_size==a['size_bytes']})
    write_json(OUT/'provenance/local_model_inventory.json',model_files)
    quant=[]
    raw=ROOT/'results/revision_v3/generation/raw/quantization'
    for p in sorted((raw/'qwen2_5_7b_instruct_q8_0').glob('*.json')):
        d=json.loads(p.read_text()); trace=d['q8_replay_of_historical_q4_path']; pair=d['q4_q8_same_path_distribution_comparison']
        q4_path=raw/'qwen2_5_7b_instruct_q4_k_m'/(d['paired_q4_replay_plan_id']+'.json')
        q4=json.loads(q4_path.read_text()); q4trace=q4['distribution_trace']
        if trace['context_token_ids']!=q4trace['context_token_ids'] or trace['observed_token_ids']!=q4trace['observed_token_ids']:
            raise ValueError('same-history comparison does not use identical contexts and observed tokens')
        n = len(trace['observed_token_ids'])
        for source_trace in [trace, q4trace]:
            if source_trace['position_count'] != n or any(len(source_trace[k]) != n for k in ['observed_ranks', 'greedy_token_ids']):
                raise ValueError('same-history position arrays disagree')
        changed = sum(a != b for a, b in zip(trace['observed_ranks'], q4trace['observed_ranks']))
        greedy_changed = sum(a != b for a, b in zip(trace['greedy_token_ids'], q4trace['greedy_token_ids']))
        if (n, changed, greedy_changed) != (pair['position_count'], pair['observed_token_rank_changed_count'], pair['greedy_token_changed_count']):
            raise ValueError('retained comparison disagrees with paired trace arrays')
        quant.append({'plan_id':d['plan_id'],'population':d['population'],'source_path':str(p.relative_to(ROOT)),
            'source_sha256':sha256_file(p),'q4_source_path':str(q4_path.relative_to(ROOT)),'q4_source_sha256':sha256_file(q4_path),
            'position_count':pair['position_count'],'rank_changed_count':pair['observed_token_rank_changed_count'],
            'greedy_changed_count':pair['greedy_token_changed_count'],
            'historical_q4_path_identical':True,'raw_logit_vectors_retained':False})
    if len(quant)!=1920: raise ValueError('expected 1920 paired Q4/Q8 paths')
    write_csv(OUT/'source_tables/quantization_same_history_audit.csv',quant)
    totals={'segmented_trials':len({r['trial_id'] for r in boundaries}), 'payload_groups':len({r['payload_name'] for r in boundaries}),
            'boundaries':len(boundaries),'structurally_eligible':sum(r['structurally_eligible'] for r in boundaries),
            'boundaries_without_ordinary_prefix':sum(not r['eligible_ordinary_prefixes'] for r in boundaries),
            'planned_transition_arms':3,'maximum_transition_scores':3*len(boundaries),
            'same_history_q4_q8_pairs':len(quant), 'same_history_positions':sum(r['position_count'] for r in quant),
            'same_history_rank_changes':sum(r['rank_changed_count'] for r in quant),
            'inputs':inputs,'source_sha256':sha256_file(Path(__file__))}
    write_json(OUT/'provenance/evidence_inventory.json',totals)
    archive=Path('/tmp/rankcloak-v4-public-archive.zip')
    if archive.exists():
        manifest=json.loads((OUT/'provenance/published_PACKAGE_MANIFEST.json').read_text())
        with zipfile.ZipFile(archive) as z:
            prefix='rankcloak-code-data-v1.0.0/'
            checked=[]
            for row in manifest['files']:
                data=z.read(prefix+row['path'])
                if len(data)!=row['size_bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
                    raise ValueError('published archive manifest content mismatch')
                checked.append(row['path'])
            sources=[]
            for row in manifest['files']:
                if row['path'].startswith(('rankcloak/','scripts/','configs/')):
                    current=ROOT/row['path']
                    sources.append({'path':row['path'],'archive_sha256':row['sha256'], 'current_sha256':sha256_file(current), 'same_bytes':sha256_file(current)==row['sha256']})
        write_csv(OUT/'source_tables/archive_source_comparison.csv',sources)
        write_json(OUT/'provenance/archive_integrity.json',{'listed_files_verified':len(checked),'zip_sha256':sha256_file(archive),
            'method':'Verify size and SHA-256 of every embedded PACKAGE_MANIFEST entry against ZIP bytes.',
            'current_source_comparisons':len(sources),'different_current_sources':sum(not s['same_bytes'] for s in sources)})
    print(json.dumps({k:v for k,v in totals.items() if k!='inputs'},indent=2))


if __name__=='__main__': main()
