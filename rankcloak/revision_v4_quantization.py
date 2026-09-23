"""Offline fixed-message Q4-to-Q8 decoding from retained common-history ranks."""
from collections import defaultdict
from pathlib import Path
import hashlib
from rankcloak.revision_v4_stage2_common import ROOT,OUT,read_json,read_jsonl,write_jsonl,atomic_json,file_hash,digest
from rankcloak.revision_v4_stage2_analysis import csv_write
from rankcloak.revision_v3_generation import payload_index,representation_from_source
from rankcloak.revision_protocol import decode_representation


def byte_fields(value):
    if isinstance(value,bytes):return {'hex':value.hex(),'length':len(value),'sha256':hashlib.sha256(value).hexdigest()}
    if isinstance(value,dict):return {k:byte_fields(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [byte_fields(v) for v in value]
    return value


def analyze():
    folder=OUT/'quantization';folder.mkdir(parents=True,exist_ok=True)
    source_path=ROOT/'results/revision_v1/primary_v2/qwen2_5_7b_instruct_q4_k_m/records.jsonl'
    sources={r['trial_id']:r for r in read_jsonl(source_path) if r['record_type']=='rankcloak_trial'}
    payloads=payload_index();rows=[];decode_rows=[];inputs={str(source_path.relative_to(ROOT)):file_hash(source_path)}
    base=ROOT/'results/revision_v3/generation/raw/quantization'
    for p in sorted((base/'qwen2_5_7b_instruct_q8_0').glob('*.json')):
        q8=read_json(p);p4=base/'qwen2_5_7b_instruct_q4_k_m'/(q8['paired_q4_replay_plan_id']+'.json');q4=read_json(p4)
        for source in [p,p4]:inputs[str(source.relative_to(ROOT))]=file_hash(source)
        if digest(q4)!=q8['paired_q4_replay_sha256']:raise ValueError('Q4 pairing hash mismatch')
        a=q4['distribution_trace'];b=q8['q8_replay_of_historical_q4_path']
        if a['context_token_ids']!=b['context_token_ids'] or a['observed_token_ids']!=b['observed_token_ids']:raise ValueError('not shared history')
        if q4['population']!=q8['population']:raise ValueError('population mismatch')
        n=len(a['observed_ranks']);changes=sum(x!=y for x,y in zip(a['observed_ranks'],b['observed_ranks']))
        if len(b['observed_ranks'])!=n or changes!=q8['q4_q8_same_path_distribution_comparison']['observed_token_rank_changed_count']:raise ValueError('rank denominator mismatch')
        plan=q8['plan_row'];row={'q8_plan_id':q8['plan_id'],'q4_plan_id':q4['plan_id'],'source_trial_id':q8['source_lineage']['rank_trial_id'],
            'population':q8['population'],'payload_name':plan['payload_name'],'payload_class':plan['payload_class'],'codec':plan['representation_name'],
            'positions':n,'observed_rank_changes':changes,'greedy_token_changes':sum(x!=y for x,y in zip(a['greedy_token_ids'],b['greedy_token_ids'])),
            'q4_input_sha256':file_hash(p4),'q8_input_sha256':file_hash(p)}
        rows.append(row)
        if row['population']!='rankcloak':continue
        source=sources[row['source_trial_id']];rep=representation_from_source(source,payloads)
        if hashlib.sha256(rep.payload_bytes).hexdigest()!=source['original_payload_sha256']:raise ValueError('original serialized payload hash mismatch')
        if list(rep.ranks)!=q4['expected_ranks'] or list(rep.ranks)!=a['observed_ranks']:raise ValueError('source representation mismatch')
        supported=rep.name in ['ascii_b8','ascii_b16'];ranks=b['observed_ranks']
        decoded=decode_representation(None,rep,ranks) if supported else {'success':False,'exact_payload_recovery':False,'error':'unsupported codec'}
        max_rank=8 if rep.name=='ascii_b8' else 16
        invalid=[i for i,r in enumerate(ranks) if r<1 or r>max_rank]
        decode_rows.append({**row,'direction':'Q4 fixed saved token path decoded with Q8 ranks','supported':supported,
            'invalid_rank_positions':invalid,'invalid_rank_count':len(invalid),'source_payload_hex':rep.payload_bytes.hex(),
            'q8_ranks_on_q4_path':ranks,'codec_metadata':rep.metadata,'decoded':byte_fields(decoded)})
    if (len(rows),sum(r['positions'] for r in rows),sum(r['observed_rank_changes'] for r in rows))!=(1920,244440,69528):raise ValueError('Stage 1 quantization totals changed')
    groups=defaultdict(list)
    for r in rows:
        for codec in [r['codec'],'all']:groups[(r['population'],codec)].append(r)
    summary=[]
    for (population,codec),group in sorted(groups.items()):
        dec=[r for r in decode_rows if r['population']==population and (codec=='all' or r['codec']==codec)]
        summary.append({'population':population,'codec':codec,'pairs':len(group),'positions':sum(r['positions'] for r in group),
            'observed_rank_changes':sum(r['observed_rank_changes'] for r in group),'greedy_token_changes':sum(r['greedy_token_changes'] for r in group),
            'decoding_trials':len(dec),'supported_decoding_trials':sum(r['supported'] for r in dec),
            'exact_payload_recoveries':sum(r['decoded'].get('exact_payload_recovery',False) for r in dec),
            'invalid_rank_trials':sum(bool(r['invalid_rank_count']) for r in dec),'invalid_rank_positions':sum(r['invalid_rank_count'] for r in dec),
            'decode_errors':sum(not r['decoded']['success'] for r in dec)})
    csv_write(folder/'shared_history_pairs.csv',rows);write_jsonl(folder/'q4_to_q8_fixed_message_decode.jsonl',decode_rows)
    csv_write(folder/'summary.csv',summary)
    csv_write(folder/'decode_endpoints.csv',({k:r[k] for k in ['q8_plan_id','q4_plan_id','source_trial_id','payload_name','payload_class','codec','supported','invalid_rank_count']}|{'exact_payload_recovery':r['decoded'].get('exact_payload_recovery',False),'decode_success':r['decoded']['success']} for r in decode_rows))
    atomic_json(folder/'manifest.json',{'inputs':inputs,'source_sha256':file_hash(Path(__file__)),'pairs':1920,'positions':244440,'observed_rank_changes':69528,'new_model_execution':False,
        'endpoint':'Bounded-codec original byte recovery using saved Q8 ranks on the historical Q4 received-token path. No Q8-to-Q4 inference.','summary':summary})
    print(summary)
if __name__=='__main__':analyze()
