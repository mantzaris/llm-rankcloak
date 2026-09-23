"""Bounded decomposition of the historical Markdown operation, without offset search."""
from __future__ import annotations
import argparse,difflib,os,signal,time
from collections import Counter
from pathlib import Path
from rankcloak.revision_v4_stage2_common import ROOT,OUT,read_json,read_jsonl,write_jsonl,atomic_json,immutable_json,file_hash,digest,hash_order,append_jsonl
from rankcloak.revision_v4_stage2_analysis import csv_write
from rankcloak.revision_protocol import text_to_token_ids,apply_transmission_transform,recover_rank_span,bounded_representation,decode_representation
from rankcloak.revision_v3_generation import payload_index
from rankcloak.revision_v4_quantization import byte_fields

ARMS=('prefix_only','outer_trim_only','trailing_trim_only','line_endings_only','declared_wrapper_removal')
STOP=False

def transform(text,arm):
    if arm=='prefix_only':return '\n'.join('> '+line for line in text.split('\n'))
    if arm=='outer_trim_only':return text.strip()
    if arm=='trailing_trim_only':return '\n'.join(line.rstrip() for line in text.split('\n'))
    if arm=='line_endings_only':return text.replace('\r\n','\n').replace('\r','\n')
    if arm=='declared_wrapper_removal':
        transported=apply_transmission_transform(text,'markdown_copy_paste')
        return remove_wrapper(transported)
    raise ValueError('unknown transport arm')

def remove_wrapper(text):
    lines=text.split('\n')
    if not all(line.startswith('> ') for line in lines):raise ValueError('declared wrapper is absent')
    return '\n'.join(line[2:] for line in lines)

def replay_key(model_id,filter_name,context,leadin,forced):
    return {'model_id':model_id,'filter':filter_name,'context_ids':context,'leadin_ids':leadin,'forced_ids':forced,
            'backend':'llama-cpp-python 0.3.23 pinned CUDA serial n_batch=1','source_sha256':file_hash(Path(__file__))}

def prepare():
    from rankcloak.revision_tokenizer_preflight import load_vocab_only_tokenizer
    folder=OUT/'transport';folder.mkdir(parents=True,exist_ok=True);pins={a['model_id']:a for a in read_json(ROOT/'configs/revision_v3/generation_requirements.json')['artifacts']}
    selected=[];unavailable=[];historical=[];sources={};input_hashes={}
    for p in sorted((ROOT/'results/revision_v1/robustness_v2').glob('*/records.jsonl')):
        records=read_jsonl(p);input_hashes[str(p.relative_to(ROOT))]=file_hash(p)
        candidates=[r for r in records if r.get('replay_mode')=='transformed_text_retokenized' and r.get('robustness_family')=='raw_transmission' and r.get('segment_outcomes') and all(x['source_text']==x['observed_text'] for x in r['segment_outcomes'])]
        for success in [True,False]:
            pool=[r for r in candidates if bool(r['exact_recovery'])==success]
            if not pool:unavailable.append({'model_id':p.parent.name,'visible_text_success':success});continue
            winner=min(pool,key=lambda r:(hash_order(20260922,r['source_trial_id']),r['work_id']))
            selected.append({'model_id':winner['model_id'],'source_trial_id':winner['source_trial_id'],'historical_visible_text_success':success,'historical_work_id':winner['work_id'],'source_stage':winner['source_stage'],'visible_baseline_evidence':'Executed historical text-transform replay with exact unchanged bytes in every segment. The historical unmodified reference points to saved-token replay and is not treated as visible-text evidence.'})
            hist=[r for r in records if r.get('source_trial_id')==winner['source_trial_id'] and r.get('model_id')==winner['model_id'] and r.get('model_id')==r.get('source_model_id') and (r['robustness_family']=='raw_transmission' and r['transformation_id'] in ['unmodified','markdown_copy_paste','line_endings','whitespace_trim','unicode_normalization',winner['transformation_id']])]
            if not {'unmodified','markdown_copy_paste'}.issubset({h['transformation_id'] for h in hist}):raise ValueError('historical arms missing')
            historical.extend(hist)
            source_path=ROOT/'results/revision_v1'/winner['source_stage']/winner['model_id']/'records.jsonl'
            input_hashes[str(source_path.relative_to(ROOT))]=file_hash(source_path)
            matches=[r for r in read_jsonl(source_path) if r.get('trial_id')==winner['source_trial_id'] and r.get('segments')]
            if len(matches)!=1 or digest(matches[0])!=winner['source_record_sha256']:raise ValueError('source lineage mismatch')
            sources[winner['source_trial_id']]=matches[0]
    if not selected:raise ValueError('no executed unchanged-text baselines')
    requests={};reused={};covers=[]
    for model_id in sorted({r['model_id'] for r in selected}):
        tokenizer=load_vocab_only_tokenizer(ROOT/pins[model_id]['expected_path'])
        try:
            for h in historical:
                if h['model_id']!=model_id or not h.get('segment_outcomes'):continue
                for s in h['segment_outcomes']:
                    request=replay_key(model_id,h['token_filter'],s['context_token_ids'],s['observed_leadin_token_ids'],s['observed_forced_token_ids']);key=digest(request)
                    reused[key]={'request_id':key,'request':request,'ranks':s['recovered_ranks'],'error':s['recovery_error'],'provenance':'historical','historical_work_id':h['work_id'],'segment_index':s['segment_index']}
            for selected_row in selected:
                if selected_row['model_id']!=model_id:continue
                source=sources[selected_row['source_trial_id']]
                for arm in ARMS:
                    segments=[]
                    for s in source['segments']:
                        text=s['full_text'];changed=transform(text,arm);ids=text_to_token_ids(tokenizer,changed);start=s['forced_start'];stop=s['forced_stop']
                        request=replay_key(model_id,source['token_filter'],s['context_token_ids'],ids[:start],ids[start:stop]);key=digest(request);requests[key]=request
                        old_ids=s['full_token_ids'];mismatch=next((i for i,(a,b) in enumerate(zip(old_ids,ids)) if a!=b),min(len(old_ids),len(ids)) if len(old_ids)!=len(ids) else None)
                        edits=[{'operation':tag,'source_bytes':[a,b],'changed_bytes':[c,d]} for tag,a,b,c,d in difflib.SequenceMatcher(None,text.encode(),changed.encode(),autojunk=False).get_opcodes() if tag!='equal']
                        segments.append({'segment_index':s['segment_index'],'request_id':key,'source_text':text,'observed_text':changed,
                            'source_text_sha256':__import__('hashlib').sha256(text.encode()).hexdigest(),'observed_text_sha256':__import__('hashlib').sha256(changed.encode()).hexdigest(),
                            'changed_byte_ranges':edits,'source_full_ids':old_ids,'observed_full_ids':ids,'first_token_difference':mismatch,
                            'saved_start':start,'saved_stop':stop,'expected_ranks':s['expected_ranks'],'source_forced_ids':s['forced_token_ids'],
                            'observed_forced_ids':ids[start:stop],'observed_leadin_ids':ids[:start]})
                    covers.append({**selected_row,'arm':arm,'cover_id':selected_row['source_trial_id']+'__'+arm,'payload_name':source['payload_name'],'payload_class':source['payload_class'],
                                   'source_record_sha256':digest(source),'segments':segments})
        finally:tokenizer.close()
    new=[{'request_id':key,'request':request} for key,request in sorted(requests.items()) if key not in reused]
    write_jsonl(folder/'selected_covers.jsonl',covers);write_jsonl(folder/'historical_outcomes.jsonl',historical);write_jsonl(folder/'reused_replays.jsonl',[v for k,v in sorted(reused.items()) if k in requests]);write_jsonl(folder/'requests.jsonl',new)
    model_counts={m:{'new_segment_replays':sum(r['request']['model_id']==m for r in new),'context_and_received_tokens':sum(len(r['request']['context_ids'])+len(r['request']['leadin_ids'])+len(r['request']['forced_ids']) for r in new if r['request']['model_id']==m)} for m in sorted({r['model_id'] for r in selected})}
    manifest={'seed':20260922,'selection':'Lowest SHA256(seed|source_trial_id) in each model by executed unchanged-visible-text success stratum. Use an executed historical text-transform replay only when every observed segment exactly equals its source bytes. Break duplicate-source ties by work ID. The unmodified reference aliases saved-ID success and is retained but not relabeled as visible-text execution. No replacement for unavailable strata.',
        'selected_source_covers':len(selected),'additional_cover_arms':len(covers),'constituent_segments':sum(len(r['segments']) for r in covers),
        'unique_new_segment_replays':len(new),'historical_reused_segment_requests':sum(k in requests for k in reused),'unavailable_strata':unavailable,
        'model_counts':model_counts,'inputs':input_hashes,'source_sha256':file_hash(Path(__file__)),'model_pins':{m:pins[m] for m in model_counts},
        'wrapper_rule':'Remove exactly the declared > space prefix from every line after the retained composite operation. No payload consultation or offset search.',
        'files':{p.name:file_hash(p) for p in folder.glob('*.jsonl')}}
    immutable_json(folder/'freeze.json',manifest);print(manifest['model_counts'])


def worker(model_id):
    from rankcloak.model_io import load_llama_cpp_model
    from rankcloak.token_filters import build_allowed_token_mask
    if not os.environ.get('RANKCLOAK_STAGE2_GPU_JOB'):raise RuntimeError('budget supervisor required')
    global STOP
    def stop(*args):
        global STOP
        STOP=True
    signal.signal(signal.SIGUSR1,stop)
    folder=OUT/'transport';freeze=read_json(folder/'freeze.json')
    if freeze['source_sha256']!=file_hash(Path(__file__)):raise ValueError('transport source changed after freeze')
    artifact=freeze['model_pins'][model_id];path=ROOT/artifact['expected_path']
    if file_hash(path)!=artifact['sha256']:raise ValueError('model pin changed')
    cachepath=folder/('replay_'+model_id+'.jsonl');cache={r['request_id']:r for r in read_jsonl(cachepath)} if cachepath.exists() else {}
    pending=[r for r in read_jsonl(folder/'requests.jsonl') if r['request']['model_id']==model_id and r['request_id'] not in cache]
    started=time.perf_counter();model=load_llama_cpp_model(path,n_ctx=512,n_gpu_layers=-1,n_threads=4,logits_all=True,verbose=True)
    completed=0
    try:
        mask=build_allowed_token_mask(model,'safe_text_filter_v1')
        for row in pending:
            if STOP:break
            r=row['request'];before=time.perf_counter();error=None
            try:values=recover_rank_span(model,r['context_ids'],r['leadin_ids'],r['forced_ids'],mask)['ranks']
            except ValueError as exc:values=[];error=str(exc)
            append_jsonl(cachepath,{**row,'ranks':values,'error':error,'provenance':'stage2_cuda_replay','elapsed_seconds':time.perf_counter()-before})
            completed+=1
    finally:model.close()
    atomic_json(folder/('execution_'+model_id+'.json'),{'new_segment_replays':completed,'model_and_replay_seconds':time.perf_counter()-started,'status':'complete' if completed==len(pending) else 'checkpointed'})
    return 0 if completed==len(pending) else 75


def analyze():
    folder=OUT/'transport';cache={}
    for p in [folder/'reused_replays.jsonl']+list(folder.glob('replay_*.jsonl')):
        for r in read_jsonl(p):
            if digest(r['request'])!=r['request_id']:raise ValueError('replay identity mismatch')
            if r['request_id'] in cache:raise ValueError('duplicate replay')
            cache[r['request_id']]=r
    payloads=payload_index();results=[];segments=[]
    for cover in read_jsonl(folder/'selected_covers.jsonl'):
        ranks=[]
        for s in cover['segments']:
            if s['request_id'] not in cache:raise ValueError('incomplete transport planned sample')
            replay=cache[s['request_id']];ranks.extend(replay['ranks']);a=s['expected_ranks'];b=replay['ranks']
            first=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)) if len(a)!=len(b) else None)
            segments.append({**{k:cover[k] for k in ['cover_id','source_trial_id','model_id','arm']},**s,'recovered_ranks':b,'recovery_error':replay['error'],'first_rank_difference':first,'replay_provenance':replay['provenance']})
        payload=payloads[cover['payload_name']];rep=bounded_representation(payload.payload_bytes,payload.payload_text,'hex_nibble');decoded=decode_representation(None,rep,ranks)
        results.append({k:v for k,v in cover.items() if k!='segments'}|{'segment_count':len(cover['segments']),'recovered_ranks':ranks,'decoded':byte_fields(decoded)})
    write_jsonl(folder/'cover_results.jsonl',results);write_jsonl(folder/'segment_diagnostics.jsonl',segments)
    csv_write(folder/'cover_summary.csv',({k:r[k] for k in ['cover_id','source_trial_id','model_id','arm','historical_visible_text_success','segment_count']}|{'exact_payload_recovery':r['decoded']['exact_payload_recovery'],'decode_success':r['decoded']['success']} for r in results))
    summary={'covers':len(results),'segments':len(segments),'by_arm':{arm:{'cases':sum(r['arm']==arm for r in results),'exact_recovery':sum(r['arm']==arm and r['decoded']['exact_payload_recovery'] for r in results)} for arm in ARMS},'population_rate_claim':False}
    atomic_json(folder/'summary.json',summary);print(summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','analyze']);p.add_argument('--model-id');a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='analyze':analyze()
    else:raise SystemExit(worker(a.model_id))
