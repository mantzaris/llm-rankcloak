"""CUDA-only metric workers with immutable request keys and resumable caches."""
from __future__ import annotations
import argparse
import base64
import importlib.metadata
import os
from pathlib import Path
import signal
import time
import numpy as np
from rankcloak.revision_v4_stage2_common import ROOT,OUT,read_json,read_jsonl,atomic_json,immutable_json,write_jsonl,append_jsonl,file_hash,digest,load_cache

STOP=False

def stop_requested(signum,frame):
    global STOP
    STOP=True


def runtime_versions(kind):
    names=['numpy','llama-cpp-python','nvidia-cuda-runtime-cu12','nvidia-cublas-cu12'] if kind=='lm' else ['numpy','torch','transformers','tokenizers','safetensors']
    return {name:importlib.metadata.version(name) for name in names}


def scoring_contract(kind):
    cfg=read_json(ROOT/'configs/revision_v4/stage2_coherence.json')
    common={'version':'v4-stage2-scoring-v1','kind':kind,'source_sha256':file_hash(Path(__file__)),
            'numpy_float':'float64 log probabilities or float32 embeddings', 'context_reset':'full KV clear per request',
            'lm_optimization':cfg['lm_optimization'],'context_limit':cfg['lm_context_limit']}
    if kind=='semantic':common['model']=read_json(ROOT/'configs/revision_v4/stage2_semantic_encoder.json')
    else:common['models']=read_json(ROOT/'configs/revision_v3/generation_requirements.json')
    return common


def make_requests(units, output, metrics=('semantic_cosine','context_gain')):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    contracts={kind:scoring_contract(kind) for kind in ['semantic','lm']}
    requests={};maps=[]
    pins={a['model_id']:a['sha256'] for a in contracts['lm']['models']['artifacts']}
    for u in units:
        mapping={'unit_id':u['unit_id'],'requests':{}}
        if 'semantic_cosine' in metrics:
            for side in ['left','right']:
                request={'kind':'semantic','contract_sha256':digest(contracts['semantic']),'text':u[side]}
                key=digest(request);requests[key]=request;mapping['requests'][side]=key
        if 'context_gain' in metrics:
            for mode in ['conditional','reference']:
                request={'kind':'lm','contract_sha256':digest(contracts['lm']),'model_id':u['evaluator_model_id'],
                    'model_sha256':pins[u['evaluator_model_id']], 'context_ids':u['evaluator_inputs'][mode+'_context_ids'],
                    'target_ids':u['evaluator_inputs']['target_ids']}
                key=digest(request);requests[key]=request;mapping['requests'][mode]=key
        maps.append(mapping)
    write_jsonl(output/'requests.jsonl',[{'request_id':key,'request':r} for key,r in sorted(requests.items())])
    write_jsonl(output/'unit_requests.jsonl',maps)
    immutable_json(output/'scoring_contracts.json',contracts)
    return {'unique_requests':len(requests),'semantic_requests':sum(r['kind']=='semantic' for r in requests.values()),
            'lm_requests':sum(r['kind']=='lm' for r in requests.values()),'files':{p.name:file_hash(p) for p in output.glob('*.json*')}}


def encode_vectors(texts,tokenizer,model,torch):
    batch=tokenizer(texts,padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda')
    with torch.inference_mode():
        output=model(**batch).last_hidden_state
        mask=batch['attention_mask'].unsqueeze(-1).expand(output.size()).float()
        pooled=(output*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
        vectors=torch.nn.functional.normalize(pooled,p=2,dim=1)
    return vectors.cpu().numpy().astype('<f4'),batch['attention_mask'].sum(1).cpu().tolist()


def run_semantic(plan):
    import torch
    from transformers import AutoTokenizer,AutoModel
    config=read_json(ROOT/'configs/revision_v4/stage2_semantic_encoder.json')
    contracts=read_json(plan/'scoring_contracts.json');contract=digest(contracts['semantic'])
    if contracts['semantic']!=scoring_contract('semantic'):raise ValueError('semantic scoring source contract changed')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required, no CPU fallback')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for name,meta in config['files'].items():
        if file_hash(Path(config['local_path'])/name)!=meta['sha256']:raise ValueError('semantic model pin changed')
    tokenizer=AutoTokenizer.from_pretrained(config['local_path'],local_files_only=True)
    started=time.perf_counter();model=AutoModel.from_pretrained(config['local_path'],local_files_only=True).float().to('cuda').eval()
    torch.cuda.synchronize();load_seconds=time.perf_counter()-started
    fixture=['A library offers books and study rooms.','Volunteers water a community garden every morning.','Café visitors enjoy a quiet place to read.']
    batch,_=encode_vectors(fixture,tokenizer,model,torch)
    solo=np.vstack([encode_vectors([t],tokenizer,model,torch)[0] for t in fixture])
    delta=float(np.max(np.abs(batch-solo)))
    if delta>1e-5:raise ValueError('embedding batching fixture differs')
    cache_path=OUT/'raw'/'semantic_cache.jsonl';cache=load_cache(cache_path,contract)
    requests=[r for r in read_jsonl(plan/'requests.jsonl') if r['request']['kind']=='semantic' and r['request_id'] not in cache]
    token_count=0;completed=0;scoring_started=time.perf_counter()
    for offset in range(0,len(requests),64):
        if STOP:break
        chunk=requests[offset:offset+64];texts=[r['request']['text'] for r in chunk]
        before=time.perf_counter();vectors,lengths=encode_vectors(texts,tokenizer,model,torch);torch.cuda.synchronize()
        elapsed=time.perf_counter()-before
        original_lengths=[len(tokenizer(t,add_special_tokens=True,truncation=False)['input_ids']) for t in texts]
        for r,v,n,raw_n in zip(chunk,vectors,lengths,original_lengths):
            append_jsonl(cache_path,{**r,'contract_sha256':contract,'vector_f32_base64':base64.b64encode(v.tobytes()).decode(),
                'dimension':384,'token_count':n,'original_token_count':raw_n,'truncated':raw_n>256,'batch_size':len(chunk),'batch_wall_seconds':elapsed})
        completed+=len(chunk);token_count+=sum(lengths)
        print('semantic',completed,'/',len(requests),flush=True)
    atomic_json(plan/'semantic_execution.json',{'status':'checkpointed' if completed<len(requests) else 'complete','new_requests':completed,
        'cached_requests_before':len(cache),'token_count':token_count,'model_load_seconds':load_seconds,'scoring_wall_seconds':time.perf_counter()-scoring_started,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'batch_fixture_max_absolute_difference':delta,'runtime_versions':runtime_versions('semantic'),'contract_sha256':contract})
    return 75 if completed<len(requests) else 0


def score_ids(model,context,targets,batched):
    from rankcloak.model_io import evaluate_context,get_last_logits
    from rankcloak.rank_codec import token_log_probability
    if not context or not targets:raise ValueError('empty scoring request')
    if len(context)+len(targets)>512:raise ValueError('context limit')
    if batched:
        evaluate_context(model,context+targets[:-1])
        return [float(token_log_probability(model.scores[len(context)-1+i],token)) for i,token in enumerate(targets)]
    evaluate_context(model,context);values=[]
    for token in targets:
        values.append(float(token_log_probability(get_last_logits(model),token)));model.eval([token])
    return values


def load_model(path,batch):
    from rankcloak.model_io import load_llama_cpp_model,preload_pip_cuda_libraries,llama_cpp_gpu_offload_supported
    if batch==1:return load_llama_cpp_model(path,n_ctx=512,n_threads=4,n_gpu_layers=-1,logits_all=True,verbose=True)
    for name,value in {'CUDA_LAUNCH_BLOCKING':'1','GGML_CUDA_DISABLE_GRAPHS':'1','GGML_CUDA_DISABLE_FUSION':'1','GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}.items():os.environ.setdefault(name,value)
    preload_pip_cuda_libraries()
    if not llama_cpp_gpu_offload_supported():raise RuntimeError('CUDA backend required')
    from llama_cpp import Llama
    return Llama(model_path=str(path),n_ctx=512,n_threads=4,n_gpu_layers=-1,logits_all=True,verbose=True,n_batch=batch,n_ubatch=batch)


def validated_backend(model_id,path,contract):
    from rankcloak.model_io import make_context_token_ids,tokenize_bytes
    from rankcloak.revision_evaluator import score_conditioned_text
    profile_path=OUT/'provenance'/('scorer_'+model_id+'.json')
    if profile_path.exists():
        profile=read_json(profile_path)
        if profile['contract_sha256']!=digest(contract):raise ValueError('backend fixture contract changed')
        return load_model(path,profile['selected_batch']),profile
    prompt='Explain an everyday system using complete thoughts.'
    texts=['The library offers books and study rooms. Readers can ask the staff for help.',
           'Volunteers water the community garden every morning.\nThey grow fresh herbs and vegetables.',
           'Café visitors often choose a quiet place to read. The team will review its project milestones tomorrow.']
    serial=load_model(path,1);fixture=[]
    try:
        for text in texts:
            context=make_context_token_ids(serial,prompt);targets=list(map(int,tokenize_bytes(serial,text.encode(),add_bos=False)))
            established=score_conditioned_text(serial,prompt,text,512)
            logps=score_ids(serial,context,targets,False)
            if abs(np.mean(logps)-established['mean_log_probability'])>1e-8:raise ValueError('serial route differs from established scorer')
            fixture.append({'prompt':prompt,'text':text,'context_ids':context,'target_ids':targets,'serial_logps':logps,'established_mean_logp':established['mean_log_probability']})
    finally:serial.close()
    candidate=load_model(path,128);max_token=max_mean=0.
    for row in fixture:
        values=score_ids(candidate,row['context_ids'],row['target_ids'],True);row['batched_logps']=values
        max_token=max(max_token,float(np.max(np.abs(np.asarray(values)-row['serial_logps']))))
        max_mean=max(max_mean,abs(float(np.mean(values))-row['established_mean_logp']))
    limits=contract['lm_optimization'];passed=max_token<=limits['maximum_per_token_logp_difference_nats'] and max_mean<=limits['maximum_mean_logp_difference_nats']
    profile={'model_id':model_id,'contract_sha256':digest(contract),'selected_batch':128 if passed else 1,'batching_validated':passed,
             'maximum_token_difference_nats':max_token,'maximum_mean_difference_nats':max_mean,'fixtures':fixture,'runtime_versions':runtime_versions('lm')}
    immutable_json(profile_path,profile)
    if passed:return candidate,profile
    candidate.close();return load_model(path,1),profile


def run_lm(plan,model_id):
    contracts=read_json(plan/'scoring_contracts.json');contract=digest(contracts['lm'])
    if contracts['lm']!=scoring_contract('lm'):raise ValueError('LM scoring source contract changed')
    if importlib.metadata.version('llama-cpp-python')!='0.3.23':raise ValueError('pinned generation backend required')
    artifact=next(a for a in contracts['lm']['models']['artifacts'] if a['model_id']==model_id)
    path=ROOT/artifact['expected_path']
    if file_hash(path)!=artifact['sha256']:raise ValueError('evaluator model pin changed')
    cache_path=OUT/'raw'/('lm_cache_'+model_id+'.jsonl');cache=load_cache(cache_path,contract)
    requests=[r for r in read_jsonl(plan/'requests.jsonl') if r['request']['kind']=='lm' and r['request']['model_id']==model_id and r['request_id'] not in cache]
    started=time.perf_counter();model,profile=validated_backend(model_id,path,contracts['lm']);load_seconds=time.perf_counter()-started
    completed=0;token_count=0;scoring_started=time.perf_counter()
    try:
        for row in requests:
            if STOP:break
            r=row['request'];before=time.perf_counter();logps=score_ids(model,r['context_ids'],r['target_ids'],profile['selected_batch']>1)
            elapsed=time.perf_counter()-before
            append_jsonl(cache_path,{**row,'contract_sha256':contract,'log_probabilities':logps,'mean_log_probability':float(np.mean(logps)),
                'target_token_count':len(logps),'context_token_count':len(r['context_ids']),'scoring_seconds':elapsed,'batch':profile['selected_batch']})
            completed+=1;token_count+=len(r['context_ids'])+len(logps)
            if completed%25==0:print(model_id,completed,'/',len(requests),round(time.perf_counter()-scoring_started,2),flush=True)
    finally:model.close()
    atomic_json(plan/('lm_execution_'+model_id+'.json'),{'status':'checkpointed' if completed<len(requests) else 'complete',
        'new_requests':completed,'cached_requests_before':len(cache),'evaluated_context_and_target_tokens':token_count,
        'load_and_fixture_seconds':load_seconds,'scoring_seconds':time.perf_counter()-scoring_started,'profile_path':str((OUT/'provenance'/('scorer_'+model_id+'.json')).relative_to(ROOT)),
        'runtime_versions':runtime_versions('lm'),'contract_sha256':contract})
    return 75 if completed<len(requests) else 0


def main():
    if not os.environ.get('RANKCLOAK_STAGE2_GPU_JOB'):raise RuntimeError('Run through the persistent budget supervisor')
    signal.signal(signal.SIGUSR1,stop_requested)
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--kind',choices=['semantic','lm'],required=True);p.add_argument('--model-id')
    a=p.parse_args();plan=ROOT/a.plan
    return run_semantic(plan) if a.kind=='semantic' else run_lm(plan,a.model_id)

if __name__=='__main__':raise SystemExit(main())
