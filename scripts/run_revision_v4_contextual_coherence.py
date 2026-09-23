"""Supervised CUDA calibration, matched controls and blinded intact-message judging."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import time
from rankcloak.revision_v4_contextual_coherence import OUT, CONFIG, judge_content, parse_score, SCHEMA
from rankcloak.revision_v4_stage2_common import ROOT, GPU, digest, file_hash, read_json, read_jsonl, append_jsonl, immutable_json, atomic_json, hash_order

STOP = False

def stop_requested(*_):
    global STOP
    STOP = True


def contract():
    names = ['rankcloak/revision_v4_contextual_coherence.py', 'scripts/run_revision_v4_contextual_coherence.py',
             'rankcloak/model_io.py', 'rankcloak/revision_protocol.py', 'rankcloak/token_filters.py', 'rankcloak/rank_codec.py']
    return {'config_sha256': file_hash(CONFIG), 'source_hashes': {n: file_hash(ROOT/n) for n in names},
            'backend_version': importlib.metadata.version('llama-cpp-python')}


def load(model_id, controls=False):
    from rankcloak.model_io import preload_pip_cuda_libraries, llama_cpp_gpu_offload_supported, load_llama_cpp_model
    cfg=read_json(CONFIG); pin=next(p for p in cfg['models'] if p['model_id']==model_id)
    if importlib.metadata.version('llama-cpp-python') != cfg['backend_version']: raise ValueError('backend pin changed')
    if file_hash(ROOT/pin['expected_path']) != pin['sha256']: raise ValueError('model pin changed')
    for key,value in {'CUDA_LAUNCH_BLOCKING':'1','GGML_CUDA_DISABLE_GRAPHS':'1','GGML_CUDA_DISABLE_FUSION':'1','GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}.items(): os.environ.setdefault(key,value)
    preload_pip_cuda_libraries()
    if not llama_cpp_gpu_offload_supported(): raise RuntimeError('CUDA required, CPU inference forbidden')
    from llama_cpp import Llama
    if controls:
        model=load_llama_cpp_model(ROOT/pin['expected_path'], n_ctx=2048,n_threads=4,n_gpu_layers=-1,logits_all=True,verbose=True)
    else:
        model=Llama(model_path=str(ROOT/pin['expected_path']),n_ctx=2048,n_threads=4,n_gpu_layers=-1,
                    n_batch=128,n_ubatch=128,logits_all=False,verbose=True,chat_format='chat_template.default')
    if model.metadata['tokenizer.chat_template'] != pin['chat_template']: raise ValueError('chat template pin changed')
    return model


def rendered_prompt(model, content):
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    formatter=Jinja2ChatFormatter(template=model.metadata['tokenizer.chat_template'],
        eos_token=model._model.token_get_text(model.token_eos()),bos_token=model._model.token_get_text(model.token_bos()),
        stop_token_ids=[model.token_eos()])
    return formatter(messages=[{'role':'user','content':content}]).prompt


def read_cache(path, contract_sha):
    rows=read_jsonl(path) if path.exists() else []
    if any(r['contract_sha256'] != contract_sha or digest(r['request']) != r['request_id'] for r in rows):
        raise ValueError('cache contract or request changed')
    if len({r['request_id'] for r in rows}) != len(rows): raise ValueError('duplicate cached request')
    return {r['request_id']:r for r in rows}


def requests_for(phase, model_id, plan):
    cfg=read_json(CONFIG)
    if phase=='controls':return [r for r in read_jsonl(plan/'controls.jsonl') if r['request']['generator']==model_id]
    if phase=='calibration':
        return [{'request_id':digest(request),'request':request} for row in read_json(OUT/'plans/calibration_cases.json')
            for request in [{'judge_model_id':model_id,'context':row['context'],'message':row['message'],
                             'config_sha256':file_hash(CONFIG)}]]
    freeze=read_json(plan/'freeze.json')
    for name,expected in freeze['files'].items():
        if file_hash(plan/name)!=expected:raise ValueError('frozen study plan changed')
    return [r for r in read_jsonl(plan/'judge_requests.jsonl') if r['request']['judge_model_id']==model_id]


def main():
    if not os.environ.get('RANKCLOAK_CONTEXTUAL_GPU_JOB'):raise RuntimeError('Use the six-hour supervised GPU runner')
    signal.signal(signal.SIGUSR1,stop_requested)
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['calibration','controls','judging'],required=True);p.add_argument('--model-id',required=True);p.add_argument('--plan',default='target_sample')
    a=p.parse_args();cfg=read_json(CONFIG);plan=OUT/'plans'/a.plan
    current=contract();immutable_json(OUT/'provenance/inference_contract.json',current);contract_sha=digest(current)
    cachepath=OUT/'raw'/a.phase/(a.model_id+'.jsonl');cache=read_cache(cachepath,contract_sha)
    requests=sorted(requests_for(a.phase,a.model_id,plan),key=lambda r:hash_order(cfg['seed'],'execution|'+r['request_id']))
    pending=[r for r in requests if r['request_id'] not in cache]
    started=time.monotonic();model=load(a.model_id,a.phase=='controls');loaded=time.monotonic()-started
    from rankcloak.model_io import make_context_token_ids
    if a.phase=='controls':
        from rankcloak.revision_protocol import build_revision_filter_mask,generate_rank_span
        mask=build_revision_filter_mask(model,'safe_text_filter_v1')
    completed=0
    try:
        for row in pending:
            if STOP:break
            request=row['request'];before=time.monotonic()
            if a.phase=='controls':
                if make_context_token_ids(model,request['prompt']) != request['context_ids']:raise ValueError('original context mismatch')
                generated=generate_rank_span(model,request['context_ids'],[],allowed_token_mask=mask,
                    leadin_token_count=request['greedy_initial_tokens'],tail_policy=request['tail_policy'])
                result={'generation':generated,'filter_sha256':digest(mask.astype(int).tolist()),'allowed_token_count':int(mask.sum()),
                        'context_token_count':len(request['context_ids']),'output_token_count':len(generated['full_token_ids']),
                        'eos_stop_rule':'no explicit EOS stop, unchanged historical mask and tail heuristic'}
            else:
                content=judge_content(request['context'],request['message']);rendered=rendered_prompt(model,content)
                rendered_ids=model.tokenize(rendered.encode(),add_bos=True,special=True)
                if len(rendered_ids)+cfg['judge_settings']['retry_max_tokens']>2048:raise ValueError('intact prompt would exceed context, do not truncate')
                attempts=[]
                for attempt,cap in enumerate([cfg['judge_settings']['max_tokens'],cfg['judge_settings']['retry_max_tokens']]):
                    model.reset();step=time.monotonic()
                    response=model.create_chat_completion(messages=[{'role':'user','content':content}],
                        temperature=0.0,top_p=1.0,top_k=0,repeat_penalty=1.0,seed=cfg['seed'],max_tokens=cap,
                        response_format={'type':'json_object','schema':SCHEMA})
                    raw=response['choices'][0]['message']['content'] or ''
                    parsed=parse_score(raw,request['message']);finish=response['choices'][0]['finish_reason']
                    if finish=='length':parsed={'parse_status':'invalid','parse_error':'output_truncated','acceptable':None}
                    attempts.append({'attempt':attempt+1,'cap':cap,'response':response,'raw_text':raw,'parsed':parsed,
                                     'elapsed_seconds':time.monotonic()-step})
                    if parsed['parse_status']=='valid':break
                result={'attempts':attempts,'parsed':attempts[-1]['parsed'],'judge_input':content,
                        'rendered_prompt':rendered,'rendered_prompt_sha256':digest(rendered),
                        'input_token_count':len(rendered_ids),'output_token_count':sum(t['response']['usage']['completion_tokens'] for t in attempts)}
            append_jsonl(cachepath,{**row,'contract_sha256':contract_sha,'result':result,'elapsed_seconds':time.monotonic()-before})
            completed+=1
            print(a.phase,a.model_id,completed,'/',len(pending),round(time.monotonic()-started,1),flush=True)
    finally:model.close()
    atomic_json(OUT/'execution'/(os.environ['RANKCLOAK_CONTEXTUAL_GPU_JOB']+'.json'),{
        'phase':a.phase,'model_id':a.model_id,'planned_unique_requests':len(requests),'cached_before':len(cache),
        'completed_new':completed,'status':'complete' if completed==len(pending) else 'checkpointed',
        'load_and_hash_seconds':loaded,'worker_wall_seconds':time.monotonic()-started,'contract_sha256':contract_sha,
        'gpu_uuid':GPU,'n_gpu_layers':-1,'backend_version':importlib.metadata.version('llama-cpp-python')})
    return 0 if completed==len(pending) else 75


if __name__=='__main__':sys.exit(main())
