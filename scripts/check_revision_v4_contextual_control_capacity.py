"""Check the 54 frozen ordinary controls at historical context allocation.

This technical check never substitutes outputs or changes the frozen study.
"""
import argparse
import importlib.metadata
import os
import signal
import sys
import time
from rankcloak.revision_v4_contextual_coherence import ROOT,OUT,CONFIG
from rankcloak.revision_v4_stage2_common import read_json,read_jsonl,file_hash,digest,atomic_json,append_jsonl
STOP=False

def stop_requested(*_):
    global STOP
    STOP=True


def main():
    if not os.environ.get('RANKCLOAK_CONTEXTUAL_GPU_JOB'):raise RuntimeError('Use the cumulative GPU supervisor')
    signal.signal(signal.SIGUSR1,stop_requested)
    parser=argparse.ArgumentParser();parser.add_argument('--model-id',required=True);args=parser.parse_args()
    plan=read_json(OUT/'plans/control_capacity_check.json')
    cfg=read_json(CONFIG);pin=next(p for p in cfg['models'] if p['model_id']==args.model_id)
    assert file_hash(ROOT/pin['expected_path'])==pin['sha256']
    assert importlib.metadata.version('llama-cpp-python')==cfg['backend_version']
    from rankcloak.model_io import load_llama_cpp_model,make_context_token_ids
    from rankcloak.token_filters import build_allowed_token_mask
    from rankcloak.revision_protocol import generate_rank_span
    source=OUT/'raw/controls'/(args.model_id+'.jsonl')
    assert file_hash(source)==plan['control_hashes'][str(source.relative_to(ROOT))]
    rows=read_jsonl(source);dest=OUT/'raw/control_capacity'/source.name
    cached={r['request_id'] for r in read_jsonl(dest)} if dest.exists() else set()
    start=time.monotonic();model=load_llama_cpp_model(ROOT/pin['expected_path'],n_ctx=4096,n_threads=4,n_gpu_layers=-1,logits_all=True,verbose=True)
    loaded=time.monotonic()-start
    try:
        mask=build_allowed_token_mask(model,'safe_text_filter_v1')
        for row in rows:
            if STOP:break
            if row['request_id'] in cached:continue
            request=row['request'];assert make_context_token_ids(model,request['prompt'])==request['context_ids']
            generated=generate_rank_span(model,request['context_ids'],[],allowed_token_mask=mask,leadin_token_count=8,tail_policy='dynamic_completion_v1')
            original=row['result']['generation']
            fields=['full_token_ids','full_text','tail_stop_reason']
            comparisons={k:generated[k]==original[k] for k in fields}
            append_jsonl(dest,{'request_id':row['request_id'],'new_context_limit':4096,'original_context_limit':2048,
                'plan_sha256':file_hash(OUT/'plans/control_capacity_check.json'),'field_agreement':comparisons,
                'generation':generated,'filter_sha256':digest(mask.astype(int).tolist()),
                'filter_matches_original':digest(mask.astype(int).tolist())==row['result']['filter_sha256']})
    finally:model.close()
    result=read_jsonl(dest)
    atomic_json(OUT/'execution'/(os.environ['RANKCLOAK_CONTEXTUAL_GPU_JOB']+'.json'),{
        'phase':'control_capacity_check','model_id':args.model_id,'n_gpu_layers':-1,
        'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'backend_version':cfg['backend_version'],
        'load_and_hash_seconds':loaded,'worker_wall_seconds':time.monotonic()-start,'checks':len(result),
        'all_agree':all(all(r['field_agreement'].values()) and r['filter_matches_original'] for r in result),
        'study_outputs_replaced':False})
    print(args.model_id,len(result),'checks',all(all(r['field_agreement'].values()) for r in result))
    if len(result)!=len(rows):sys.exit(75)


if __name__=='__main__':main()
