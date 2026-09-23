"""Check every reused transport endpoint at the Stage 2 context capacity.

Historical runs allocated 4096 context slots. The new bounded worker uses 512.
This verifies the retained reused ranks/errors on their exact short inputs rather
than assuming a change in allocation capacity preserves numerical results.
"""
import argparse,os,signal,time
from rankcloak.revision_v4_stage2_common import ROOT,OUT,read_json,read_jsonl,file_hash,append_jsonl,atomic_json
from rankcloak.model_io import load_llama_cpp_model
from rankcloak.token_filters import build_allowed_token_mask
from rankcloak.revision_protocol import recover_rank_span

p=argparse.ArgumentParser();p.add_argument('--model-id',required=True);a=p.parse_args()
if not os.environ.get('RANKCLOAK_STAGE2_GPU_JOB'):raise RuntimeError('budget supervisor required')
stop=False

def checkpoint(*args):
 global stop
 stop=True
signal.signal(signal.SIGUSR1,checkpoint)
folder=OUT/'transport';freeze=read_json(folder/'freeze.json');artifact=freeze['model_pins'][a.model_id];path=ROOT/artifact['expected_path']
if file_hash(path)!=artifact['sha256']:raise ValueError('model hash mismatch')
records=[r for r in read_jsonl(folder/'reused_replays.jsonl') if r['request']['model_id']==a.model_id]
output=folder/('reuse_validation_'+a.model_id+'.jsonl');seen={r['request_id'] for r in read_jsonl(output)} if output.exists() else set();todo=[r for r in records if r['request_id'] not in seen]
started=time.perf_counter();model=load_llama_cpp_model(path,n_ctx=512,n_threads=4,n_gpu_layers=-1,logits_all=True,verbose=True);completed=0
try:
 mask=build_allowed_token_mask(model,'safe_text_filter_v1')
 for old in todo:
  if stop:break
  r=old['request'];error=None;before=time.perf_counter()
  try:ranks=recover_rank_span(model,r['context_ids'],r['leadin_ids'],r['forced_ids'],mask)['ranks']
  except ValueError as exc:ranks=[];error=str(exc)
  prior_error=old['error'];normalized_old=prior_error.removeprefix('ValueError: ') if prior_error else None
  agreement=ranks==old['ranks'] and error==normalized_old
  append_jsonl(output,{'request_id':old['request_id'],'historical_work_id':old['historical_work_id'],'historical_context_capacity':4096,'stage2_context_capacity':512,'ranks':ranks,'error':error,'historical_rank_and_error_agreement':agreement,'model_evaluated_tokens':int(model.n_tokens),'elapsed_seconds':time.perf_counter()-before})
  completed+=1
finally:model.close()
allrows=read_jsonl(output);atomic_json(folder/('reuse_validation_summary_'+a.model_id+'.json'),{'planned_requests':len(records),'completed_requests':len(allrows),'all_agree':all(r['historical_rank_and_error_agreement'] for r in allrows),'elapsed_seconds_including_load':time.perf_counter()-started,'source_sha256':file_hash(__file__)})
if not all(r['historical_rank_and_error_agreement'] for r in allrows):raise ValueError('historical reuse differs at Stage 2 context capacity')
raise SystemExit(0 if len(allrows)==len(records) else 75)
