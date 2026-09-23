"""Reconcile final GPU-job charges with measured model work, without running models."""
from collections import Counter
from rankcloak.revision_v4_stage2_common import ROOT,OUT,read_json,read_jsonl,atomic_json,file_hash
from rankcloak.revision_v4_stage2_analysis import csv_write
from scripts.validate_revision_v4_stage2 import verify_gpu_accounting,verify_transport_reuse

ledger=read_json(OUT/'gpu/ledger.json')
if any(j['status']=='running' for j in ledger['jobs']):raise ValueError('GPU work still active')
charged=verify_gpu_accounting(ledger,read_json(OUT/'gpu/unadmitted_invocations.json'))
rechecks=verify_transport_reuse(OUT/'transport')
model_work=[]
runtime_records={'lm':[],'semantic':[]}
for plan in ['pilot_initial','pilot_amendment1','coherence_study']:
    for path in sorted((OUT/'plans'/plan).glob('*execution*.json')):
        row=read_json(path)
        if row['status']!='complete':raise ValueError('incomplete model execution '+str(path))
        kind='semantic' if path.name=='semantic_execution.json' else 'lm'
        model=path.stem.removeprefix('lm_execution_') if kind=='lm' else 'all-MiniLM-L6-v2'
        runtime_records[kind].append(row['runtime_versions'])
        model_work.append({'plan':plan,'kind':kind,'model':model,'new_requests':row['new_requests'],
            'context_and_target_tokens':row.get('evaluated_context_and_target_tokens',0),
            'semantic_nonpadding_tokens':row.get('token_count',0),'source':str(path.relative_to(ROOT))})
for kind,records in runtime_records.items():
    if not records or any(r!=records[0] for r in records):raise ValueError('runtime version changed within '+kind+' execution')
backend_pins=read_json(OUT/'compatibility/example_header.json')['configuration']['backend']['library_sha256']
for name,sha in backend_pins.items():
    matches=list(ROOT.glob('.venv-generation-v3/lib/python*/site-packages/llama_cpp/lib/'+name))
    if len(matches)!=1 or file_hash(matches[0])!=sha:raise ValueError('pinned backend library changed '+name)
cache_requests=sum(len(read_jsonl(path)) for path in (OUT/'raw').glob('lm_cache_*.jsonl'))
if cache_requests!=sum(r['new_requests'] for r in model_work if r['kind']=='lm'):
    raise ValueError('model execution/cache count mismatch')
replay_rows=[r for path in (OUT/'transport').glob('replay_*.jsonl') for r in read_jsonl(path)]
recheck_rows=[r for path in (OUT/'transport').glob('reuse_validation_*.jsonl') for r in read_jsonl(path)]
summary={'gpu_job_wall_seconds':charged,'gpu_job_wall_hours':charged/3600,'ceiling_seconds':28800,
    'remaining_seconds':28800-charged,'job_status_counts':dict(Counter(j['status'] for j in ledger['jobs'])),
    'peak_supervised_process_memory_mib':max(j.get('peak_process_memory_mib',0) for j in ledger['jobs']),
    'recorded_runtime_versions':{k:v[0] for k,v in runtime_records.items()},
    'backend_shared_libraries_verified':backend_pins,
    'ledger_sha256':file_hash(OUT/'gpu/ledger.json'),'model_work':model_work,
    'lm_new_unique_scoring_requests':cache_requests,
    'lm_evaluated_context_and_target_tokens':sum(r['context_and_target_tokens'] for r in model_work),
    'semantic_new_unique_windows':sum(r['new_requests'] for r in model_work if r['kind']=='semantic'),
    'semantic_nonpadding_tokens':sum(r['semantic_nonpadding_tokens'] for r in model_work),
    'transport_missing_segment_replays':len(replay_rows),'transport_reused_endpoint_capacity_checks':rechecks,
    'transport_recheck_evaluated_tokens':sum(r['model_evaluated_tokens'] for r in recheck_rows),
    'transport_missing_preflight_context_and_received_tokens':805,
    'accounting_notes':['Every admitted job includes loading, hashing, fixtures and process shutdown in wall time.',
      'The failed model-ID invocation and the refused concurrent invocation are conservatively charged.',
      'LM token counters cover retained scoring requests, including prefills and evaluated final target tokens. They exclude numerical fixture calls.',
      'The 26 missing transport replays retain a preflight upper count of 805 context/received tokens. Early rejection lowers actual work, which was not separately retained for those requests.',
      'The 53 context-capacity checks retain actual evaluated-token counts. They add no logical cover arms.',
      'Peak MiB values come from nvidia-smi process sampling. Semantic allocated/reserved-byte peaks are additionally retained in its execution records.',
      'GPU kernel time is not substituted for cumulative job wall time. Idle CPU analysis and queue waits are excluded.']}
atomic_json(OUT/'gpu/resource_summary.json',summary)
csv_write(OUT/'gpu/job_summary.csv',[{k:j.get(k) for k in ['name','status','charged_seconds','forecast_seconds','hard_limit_seconds','peak_process_memory_mib','exit_code','started_epoch','finished_epoch']} for j in ledger['jobs']])
csv_write(OUT/'gpu/model_work.csv',model_work)
print({k:v for k,v in summary.items() if k in ['gpu_job_wall_seconds','lm_new_unique_scoring_requests','lm_evaluated_context_and_target_tokens','transport_reused_endpoint_capacity_checks']})
