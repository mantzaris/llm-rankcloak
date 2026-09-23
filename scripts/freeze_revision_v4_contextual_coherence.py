"""Admit and freeze the final sample from calibration costs before any study judging."""
from collections import Counter
import datetime
from pathlib import Path
import shutil
from rankcloak.revision_v4_contextual_coherence import ROOT, OUT, CONFIG, inventory_and_select, judge_content
from rankcloak.revision_v4_stage2_common import read_json, read_jsonl, digest, file_hash, write_jsonl, immutable_json, hash_order
from scripts.run_revision_v4_contextual_coherence import rendered_prompt


def requests_for(messages, controls, cfg):
    requests={};units=[]
    for m in messages:
        ordinary=controls[m['control_request_id']]['result']['generation']['full_text']
        for arm,text in [('encoded',m['text']),('ordinary',ordinary)]:
            if text is None:continue
            for judge in cfg['judge_assignment'][m['generator']]:
                request={'judge_model_id':judge,'context':m['prompt']['prompt_text'],'message':text,'config_sha256':file_hash(CONFIG)}
                identity=digest(request);requests[identity]={'request_id':identity,'request':request}
                units.append({k:m[k] for k in ['message_id','trial_id','payload_name','payload_class','generator','schedule','control_request_id']} |
                             {'arm':arm,'judge_model_id':judge,'request_id':identity})
    return list(requests.values()),units


def main():
    cfg=read_json(CONFIG)
    if list((OUT/'raw/judging').glob('*.jsonl')):raise ValueError('Cannot select a sample after final judging began')
    calibration=read_json(OUT/'analysis/calibration.json')
    if len(calibration['models'])!=3 or any(x['examples']!=18 for x in calibration['models'].values()):raise ValueError('Calibration incomplete')
    controls={r['request_id']:r for path in (OUT/'raw/controls').glob('*.jsonl') for r in read_jsonl(path)}
    target=OUT/'plans/target_sample';messages=read_jsonl(target/'messages.jsonl')
    if not {m['control_request_id'] for m in messages}<=controls.keys():raise ValueError('Matched controls incomplete')
    requests,units=requests_for(messages,controls,cfg)
    # Vocabulary-only work. No tensor loading or CUDA inference outside the ledger.
    from rankcloak.revision_tokenizer_preflight import load_vocab_only_tokenizer
    lengths={}
    for pin in cfg['models']:
        model=load_vocab_only_tokenizer(ROOT/pin['expected_path'])
        try:
            for row in requests:
                r=row['request']
                if r['judge_model_id']!=pin['model_id']:continue
                prompt=rendered_prompt(model,judge_content(r['context'],r['message']))
                lengths[row['request_id']]=len(model.tokenize(prompt.encode(),add_bos=False,special=True))
        finally:model.close()
    if max(lengths.values())+192>2048:raise ValueError('Intact text would exceed the frozen judge context')
    ledger=read_json(OUT/'gpu/ledger.json')
    if any(j['status']=='running' for j in ledger['jobs']):raise ValueError('Wait for current GPU job to checkpoint before admission')
    used=sum(j['charged_seconds'] for j in ledger['jobs']);available=21600-used-cfg['budget']['reserve_seconds']
    rates={m:sum(c['seconds'] for c in r['costs'])/sum(c['input_tokens']+c['output_tokens'] for c in r['costs']) for m,r in calibration['models'].items()}
    ordered={}
    for m in messages:ordered.setdefault(m['payload_class'],set()).add(m['payload_name'])
    ordered={c:sorted(p,key=lambda x:hash_order(cfg['seed'],x)) for c,p in ordered.items()}
    forecasts=[];selected_count=0
    for count in range(1,13):
        payloads={p for group in ordered.values() for p in group[:count]}
        subset=[m for m in messages if m['payload_name'] in payloads]
        req,_=requests_for(subset,controls,cfg)
        # Reserve a full second request and 192-token retry for every identity.
        costs={m:sum((2*lengths[r['request_id']]+128+192)*rates[m]*cfg['budget']['forecast_multiplier'] for r in req if r['request']['judge_model_id']==m)+60 for m in rates}
        total=sum(costs.values());forecasts.append({'payloads_per_class':count,'messages':len(subset),'unique_requests':len(req),'forecast_seconds':total,'by_model':costs})
        if total<=available:selected_count=count
    if selected_count==0:raise ValueError('No balanced sample fits the conservative budget')
    selection=inventory_and_select(selected_count,'final_sample');plan=OUT/'plans/final_sample'
    final_messages=read_jsonl(plan/'messages.jsonl');final_requests,final_units=requests_for(final_messages,controls,cfg)
    write_jsonl(plan/'judge_requests.jsonl',sorted(final_requests,key=lambda r:r['request_id']))
    write_jsonl(plan/'unit_requests.jsonl',final_units)
    source_files=['configs/revision_v4/coherence_replacement.json','rankcloak/revision_v4_contextual_coherence.py',
                  'scripts/run_revision_v4_contextual_coherence.py','scripts/analyze_revision_v4_contextual_coherence.py',
                  'scripts/freeze_revision_v4_contextual_coherence.py','scripts/run_revision_v4_contextual_job.py']
    freeze={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':read_json(OUT/'provenance/initial_state.json')['head'],
        'rankcloak_judge_outcomes_seen':False,'forecast_uses_text_quality':False,'selected_payloads_per_class':selected_count,
        'selected_payloads':selection['selected_payloads'],'messages':len(final_messages),'unit_count':len(final_units),
        'unique_requests':len(final_requests),'actual_input_token_counts':{r['request_id']:lengths[r['request_id']] for r in final_requests},
        'calibration_sanity_passed':{m:r['basic_sanity_check_passed'] for m,r in calibration['models'].items()},
        'rubric_amendments':0,'charged_seconds_at_admission':used,'available_after_reserve_seconds':available,
        'forecasts':forecasts,'selected_forecast':forecasts[selected_count-1],'seconds_per_actual_input_plus_output_token':rates,
        'selection_limit':'Representative hash sample of the 148 outside-cohort payloads, balanced by class. Prior 92 have zero inclusion probability.',
        'analysis_contract':'Panel complete-pair estimates require both assigned judges to provide valid scores for both conditions in a selected message. Physical-judge estimates use that judge\u2019s complete pairs. All-slot worst/best missing-response bounds are reported. Two judges are not independent participants.',
        'source_hashes':{p:file_hash(ROOT/p) for p in source_files},
        'control_cache_hashes':{str(p.relative_to(ROOT)):file_hash(p) for p in (OUT/'raw/controls').glob('*.jsonl')},
        'files':{p.name:file_hash(p) for p in plan.glob('*.json*')}}
    immutable_json(plan/'freeze.json',freeze)
    print({k:freeze[k] for k in ['selected_payloads_per_class','messages','unit_count','unique_requests','selected_forecast','charged_seconds_at_admission']})


if __name__=='__main__':main()
