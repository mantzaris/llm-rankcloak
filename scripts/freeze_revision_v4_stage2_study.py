"""Admit the largest prospective payload sample using control-only measured costs."""
from collections import defaultdict,Counter
from datetime import datetime,timezone
from rankcloak.revision_v4_stage2_common import *
from rankcloak.revision_v4_stage2_scoring import make_requests,scoring_contract
from rankcloak.revision_v4_stage2_analysis import csv_write

pilot=read_json(OUT/'analysis/pilot_amendment1/summary.json');cfg=read_json(ROOT/'configs/revision_v4/stage2_coherence_amendment1.json')
metrics=pilot['validated_metrics']
if not metrics:raise ValueError('no validated metric for exploratory study')
units=read_jsonl(OUT/'plans/amendment1/full_eligible_units.jsonl');boundaries=read_jsonl(OUT/'plans/amendment1/boundaries.jsonl')
classes={r['payload_name']:r['payload_class'] for r in boundaries};strata=defaultdict(list)
for name,kind in classes.items():strata[kind].append(name)
for names in strata.values():names.sort(key=lambda name:hash_order(cfg['seed'],name))
assert len(classes)==240 and len(strata)==4
rates={};cached=set()
for model in cfg['evaluator_map'].values():
    estimates=[]
    for p in ['pilot_initial','pilot_amendment1']:
        r=read_json(OUT/'plans'/p/('lm_execution_'+model+'.json'))
        estimates.append(r['scoring_seconds']/r['evaluated_context_and_target_tokens'])
    rates[model]=max(estimates)
    cached.update(r['request_id'] for r in read_jsonl(OUT/'raw'/('lm_cache_'+model+'.jsonl')))
cached.update(r['request_id'] for r in read_jsonl(OUT/'raw/semantic_cache.jsonl'))
ledger=read_json(OUT/'gpu/ledger.json');charged=sum(r['charged_seconds'] for r in ledger['jobs']);reserve=cfg['transport_reserve_seconds']+cfg['safety_reserve_seconds'];available=cfg['budget_seconds']-charged-reserve
# Construct identities from unscored inputs only. Scored values are not read for selection.
temp=OUT/'plans/full_candidates_unscored';temp.mkdir(parents=True,exist_ok=True)
make_requests(units,temp,metrics);maps={r['unit_id']:r['requests'] for r in read_jsonl(temp/'unit_requests.jsonl')};requests={r['request_id']:r['request'] for r in read_jsonl(temp/'requests.jsonl')}
per_payload=defaultdict(set)
for u in units:per_payload[u['payload_name']].update(maps[u['unit_id']].values())

def forecast(names):
    keys=set().union(*(per_payload[name] for name in names));counts=Counter();tokens=Counter();seconds=Counter()
    for key in keys-cached:
        r=requests[key]
        if r['kind']=='lm':
            m=r['model_id'];n=len(r['context_ids'])+len(r['target_ids']);counts[m]+=1;tokens[m]+=n;seconds[m]+=rates[m]*n
    bymodel={m:float(seconds[m]*cfg['forecast_multiplier']+30) for m in rates}
    semantic_seconds=60 if 'semantic_cosine' in metrics else 0
    return {'unique_requests':len(keys),'uncached_requests':len(keys-cached),'model_request_counts':dict(counts),'context_and_target_tokens':dict(tokens),
        'forecast_gpu_seconds_by_model':bymodel,'forecast_gpu_seconds':sum(bymodel.values())+semantic_seconds}

candidates=[];choice=None
for count in range(1,max(map(len,strata.values()))+1):
    names=sorted(name for pool in strata.values() for name in pool[:count]);f=forecast(names)
    candidates.append({'payloads_per_class':count,'payloads':len(names),**f})
    if f['forecast_gpu_seconds']<=available:choice=(count,names,f)
if choice is None:raise ValueError('No balanced payload sample fits conservative budget')
count,names,f=choice;chosen=[u for u in units if u['payload_name'] in set(names)];plan=OUT/'plans/coherence_study';plan.mkdir(parents=True,exist_ok=True);write_jsonl(plan/'units.jsonl',chosen)
made=make_requests(chosen,plan,metrics)
freeze={'created_utc':datetime.now(timezone.utc).isoformat(),'analysis_status':'completed two-metric control validation' if pilot['all_primary_validated'] else 'limited exploratory analysis using validated context gain; R1.4 partly unresolved',
    'validated_metrics':metrics,'scored_metrics':metrics,'pilot_summary_sha256':file_hash(OUT/'analysis/pilot_amendment1/summary.json'),
    'plan_config_sha256':file_hash(ROOT/'configs/revision_v4/stage2_coherence_amendment1.json'),'inventory_manifest_sha256':file_hash(OUT/'plans/amendment1/inventory_manifest.json'),
    'rankcloak_scores_seen_at_freeze':False,'selection_rule':cfg['sample_rule'],'selected_payload_names':names,'payloads_per_class':count,
    'selected_payloads':len(names),'selected_structural_boundaries':sum(r['payload_name'] in names for r in boundaries),'selected_planned_trials':len({r['trial_id'] for r in boundaries if r['payload_name'] in names}),
    'selected_eligible_boundaries':len(chosen)//3,'selected_scoring_trials':len({r['trial_id'] for r in chosen}),'selection_is_full':len(names)==240,
    'charged_gpu_seconds_at_admission':charged,'reserved_seconds':reserve,'remaining_for_study_seconds':available,
    'cost_rates_seconds_per_context_or_target_token':rates,'forecast_multiplier':cfg['forecast_multiplier'],'all_sample_forecasts':candidates,**f,**made}
immutable_json(plan/'freeze.json',freeze)
# Every structural boundary stays in a stratified eligibility table, including unselected payloads.
groups=defaultdict(list)
for row in boundaries:groups[(row['model_id'],row['schedule'],row['payload_class'],row['payload_name'] in names)].append(row)
report=[]
for (model,schedule,kind,selected),rs in sorted(groups.items()):
    reasons=Counter(reason for r in rs for reason in r['exclusion_reasons'])
    report.append({'model_id':model,'schedule':schedule,'payload_class':kind,'selected_payload':selected,'structural_boundaries':len(rs),'eligible_boundaries':sum(r['eligible'] for r in rs),'excluded_boundaries':sum(not r['eligible'] for r in rs),'reasons':dict(reasons)})
csv_write(plan/'eligibility_by_stratum.csv',report)
atomic_json(plan/'donor_and_exclusion_pre_score.json',{'all_structural_boundaries':len(boundaries),'all_eligible_boundaries':sum(r['eligible'] for r in boundaries),
    'exact_identical_actual_donor_candidates_skipped':sum(r['identical_actual_donors_skipped'] for r in boundaries),
    'exact_identical_pilot_donor_candidates_skipped':sum(r['identical_pilot_donors_skipped'] for r in boundaries),
    'reason_counts_overlap':dict(Counter(reason for r in boundaries for reason in r['exclusion_reasons'])),
    'no_eligible_boundaries_trials':[t for t in sorted({r['trial_id'] for r in boundaries}) if not any(u['trial_id']==t for u in units)]})
atomic_json(OUT/'plans/pilot_amendment1/decision.json',{'decision_utc':datetime.now(timezone.utc).isoformat(),'validated_metrics':metrics,'all_primary_validated':pilot['all_primary_validated'],'further_amendments':0,'decision':'Continue a frozen exploratory context-gain study. Retain semantic validation failure. No human-naturalness or completed two-metric claim.','rankcloak_scores_seen':False})
print({k:freeze[k] for k in ['selected_payloads','payloads_per_class','selected_eligible_boundaries','unique_requests','uncached_requests','forecast_gpu_seconds','forecast_gpu_seconds_by_model','selection_is_full']})
