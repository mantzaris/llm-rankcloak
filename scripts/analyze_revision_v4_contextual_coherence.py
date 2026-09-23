"""Offline calibration and nested payload-cluster estimates from retained judge responses."""
from collections import Counter, defaultdict
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from rankcloak.revision_v4_contextual_coherence import ROOT, OUT, CONFIG, calibration_cases
from rankcloak.revision_v4_stage2_common import read_json, read_jsonl, atomic_json, digest, hash_order, file_hash


def calibration_summary():
    cases={(r['context'],r['message']):r for r in calibration_cases()}
    summaries={}
    for path in sorted((OUT/'raw/calibration').glob('*.jsonl')):
        rows=read_jsonl(path);confusion=Counter();groups=defaultdict(list);scores=Counter();invalid=0;cost=[]
        for row in rows:
            case=cases[(row['request']['context'],row['request']['message'])];parsed=row['result']['parsed']
            if parsed['parse_status']=='valid':
                confusion[(int(case['intended_acceptable']),int(parsed['acceptable']))]+=1
                groups[case['construction']].append(int(parsed['acceptable']));scores[parsed['disruption']]+=1
            else:invalid+=1
            for a in row['result']['attempts']:
                cost.append({'input_tokens':a['response']['usage']['prompt_tokens'],'output_tokens':a['response']['usage']['completion_tokens'],'seconds':a['elapsed_seconds']})
        passed=(len(rows)==18 and invalid==0 and sum(groups['minor'])>=4 and len(groups['substantial'])-sum(groups['substantial'])>=4 and len(scores)>1)
        summaries[path.stem]={'examples':len(rows),'invalid':invalid,'confusion_intended_rows_observed_columns':[[confusion[(i,j)] for j in [0,1]] for i in [0,1]],
            'acceptable_by_construction':{g:{'acceptable':sum(v),'total':len(v)} for g,v in groups.items()},
            'four_level_counts':{str(k):scores[k] for k in range(4)},'basic_sanity_check_passed':passed,
            'seconds':sum(r['elapsed_seconds'] for r in rows),'actual_prompt_tokens':sum(c['input_tokens'] for c in cost),
            'completion_tokens':sum(c['output_tokens'] for c in cost),'attempts':len(cost),
            'costs':cost,'raw_sha256':file_hash(path)}
    result={'rubric_amendments':0,'models':summaries,'interpretation':'Constructed calibration checks interpretation and instrumentation, not equivalence to human readers.'}
    atomic_json(OUT/'analysis/calibration.json',result)
    return result


def nested_estimate(rows, field):
    trials=defaultdict(list);classes={}
    for r in rows:
        if r.get(field) is not None:
            trials[(r['payload_name'],r['trial_id'])].append(float(r[field]));classes[r['payload_name']]=r['payload_class']
    payloads=defaultdict(list)
    for (p,t),values in trials.items():payloads[p].append(float(np.mean(values)))
    return {p:float(np.mean(values)) for p,values in payloads.items()},classes


def shared_bootstrap(payload_classes, seed, draws=2000):
    rng=np.random.default_rng(seed);by=defaultdict(list)
    for p,c in sorted(payload_classes.items()):by[c].append(p)
    return [[p for cls in sorted(by) for p in rng.choice(by[cls],len(by[cls]),replace=True)] for _ in range(draws)]


def effect(rows, field, draws):
    values,_=nested_estimate(rows,field)
    if not values:return {'estimate':None,'lower':None,'upper':None,'payloads':0}
    estimates=[np.mean([values[p] for p in sample if p in values]) for sample in draws]
    lower,upper=np.quantile(estimates,[.025,.975])
    return {'estimate':float(np.mean(list(values.values()))),'lower':float(lower),'upper':float(upper),'payloads':len(values)}


def analyze_study(plan):
    cfg=read_json(CONFIG);messages=read_jsonl(plan/'messages.jsonl');units=read_jsonl(plan/'unit_requests.jsonl')
    scores={}
    for path in (OUT/'raw/judging').glob('*.jsonl'):
        for r in read_jsonl(path):scores[r['request_id']]=r
    grouped=defaultdict(dict);joined=[]
    for unit in units:
        r=scores.get(unit['request_id']);parsed=r['result']['parsed'] if r else {'parse_status':'missing','acceptable':None}
        row={**unit,**parsed};joined.append(row);grouped[unit['message_id']][(unit['arm'],unit['judge_model_id'])]=row
    classes={m['payload_name']:m['payload_class'] for m in messages}
    draws=shared_bootstrap(classes,cfg['analysis']['bootstrap_seed'],cfg['analysis']['bootstrap_draws'])
    panels=[];individual=[]
    for m in messages:
        base={k:m[k] for k in ['message_id','trial_id','payload_name','payload_class','generator','schedule']}
        judges=cfg['judge_assignment'][m['generator']];panel=dict(base)
        all_valid=all(grouped[m['message_id']][(arm,j)]['parse_status']=='valid' for arm in ['encoded','ordinary'] for j in judges)
        for arm in ['encoded','ordinary']:
            rs=[grouped[m['message_id']][(arm,j)] for j in judges]
            vals=[int(r['acceptable']) if r['parse_status']=='valid' else None for r in rs]
            panel[arm]=float(np.mean(vals)) if all_valid else None
            panel[arm+'_lower_missing']=float(np.mean([0 if x is None else x for x in vals]))
            panel[arm+'_upper_missing']=float(np.mean([1 if x is None else x for x in vals]))
            panel[arm+'_disagreement']=float(vals[0]!=vals[1]) if all(x is not None for x in vals) else None
            panel[arm+'_connectedness']=float(np.mean([r['connectedness'] for r in rs])) if all_valid else None
            for level in range(4):panel[arm+'_score_'+str(level)]=float(np.mean([r['disruption']==level for r in rs])) if all_valid else None
        panel['difference']=panel['encoded']-panel['ordinary'] if all_valid else None
        panel['difference_lower_missing']=panel['encoded_lower_missing']-panel['ordinary_upper_missing']
        panel['difference_upper_missing']=panel['encoded_upper_missing']-panel['ordinary_lower_missing']
        panels.append(panel)
        for j in judges:
            rs={a:grouped[m['message_id']][(a,j)] for a in ['encoded','ordinary']}
            valid=all(r['parse_status']=='valid' for r in rs.values());row={**base,'judge_model_id':j}
            for arm,r in rs.items():
                row[arm]=float(r['acceptable']) if valid else None
                row[arm+'_connectedness']=float(r['connectedness']) if valid else None
                for level in range(4):row[arm+'_score_'+str(level)]=float(r['disruption']==level) if valid else None
            row['difference']=row['encoded']-row['ordinary'] if valid else None
            individual.append(row)
    fields=['encoded','ordinary','difference','encoded_connectedness','ordinary_connectedness']+[a+'_score_'+str(s) for a in ['encoded','ordinary'] for s in range(4)]
    summaries=[]
    for group,key,values in [('panel','all',['all']),('generator','generator',sorted({m['generator'] for m in messages})),('schedule','schedule',sorted({m['schedule'] for m in messages})),('judge','judge_model_id',sorted({r['judge_model_id'] for r in individual}))]:
        for value in values:
            source=individual if group=='judge' else panels
            subset=source if value=='all' else [r for r in source if r[key]==value]
            row={'group':group,'value':value,'message_pairs':len(subset),'complete_pairs':sum(r['difference'] is not None for r in subset),'trials':len({r['trial_id'] for r in subset})}
            for field in fields+([] if group=='judge' else ['encoded_disagreement','ordinary_disagreement']):row[field]=effect(subset,field,draws)
            summaries.append(row)
    sensitivity={f:effect(panels,f,draws) for f in ['encoded_lower_missing','encoded_upper_missing','ordinary_lower_missing','ordinary_upper_missing','difference_lower_missing','difference_upper_missing']}
    raw_counts=Counter((r['arm'],r['judge_model_id'],str(r.get('disruption','missing'))) for r in joined)
    controls={}
    for p in (OUT/'raw/controls').glob('*.jsonl'):
        controls.update({r['request_id']:r for r in read_jsonl(p)})
    reuse=Counter(m['control_request_id'] for m in messages)
    lengths=[]
    for m in messages:
        c=controls[m['control_request_id']]['result']['generation']
        for arm,text,n,stop in [('encoded',m['text'],len(m['full_token_ids']),m['tail_stop_reason']),('ordinary',c['full_text'],len(c['full_token_ids']),c['tail_stop_reason'])]:
            lengths.append({'message_id':m['message_id'],'generator':m['generator'],'arm':arm,'tokens':n,'characters':len(text),'words':len(text.split()),'stop_reason':stop})
    length_summary=[]
    for g in ['all',*sorted({m['generator'] for m in messages})]:
        for arm in ['encoded','ordinary']:
            values=[r for r in lengths if r['arm']==arm and (g=='all' or r['generator']==g)]
            length_summary.append({'generator':g,'arm':arm,'messages':len(values),'mean_tokens':float(np.mean([r['tokens'] for r in values])),
                'median_tokens':float(np.median([r['tokens'] for r in values])),'min_tokens':min(r['tokens'] for r in values),'max_tokens':max(r['tokens'] for r in values),
                'cap_stops':sum(r['stop_reason']=='emergency_cap_censored' for r in values)})
    examples=[]
    for name,predicate in [('acceptable',lambda v:v==1),('disagreement',lambda v:v==.5),('disrupted',lambda v:v==0)]:
        candidates=[r for r in panels if predicate(r['encoded'])]
        if candidates:
            chosen=min(candidates,key=lambda r:hash_order(cfg['seed'],r['message_id']));m=next(m for m in messages if m['message_id']==chosen['message_id'])
            examples.append({'category':name,**m,'judgments':[grouped[m['message_id']][('encoded',j)] for j in cfg['judge_assignment'][m['generator']]]})
    result={'status':'complete' if len(scores)==len(read_jsonl(plan/'judge_requests.jsonl')) else 'incomplete',
        'selected_payloads':len(classes),'trials':len({m['trial_id'] for m in messages}),'encoded_messages':len(messages),
        'logical_judge_units':len(units),'unique_judge_requests':len(read_jsonl(plan/'judge_requests.jsonl')),'completed_unique_requests':len(scores),
        'invalid_logical_scores':sum(r['parse_status']!='valid' for r in joined),'invalid_unique_responses':sum(r['result']['parsed']['parse_status']!='valid' for r in scores.values()),
        'retries':sum(len(r['result']['attempts'])-1 for r in scores.values()),'summaries':summaries,'missing_sensitivity':sensitivity,
        'control_reuse':{'unique_requests':len(reuse),'maximum_reuse':max(reuse.values()),'minimum_reuse':min(reuse.values()),'reuse_counts':dict(reuse)},
        'lengths':length_summary,'raw_disruption_counts':[{'arm':a,'judge':j,'score':s,'count':n} for (a,j,s),n in sorted(raw_counts.items())],
        'inference':'Paired recipient-payload cluster bootstrap with 2000 shared class-stratified draws. Fixed control set. Physical judge and source generator are not fully crossed. Automated rubric scores, not human perception.'}
    dest=OUT/'analysis';dest.mkdir(exist_ok=True)
    atomic_json(dest/'summary.json',result);atomic_json(dest/'examples.json',examples)
    for name,rows in [('joined_scores',joined),('panel_message_scores',panels),('individual_judge_pairs',individual),('lengths',lengths)]:
        from rankcloak.revision_v4_stage2_common import write_jsonl
        write_jsonl(dest/(name+'.jsonl'),rows)
    flat=[]
    for r in summaries:
        flat.append({k:v for k,v in r.items() if not isinstance(v,dict)} | {k+'_'+measure:val for k,v in r.items() if isinstance(v,dict) for measure,val in v.items()})
    with (dest/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in flat))));w.writeheader();w.writerows(flat)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--calibration',action='store_true');p.add_argument('--plan',default='final_sample');a=p.parse_args()
    result=calibration_summary() if a.calibration else analyze_study(OUT/'plans'/a.plan)
    print(json.dumps(result if a.calibration else {k:v for k,v in result.items() if k not in ['summaries','raw_disruption_counts','control_reuse','lengths','missing_sensitivity']},indent=2))


if __name__=='__main__':main()
