"""Paired payload-cluster inference and strict joins for the frozen V4 study."""
from __future__ import annotations
import argparse,base64,csv
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from rankcloak.revision_v4_stage2_common import ROOT,OUT,read_json,read_jsonl,write_jsonl,atomic_json,digest,load_cache,file_hash,hash_order

METRICS=('semantic_cosine','context_gain','conditional_logp')

def csv_write(path,rows):
    rows=list(rows);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:(str(v) if isinstance(v,(dict,list)) else v) for k,v in r.items()} for r in rows)


def validate_score_record(row):
    request=row['request']
    if request['kind']=='lm':
        values=np.asarray(row['log_probabilities'],dtype=float)
        if len(values)!=len(request['target_ids']) or len(values)!=row['target_token_count']:
            raise ValueError('score tail denominator mismatch')
        if row['context_token_count']!=len(request['context_ids']):raise ValueError('score context denominator mismatch')
        if not len(values) or not np.all(np.isfinite(values)) or not np.isfinite(row['mean_log_probability']):raise ValueError('nonfinite likelihood score')
        if abs(float(np.mean(values))-row['mean_log_probability'])>1e-10:raise ValueError('score mean inconsistent with token likelihoods')
    else:
        vector=np.frombuffer(base64.b64decode(row['vector_f32_base64']),dtype='<f4').astype(float)
        if len(vector)!=384 or row['dimension']!=384:raise ValueError('embedding dimension mismatch')
        if not np.all(np.isfinite(vector)) or abs(float(np.linalg.norm(vector))-1)>1e-5:raise ValueError('embedding normalization mismatch')


def join_scores(plan):
    contracts=read_json(plan/'scoring_contracts.json');units=read_jsonl(plan/'units.jsonl');maps=read_jsonl(plan/'unit_requests.jsonl')
    if len({u['unit_id'] for u in units})!=len(units):raise ValueError('duplicate unit')
    mapping={r['unit_id']:r['requests'] for r in maps}
    if len(mapping)!=len(maps) or set(mapping)!={u['unit_id'] for u in units}:raise ValueError('unit mapping join mismatch')
    cache={}
    for p in (OUT/'raw').glob('*cache*.jsonl'):
        kind='semantic' if p.name.startswith('semantic') else 'lm'
        cache.update(load_cache(p,digest(contracts[kind])))
    planned=read_jsonl(plan/'requests.jsonl')
    missing=[r['request_id'] for r in planned if r['request_id'] not in cache]
    if missing:raise ValueError(f'incomplete planned sample: {len(missing)} requests')
    for row in planned:validate_score_record(cache[row['request_id']])
    scores=[]
    for u in units:
        row={k:v for k,v in u.items() if k!='evaluator_inputs'};q=mapping[u['unit_id']]
        if 'left' in q:
            vectors=[np.frombuffer(base64.b64decode(cache[q[k]]['vector_f32_base64']),dtype='<f4').astype(float) for k in ['left','right']]
            row['semantic_cosine']=float(np.dot(*vectors))
            row['semantic_truncated']=any(cache[q[k]]['truncated'] for k in ['left','right'])
        if 'conditional' in q:
            a,b=[cache[q[k]] for k in ['conditional','reference']]
            if a['request']['target_ids']!=b['request']['target_ids']:raise ValueError('tail path mismatch')
            row.update(context_gain=a['mean_log_probability']-b['mean_log_probability'],conditional_logp=a['mean_log_probability'],
                       reference_logp=b['mean_log_probability'],evaluator_tail_tokens=a['target_token_count'],
                       standalone_right_ids_match=u['evaluator_inputs']['standalone_right_ids_match'])
        scores.append(row)
    return scores


def payload_values(rows,arms,metric):
    """Pair first, average segments within trial, then trials within payload."""
    by_boundary=defaultdict(dict)
    for r in rows:
        if r['arm'] not in arms:continue
        if r['arm'] in by_boundary[r['boundary_id']]:raise ValueError('duplicate arm')
        by_boundary[r['boundary_id']][r['arm']]=r
    trial_values=defaultdict(list);classes={}
    for boundary,pair in by_boundary.items():
        if set(pair)!=set(arms):raise ValueError('unpaired boundary')
        a,b=[pair[k] for k in arms]
        if (a['payload_name'],a['trial_id'])!=(b['payload_name'],b['trial_id']):raise ValueError('cluster identity mismatch')
        key=(a['payload_name'],a['trial_id']);trial_values[key].append(a[metric]-b[metric]);classes[a['payload_name']]=a['payload_class']
    payloads=defaultdict(list)
    for (payload,trial),values in trial_values.items():payloads[payload].append(float(np.mean(values)))
    return {p:float(np.mean(v)) for p,v in payloads.items()},classes


def bootstrap_draws(classes,seed=20260922,count=2000):
    names=sorted(classes);index={p:i for i,p in enumerate(names)};strata=defaultdict(list)
    for p in names:strata[classes[p]].append(index[p])
    rng=np.random.default_rng(seed)
    draws=np.concatenate([rng.choice(strata[k],size=(count,len(strata[k])),replace=True) for k in sorted(strata)],axis=1)
    return names,draws


def effect(rows,arms,metric,seed=20260922,count=2000,universe=None):
    values,classes=payload_values(rows,arms,metric)
    if not values:raise ValueError('no complete pairs')
    # A common payload universe preserves identical draws across models despite missing strata.
    allclasses=universe or classes;names,draws=bootstrap_draws(allclasses,seed,count)
    vector=np.array([values.get(p,np.nan) for p in names]);draw_values=vector[draws]
    valid=np.isfinite(draw_values);den=valid.sum(axis=1)
    estimates=np.divide(np.nansum(draw_values,axis=1),den,out=np.full(count,np.nan),where=den>0)
    estimates=estimates[np.isfinite(estimates)]
    lo,hi=np.quantile(estimates,[.025,.975])
    return {'contrast':arms[0]+' minus '+arms[1],'metric':metric,'effect':float(np.mean(list(values.values()))),
            'ci_low':float(lo),'ci_high':float(hi),'payload_clusters':len(values),'bootstrap_draws':count,
            'nonempty_bootstrap_draws':len(estimates),'class_counts':dict(Counter(classes.values()))}


def analyze(plan,pilot=False):
    plan=Path(plan);rows=join_scores(plan);folder=OUT/'analysis'/plan.name;folder.mkdir(parents=True,exist_ok=True)
    write_jsonl(folder/'joined_scores.jsonl',rows);csv_write(folder/'joined_scores.csv',rows)
    contrasts=[('pilot_ordinary','pilot_shuffled')] if pilot else [('actual','ordinary'),('actual','shuffled'),('ordinary','shuffled')]
    metrics=[m for m in METRICS if m in rows[0]]
    universe={r['payload_name']:r['payload_class'] for r in rows}
    results=[]
    for arms in contrasts:
        for metric in metrics:
            for model in ['all']+sorted({r['model_id'] for r in rows}):
                subset=rows if model=='all' else [r for r in rows if r['model_id']==model]
                results.append({**effect(subset,arms,metric,universe=universe),'model_id':model})
    csv_write(folder/'effects.csv',results)
    summary={'plan':str(plan.relative_to(ROOT)),'plan_freeze_sha256':file_hash(plan/'freeze.json'),
        'scoring_units':len(rows),'boundaries':len({r['boundary_id'] for r in rows}),'trials':len({r['trial_id'] for r in rows}),
        'payload_clusters':len(universe),'primary_intervals_condition_on_fixed_control_corpus':True,
        'missing_planned_requests':0,'semantic_truncated_units':sum(r.get('semantic_truncated',False) for r in rows),
        'standalone_right_tokenization_differs_units':sum(not r.get('standalone_right_ids_match',True) for r in rows),
        'effects':results,'classes':dict(Counter(universe.values()))}
    if pilot:
        primary=[r for r in results if r['model_id']=='all' and r['metric']!='conditional_logp']
        summary.update(validated_metrics=[r['metric'] for r in primary if r['ci_low']>0],
            all_primary_validated=all(r['ci_low']>0 for r in primary) and len(primary)==2,
            strata_coverage=len({(r['model_id'],r['schedule'],r['prompt_category']) for r in rows}),
            actual_recipient_payloads=sorted({r['left_payload'] for r in rows}),donor_payloads=sorted({r['right_payload'] for r in rows}),
            rankcloak_metric_scores_seen=False)
    else:
        summary['window_token_counts']={arm:{
            'left_generator_distribution':dict(sorted(Counter(r['left_generator_tokens'] for r in rows if r['arm']==arm).items())),
            'right_generator_distribution':dict(sorted(Counter(r['right_generator_tokens'] for r in rows if r['arm']==arm).items())),
            'evaluator_tail_min':min(r['evaluator_tail_tokens'] for r in rows if r['arm']==arm),
            'evaluator_tail_max':max(r['evaluator_tail_tokens'] for r in rows if r['arm']==arm)
        } for arm in ['actual','ordinary','shuffled']}
        donors=[]
        for arm in ['ordinary','shuffled']:
            use=Counter(r['right_payload'] for r in rows if r['arm']==arm)
            ids=Counter(r['right_source_id'] for r in rows if r['arm']==arm)
            donors.extend({'arm':arm,'donor_payload':p,'uses':n} for p,n in sorted(use.items()))
            summary[arm+'_donor_support']={'payloads':len(use),'source_identities':len(ids),'max_payload_reuse':max(use.values()),'max_source_reuse':max(ids.values())}
        csv_write(folder/'donor_reuse.csv',donors)
        # Delete every pair touched by a specified donor, then reaggregate. This is an influence analysis, not a replacement CI.
        sensitivity=[]
        for arms in contrasts[:2]:
            reference_arm=arms[1];donor_names=sorted({r['right_payload'] for r in rows if r['arm']==reference_arm})
            touched=defaultdict(set)
            for r in rows:
                if r['arm']==reference_arm:touched[r['right_payload']].add(r['boundary_id'])
            for metric in metrics:
                influences=[]
                for donor in donor_names:
                    retained=[r for r in rows if r['boundary_id'] not in touched[donor]]
                    values,_=payload_values(retained,arms,metric)
                    if values:influences.append(float(np.mean(list(values.values()))))
                sensitivity.append({'contrast':' minus '.join(arms),'metric':metric,'removed_donor_payloads':len(influences),
                    'minimum_leave_one_donor_out_effect':min(influences),'maximum_leave_one_donor_out_effect':max(influences),
                    'interpretation':'Fixed corpus influence diagnostic, not a two-way population confidence interval.'})
        csv_write(folder/'donor_sensitivity.csv',sensitivity);summary['donor_sensitivity']=sensitivity
        freeze=read_json(plan/'freeze.json');primary_metric='semantic_cosine' if 'semantic_cosine' in freeze['validated_metrics'] else 'context_gain'
        actual=sorted([r for r in rows if r['arm']=='actual'],key=lambda r:(r[primary_metric],hash_order(20260922,r['boundary_id'])))
        examples=[{'label':label,'selection_metric':primary_metric,**actual[index]} for label,index in [('weak',0),('typical',len(actual)//2),('strong',len(actual)-1)]]
        atomic_json(folder/'examples.json',examples)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,len(metrics[:2]),figsize=(9,3.2),layout='constrained')
        for ax,metric in zip(np.atleast_1d(axes),metrics[:2]):
            vals=[r for r in results if r['metric']==metric and r['model_id']=='all']
            for y,r in enumerate(vals):ax.errorbar(r['effect'],y,xerr=[[r['effect']-r['ci_low']],[r['ci_high']-r['effect']]],fmt='o',capsize=3,color='#245789')
            ax.axvline(0,color='gray',lw=.8);ax.set_yticks(range(len(vals)),[r['contrast'] for r in vals]);ax.set_xlabel({'semantic_cosine':'Cosine difference','context_gain':'Context gain difference\n(nats per tail token)','conditional_logp':'Conditional log likelihood difference\n(nats per tail token)'}[metric]);ax.set_title('Exploratory primary' if metric=='context_gain' else 'Secondary likelihood' if metric=='conditional_logp' else 'Semantic relatedness');ax.invert_yaxis()
        fig.savefig(folder/'boundary_coherence.pdf');fig.savefig(folder/'boundary_coherence.png',dpi=180);plt.close(fig)
    atomic_json(folder/'summary.json',summary)
    text=['# '+('Control-only pilot' if pilot else 'Frozen boundary study'),'','All planned scoring requests were present. Intervals resample recipient payloads within artifact class and condition on the retained control corpus.','']
    for r in results:
        if r['model_id']=='all':text.append(f"- {r['contrast']}, {r['metric']}: {r['effect']:.4f}, 95% interval [{r['ci_low']:.4f}, {r['ci_high']:.4f}].")
    if pilot:text+=['',f"Validated primary metrics: {', '.join(summary['validated_metrics']) or 'none'}. These results concern separation of ordinary controls, not perceived naturalness."]
    (folder/'report.md').write_text('\n'.join(text)+'\n')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    result=analyze(ROOT/a.plan,a.pilot)
    print({k:v for k,v in result.items() if k in ['boundaries','payload_clusters','validated_metrics','all_primary_validated','strata_coverage']})
