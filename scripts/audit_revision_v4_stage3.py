"""Independent CPU reconstruction of retained V4 endpoints. Never writes its inputs.

The implementation deliberately does not call the Stage 2 join, bootstrap or codec
analysis functions. --root and --output allow the same check on an extracted archive.
"""
import argparse
import base64
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bounded_bytes(ranks, metadata):
    width = metadata['bits_per_symbol']
    require(all(1 <= r <= 2**width for r in ranks), 'invalid bounded rank')
    bits = ''.join(format(r-1, f'0{width}b') for r in ranks)
    padding = metadata['padding_bits']
    # The historical decoder discards declared padding without validating its value.
    if padding:
        bits = bits[:-padding]
    require(len(bits) == 8*metadata['original_byte_length'], 'wrong byte length')
    return bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))


def paired_effect(rows, arms, metric, universe):
    """Explicit boundary -> trial -> recipient means and shared stratified draws."""
    pairs = defaultdict(dict)
    for r in rows:
        if r['arm'] in arms:
            require(r['arm'] not in pairs[r['boundary_id']], 'duplicate arm')
            pairs[r['boundary_id']][r['arm']] = r
    trials = defaultdict(list)
    for pair in pairs.values():
        require(set(pair) == set(arms), 'missing paired arm')
        a, b = (pair[k] for k in arms)
        require((a['payload_name'], a['trial_id']) == (b['payload_name'], b['trial_id']), 'wrong recipient join')
        trials[(a['payload_name'], a['trial_id'])].append(a[metric]-b[metric])
    recipients = defaultdict(list)
    for (payload, _), values in trials.items():
        recipients[payload].append(sum(values)/len(values))
    values = {p: sum(v)/len(v) for p, v in recipients.items()}
    names = sorted(universe)
    strata = defaultdict(list)
    for i, p in enumerate(names):
        strata[universe[p]].append(i)
    rng = np.random.default_rng(20260922)
    indices = np.concatenate([rng.choice(strata[c], (2000, len(strata[c])), replace=True)
                              for c in sorted(strata)], axis=1)
    vector = np.array([values.get(p, np.nan) for p in names])
    interval = np.quantile(np.nanmean(vector[indices], axis=1), [.025, .975])
    return [sum(values.values())/len(values), *interval.tolist()]


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output != root and not str(output).startswith(str(root/'results/revision_v4/stage2')), 'refuse historical output')
    output.mkdir(parents=True, exist_ok=True)
    inputs = {}

    def raw(path):
        path = root/path
        data = path.read_bytes()
        inputs[path.relative_to(root).as_posix()] = sha(data)
        return data

    def obj(path):
        return json.loads(raw(path))

    def rows(path):
        return [json.loads(line) for line in raw(path).splitlines() if line]

    def save(name, value):
        (output/name).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')

    primary = []
    for p in sorted(root.glob('results/revision_v1/primary_v2/*/records.jsonl')):
        primary.extend(rows(p.relative_to(root)))
    encoded = [r for r in primary if r['record_type'] == 'rankcloak_trial']
    require(len(primary) == 14400 and len(encoded) == 6480, 'primary denominator')
    for r in encoded:
        replay = r['saved_token_id_replay']
        require(replay['exact_payload_recovery'] and replay['exact_recovery'], 'primary recovery')
        require(sha(r['payload_text'].encode()) == r['original_payload_sha256'], 'original payload hash')
        require(replay['decoded']['recovered_payload_sha256'] == r['original_payload_sha256'], 'decoded payload hash')
    segmented = [r for r in encoded if r['segmented']]
    structural = {r['trial_id']+'__segment_'+str(i) for r in segmented for i, s in enumerate(r['segments'])
                  if s['tail_token_ids']}
    require((len(segmented), len(structural), len({r['payload_name'] for r in segmented})) == (1440,8280,240), 'structural source count')
    recovery = {'primary_trials':6480, 'primary_exact':6480, 'payloads':len({r['payload_name'] for r in encoded}),
                'structural_boundaries':len(structural), 'segmented_classes':dict(Counter({r['payload_name']:r['payload_class'] for r in segmented}.values()))}
    robust = []
    for p in sorted(root.glob('results/revision_v1/robustness_v2/*/records.jsonl')):
        robust.extend(rows(p.relative_to(root)))
    groups = defaultdict(list)
    for r in robust:
        if r['record_type'] in ['robustness_decode', 'robustness_reference']:
            groups[(r['robustness_family'], r['replay_mode'], r['transformation_id'], r['record_type'])].append(r)
            if r['record_type'] == 'robustness_reference':
                require(r['reference_field']=='saved_token_id_replay.exact_recovery' and not r['decode_performed'], 'reference misclassified')
    recovery['robustness'] = [dict(family=k[0],mode=k[1],transformation=k[2],record_type=k[3],
                                  numerator=sum(r['exact_recovery'] for r in v),denominator=len(v)) for k,v in sorted(groups.items())]
    visible = next(r for r in recovery['robustness'] if r['mode']=='detokenized_text_retokenized')
    require((visible['numerator'],visible['denominator']) == (88,144), 'visible endpoint')
    for name, n in [('markdown_copy_paste',0),('paraphrase',0),('unicode_normalization',82),('quote_conversion',56),('whitespace_trim',28)]:
        x = next(r for r in recovery['robustness'] if r['family']=='raw_transmission' and r['transformation']==name)
        require((x['numerator'],x['denominator']) == (n,144), 'historical transform '+name)
    save('recovery.json', recovery)

    stage2 = Path('results/revision_v4/stage2')
    inventory = rows(stage2/'plans/amendment1/boundaries.jsonl')
    require({r['boundary_id'] for r in inventory} == structural, 'structural inventory source linkage')
    freeze = obj(stage2/'plans/coherence_study/freeze.json')
    classes = {r['payload_name']:r['payload_class'] for r in segmented}
    chosen = set()
    for c in sorted(set(classes.values())):
        chosen.update(sorted((p for p in classes if classes[p]==c),key=lambda p:sha(('20260922|'+p).encode()))[:23])
    require(chosen == set(freeze['selected_payload_names']), 'selection identity hash rule')
    selected = [r for r in inventory if r['payload_name'] in chosen]
    require(len(selected)==3174 and sum(r['eligible'] for r in selected)==2750, 'selected/eligible counts')
    counts = {'original_structural':8280,'selected_payloads':92,'selected_structural':3174,'eligible_boundaries':2750,
              'planned_trials':len({r['trial_id'] for r in selected}), 'by_generator':{}}
    for model in sorted({r['model_id'] for r in selected}):
        subset = [r for r in selected if r['model_id']==model]
        counts['by_generator'][model] = {'structural':len(subset),'excluded':sum(not r['eligible'] for r in subset)}

    cache = {}
    for p in sorted(root.glob(str(stage2/'raw/*cache*.jsonl'))):
        for r in rows(p.relative_to(root)):
            require(r['request_id'] not in cache, 'duplicate cache key')
            cache[r['request_id']] = r
    all_effects = []
    pilot_payloads = []
    for plan in ['pilot_initial', 'pilot_amendment1', 'coherence_study']:
        units = rows(stage2/'plans'/plan/'units.jsonl')
        mapping = {r['unit_id']:r['requests'] for r in rows(stage2/'plans'/plan/'unit_requests.jsonl')}
        require(len(mapping)==len(units) and set(mapping)=={u['unit_id'] for u in units}, 'unit mapping')
        scores = []
        for u in units:
            m = mapping[u['unit_id']]
            a,b = (cache[m[k]] for k in ['conditional','reference'])
            for r in [a,b]:
                require(len(r['log_probabilities'])==len(r['request']['target_ids'])==r['target_token_count'], 'target denominator')
                require(np.isclose(np.mean(r['log_probabilities']),r['mean_log_probability'],rtol=0,atol=1e-12), 'mean log probability')
                require(r['request_id']==sha(json.dumps(r['request'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()), 'request identity')
            require(a['request']['target_ids']==b['request']['target_ids']==u['evaluator_inputs']['target_ids'], 'shared fixed tail path')
            require(a['request']['context_ids']==u['evaluator_inputs']['conditional_context_ids'] and b['request']['context_ids']==u['evaluator_inputs']['reference_context_ids'], 'context mapping')
            score = {**u, 'conditional_logp':float(np.mean(a['log_probabilities'])),
                     'context_gain':float(np.mean(np.array(a['log_probabilities'])-np.array(b['log_probabilities'])))}
            if 'left' in m:
                vectors = [np.frombuffer(base64.b64decode(cache[m[k]]['vector_f32_base64']),dtype='<f4').astype(float) for k in ['left','right']]
                score['semantic_cosine'] = float(vectors[0] @ vectors[1])
                require(not cache[m['left']]['truncated'] and not cache[m['right']]['truncated'], 'pilot truncation')
            scores.append(score)
        old = {r['unit_id']:r for r in rows(stage2/'analysis'/plan/'joined_scores.jsonl')}
        metrics = ['context_gain','conditional_logp'] + (['semantic_cosine'] if plan!='coherence_study' else [])
        for r in scores:
            require(all(abs(r[k]-old[r['unit_id']][k])<1e-10 for k in metrics), 'joined score mismatch')
        universe = {r['payload_name']:r['payload_class'] for r in scores}
        contrasts = [('actual','ordinary'),('actual','shuffled'),('ordinary','shuffled')] if plan=='coherence_study' else [('pilot_ordinary','pilot_shuffled')]
        summary = obj(stage2/'analysis'/plan/'summary.json')
        for arms in contrasts:
            for metric in metrics:
                for model in ['all']+sorted({r['model_id'] for r in scores}):
                    subset = scores if model=='all' else [r for r in scores if r['model_id']==model]
                    estimate = paired_effect(subset, arms, metric, universe)
                    prior = next(r for r in summary['effects'] if r['contrast']==' minus '.join(arms) and r['metric']==metric and r['model_id']==model)
                    require(np.allclose(estimate,[prior[k] for k in ['effect','ci_low','ci_high']],rtol=0,atol=1e-10), 'bootstrap effect mismatch')
                    all_effects.append(dict(plan=plan, contrast=' minus '.join(arms), metric=metric,model=model,
                                           effect=estimate[0],ci_low=estimate[1],ci_high=estimate[2]))
        if plan!='coherence_study':
            require(len(units)==72 and not any(r['arm']=='actual' for r in units), 'pilot blindness')
            pilot_payloads.append({r[k] for r in units for k in ['left_payload','right_payload']})
        else:
            require(len(units)==8250 and len({r['trial_id'] for r in units})==550, 'scored denominator')
            counts.update(scored_trials=550,arm_units=len(units),standalone_right_tokenization_differs=sum(not r['evaluator_inputs']['standalone_right_ids_match'] for r in units))
            require(counts['standalone_right_tokenization_differs']==3057, 'tokenization difference count')
            counts['donors'] = {}
            for arm in ['ordinary','shuffled']:
                part = [r for r in units if r['arm']==arm]
                donors = Counter(r['right_payload'] for r in part); identities=Counter(r['right_source_id'] for r in part)
                counts['donors'][arm] = dict(payloads=len(donors),identities=len(identities),max_payload_reuse=max(donors.values()),max_identity_reuse=max(identities.values()))
            ordered = sorted((r for r in scores if r['arm']=='actual'),key=lambda r:(r['context_gain'],sha(('20260922|'+r['boundary_id']).encode())))
            examples = [ordered[i] for i in [0,len(ordered)//2,-1]]
            require([r['boundary_id'] for r in examples]==[r['boundary_id'] for r in obj(stage2/'analysis/coherence_study/examples.json')], 'predeclared examples')
            save('examples.json',[{k:r[k] for k in ['boundary_id','left','right','context_gain']} for r in examples])
    require(not pilot_payloads[0]&pilot_payloads[1] and len(pilot_payloads[0])==69, 'pilot disjointness')
    counts['pilot_payload_overlap']=0
    save('boundary_counts.json',counts)
    save('reconstructed_effects.json',all_effects)
    with (output/'reconstructed_effects.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_effects[0]),lineterminator='\n');writer.writeheader();writer.writerows(all_effects)

    quant=defaultdict(lambda:Counter(pairs=0,positions=0,rank_changes=0,invalid_trials=0,valid_wrong=0,exact=0))
    source={r['trial_id']:r for r in encoded}
    endpoints={r['q8_plan_id']:r for r in rows(stage2/'quantization/q4_to_q8_fixed_message_decode.jsonl')}
    base=Path('results/revision_v3/generation/raw/quantization')
    for path in sorted(root.glob(str(base/'qwen2_5_7b_instruct_q8_0/*.json'))):
        q8=obj(path.relative_to(root));q4=obj(base/'qwen2_5_7b_instruct_q4_k_m'/(q8['paired_q4_replay_plan_id']+'.json'))
        a=q4['distribution_trace'];b=q8['q8_replay_of_historical_q4_path'];group=quant[q8['population']]
        require(a['observed_token_ids']==b['observed_token_ids'] and a['context_token_ids']==b['context_token_ids'], 'shared history mismatch')
        require(len(a['observed_ranks'])==len(b['observed_ranks']), 'rank length mismatch')
        group.update(pairs=1,positions=len(a['observed_ranks']),rank_changes=sum(x!=y for x,y in zip(a['observed_ranks'],b['observed_ranks'])))
        if q8['population']!='rankcloak':continue
        src=source[q8['source_lineage']['rank_trial_id']];meta=src['representation']['metadata']
        require(bounded_bytes(a['observed_ranks'],meta)==src['payload_text'].encode(), 'Q4 original-byte reconstruction')
        invalid=any(not 1<=r<=meta['alphabet_size'] for r in b['observed_ranks'])
        endpoint=endpoints[q8['plan_id']]
        require(invalid==bool(endpoint['invalid_rank_count']), 'invalid-rank classification')
        if invalid:group.update(invalid_trials=1)
        else:
            decoded=bounded_bytes(b['observed_ranks'],meta)
            if meta['padding_bits'] and (b['observed_ranks'][-1]-1) % (2**meta['padding_bits']):
                group.update(accepted_nonzero_terminal_padding=1)
            require(decoded.hex()==endpoint['decoded']['recovered_bytes']['hex'], 'Q8 reconstructed bytes differ')
            group.update(exact=int(decoded==src['payload_text'].encode()),valid_wrong=int(decoded!=src['payload_text'].encode()))
    require(quant['rankcloak']['invalid_trials']==766 and quant['rankcloak']['valid_wrong']==194 and not quant['rankcloak']['exact'], 'quantization endpoint')
    require(sum(r['rank_changes'] for r in quant.values())==69528, 'quantization changes')
    save('quantization.json',dict(quant))

    entropy=defaultdict(lambda:Counter(attempts=0,complete=0));failures=[]
    for path in sorted(root.glob('results/revision_v3/generation/raw/entropy/*/*.json')):
        r=obj(path.relative_to(root))
        if r['record_type']!='entropy_rankcloak_trial':continue
        g=r['generation'];gate=r['plan_row']['gate_level']
        entropy[gate].update(attempts=1,complete=int(g['payload_completion']))
        if not g['payload_completion']:
            require(r['source_lineage']['source_token_filter']=='none', 'entropy filter')
            require(g['ineligible_token_policy']=='ordinary_seeded_top_p_sampling', 'skip policy')
            eligible=np.array(g['embedding_entropies_bits'])>=g['entropy_threshold_bits']
            require(eligible.tolist()==g['embedding_eligible_mask'], 'eligibility inconsistency')
            runs=[];start=None
            for i,ok in enumerate(list(eligible)+[True]):
                if not ok and start is None:start=i
                if ok and start is not None:runs.append((i-start,start,i));start=None
            length,start,stop=max(runs,key=lambda x:(x[0],-x[1]))
            failures.append(dict(plan_id=r['plan_id'],model=r['model_id'],requested=g['requested_payload_rank_count'],consumed=g['consumed_payload_rank_count'],budget=len(eligible),longest=length,start=start,stop=stop))
    require(entropy['strict']==Counter(attempts=120,complete=114), 'strict denominator')
    require(sorted(r['longest'] for r in failures)==[35,39,50,51,163,267], 'longest runs')
    require(all(r['budget']==6*r['requested'] for r in failures), 'original gate budgets')
    require(len(failures)==6 and sum(v['attempts'] for v in entropy.values())==360, 'gate denominators')
    save('entropy.json',dict(conditions=dict(entropy),failures=failures))
    covers=rows(stage2/'transport/cover_results.jsonl');segments=rows(stage2/'transport/segment_diagnostics.jsonl')
    require(len(covers)==25 and len(segments)==135 and len({r['source_trial_id'] for r in covers})==5,'transport population')
    prefix=[r for r in segments if r['arm']=='prefix_only']
    require(len(prefix)==27 and all(r['first_token_difference']==0 and not r['recovered_ranks'] and 'not allowed' in r['recovery_error'] for r in prefix), 'prefix mechanism')
    checks=[]
    for p in sorted(root.glob(str(stage2/'transport/reuse_validation_*.jsonl'))):checks.extend(rows(p.relative_to(root)))
    require(len(checks)==53 and all(r['historical_rank_and_error_agreement'] for r in checks),'capacity checks')
    require(len(rows(stage2/'transport/requests.jsonl'))==26,'new transport requests')
    save('transport.json',{'sources':5,'logical_arms':25,'new_requests':26,'capacity_checks':53,'segment_instances':135,
                         'prefix_first_token_rejections':27,'exact_by_arm':{a:sum(r['decoded']['exact_payload_recovery'] for r in covers if r['arm']==a) for a in sorted({r['arm'] for r in covers})}})
    save('input_hashes.json',inputs)
    save('audit_summary.json',{'status':'passed','independent_of_stage2_analysis_helpers':True,'gpu_execution_seconds':0,
                              'input_files':len(inputs),'effects_reconstructed':len(all_effects),
                              'limitations':'Retained token scores are audited and reaggregated, not recomputed by a model. Rank and byte endpoints use original records. No new empirical coherence claim.'})
    print('Stage 3 independent audit passed',len(inputs),'inputs',len(all_effects),'effects')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.root,args.output)
