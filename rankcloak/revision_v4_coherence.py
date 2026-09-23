"""Frozen boundary instrumentation and exact scoring-request identities for V4."""
from __future__ import annotations
from collections import Counter, defaultdict
from functools import lru_cache
import re
from pathlib import Path
from rankcloak.revision_v4_stage2_common import ROOT, OUT, digest, file_hash, read_json, atomic_json, write_jsonl, hash_order
from rankcloak.model_io import detokenize_bytes, tokenize_bytes, make_context_token_ids


def window_from_ids(tokenizer, ids, boundary, right_count, left_limit=32):
    """Keep exact full-prefix byte boundaries. Never concatenate token pieces."""
    if boundary <= 0 or right_count <= 0 or boundary + right_count > len(ids):
        raise ValueError('short_token_window')
    raw = detokenize_bytes(tokenizer, list(ids))
    prefix = [detokenize_bytes(tokenizer, list(ids[:i])) for i in [max(0,boundary-left_limit), boundary, boundary+right_count]]
    if any(not raw.startswith(p) for p in prefix): raise ValueError('generator_prefix_alignment')
    a,b,c = map(len,prefix)
    if not 0<=a<=b<c<=len(raw): raise ValueError('empty_or_unordered_byte_window')
    try:left,right=raw[a:b].decode('utf8'),raw[b:c].decode('utf8')
    except UnicodeDecodeError as exc: raise ValueError('generator_utf8_split') from exc
    if '\ufffd' in left+right:raise ValueError('replacement_character')
    return {'left':left,'right':right,'left_byte_start':a,'boundary_byte':b,'right_byte_stop':c,
            'left_generator_tokens':boundary-max(0,boundary-left_limit),'right_generator_tokens':right_count}


def word_eligible(window, minimum):
    return all(len(re.findall(r'\b\w+\b',window[k],flags=re.UNICODE))>=minimum for k in ['left','right'])


def evaluator_inputs(tokenizer, prompt, left, right, limit=512):
    raw=(left+right).encode(); boundary=len(left.encode())
    ids=list(map(int,tokenize_bytes(tokenizer,raw,add_bos=False)))
    decoded=detokenize_bytes(tokenizer,ids)
    rendering_prefix=b''
    if decoded!=raw:
        if decoded==b' '+raw:rendering_prefix=b' '
        else:raise ValueError('evaluator_roundtrip')
    raw=rendering_prefix+raw;boundary+=len(rendering_prefix)
    offsets=[0]
    for i in range(1,len(ids)+1):
        prefix=detokenize_bytes(tokenizer,ids[:i])
        if not raw.startswith(prefix) or len(prefix)<offsets[-1]:raise ValueError('evaluator_prefix_alignment')
        offsets.append(len(prefix))
    if boundary not in offsets:raise ValueError('evaluator_boundary_straddle')
    k=offsets.index(boundary); targets=ids[k:]
    if not targets or not k:raise ValueError('empty_evaluator_side')
    prompt_ids=make_context_token_ids(tokenizer,prompt)
    if len(prompt_ids)+len(ids)>limit:raise ValueError('evaluator_context_limit')
    return {'conditional_context_ids':prompt_ids+ids[:k], 'reference_context_ids':prompt_ids,
            'target_ids':targets,'joint_text_ids':ids,'evaluator_boundary_token':k,
            'tokenizer_rendering_prefix_hex':rendering_prefix.hex(),'standalone_right_ids_match':targets==list(map(int,tokenize_bytes(tokenizer,right.encode(),add_bos=False)))}


def build_inventory(config_path, output_name='initial', excluded_pilot_payloads=()):
    from rankcloak.revision_tokenizer_preflight import load_vocab_only_tokenizer
    cfg=read_json(config_path); seed=cfg['seed']; right_max=cfg['right_generator_tokens']
    requirements=read_json(ROOT/'configs/revision_v3/generation_requirements.json')
    pins={a['model_id']:a for a in requirements['artifacts']}
    models={m:load_vocab_only_tokenizer(ROOT/pins[m]['expected_path']) for m in cfg['evaluator_map']}
    @lru_cache(maxsize=100000)
    def window(m,ids,n,k):
        return window_from_ids(models[m],ids,n,k,cfg['left_generator_tokens'])
    actual=[]; controls=defaultdict(list); contexts={}; source_hashes={}
    try:
        for p in sorted((ROOT/'results/revision_v1/primary_v2').glob('*/records.jsonl')):
            source_hashes[str(p.relative_to(ROOT))]=file_hash(p)
            import json
            for line in p.open():
                r=json.loads(line)
                if r['record_type']=='rankcloak_trial':
                    for s in r['segments']:
                        contexts[(r['model_id'],tuple(s['context_token_ids']))]=s['prompt']
                        if not r['segmented']:continue
                        actual.append({'boundary_id':r['trial_id']+'__segment_'+str(s['segment_index']),
                            'trial_id':r['trial_id'],'payload_name':r['payload_name'],'payload_class':r['payload_class'],
                            'model_id':r['model_id'],'schedule':r['protocol_variant'],'segment_index':s['segment_index'],
                            'prompt':s['prompt'],'context_ids':s['context_token_ids'],'forced_stop':s['forced_stop'],
                            'tokens':s['full_token_ids'][:s['forced_stop']+right_max],
                            'source_path':str(p.relative_to(ROOT)),'source_record_sha256':digest(r),
                            'right_count':min(right_max,len(s['tail_token_ids']))})
                elif r['record_type']=='ordinary_control' and r['control_view']=='full_message':
                    g=r['generation'];key=(r['model_id'],tuple(g['context_token_ids']))
                    controls[key].append({'control_id':r['control_id'],'payload_name':r['payload_name'],'payload_class':r['payload_class'],
                        'tokens':g['token_ids'][:64+right_max],'source_path':str(p.relative_to(ROOT)),
                        'source_record_sha256':digest(r),'source_trial_id':r['source_trial_id']})
        if (len(actual),len({r['trial_id'] for r in actual}),len({r['payload_name'] for r in actual}))!=(8280,1440,240):
            raise ValueError('structural denominators changed')
        source_inventory=__import__('csv').DictReader((ROOT/'results/revision_v4/source_tables/boundary_inventory.csv').open())
        if {r['boundary_id'] for r in source_inventory}!={r['boundary_id'] for r in actual}:raise ValueError('Stage 1 identities changed')
        usage=Counter(); boundary_rows=[]; full_units=[]; pilot_candidates=[]; shuffled_groups=defaultdict(list)
        for r in actual:
            key=(r['model_id'],tuple(r['context_ids']),r['schedule'],r['forced_stop'])
            shuffled_groups[key].append(r)
        for group in shuffled_groups.values():group.sort(key=lambda r:hash_order(seed,r['boundary_id']))
        for r in sorted(actual,key=lambda x:hash_order(seed,x['boundary_id'])):
            m=r['model_id'];n=r['forced_stop'];k=r['right_count'];bid=r['boundary_id'];pool=controls[(m,tuple(r['context_ids']))]
            reasons=[];meta={key:r[key] for key in ['boundary_id','trial_id','payload_name','payload_class','model_id','schedule','segment_index','source_path','source_record_sha256','right_count']}
            meta.update(prompt_category=r['prompt']['prompt_category'],prompt_id=r['prompt']['prompt_id'],prompt_text=r['prompt']['prompt_text'],context_sha256=digest(r['context_ids']),forced_stop=n)
            try:
                own=window(m,tuple(r['tokens']),n,k)
                if not word_eligible(own,cfg['minimum_words_each_side']):raise ValueError('short_actual_words')
            except ValueError as exc:own=None;reasons.append(str(exc))
            candidates=[]
            for c in pool:
                if c['payload_name']==r['payload_name'] or c['payload_name'] in excluded_pilot_payloads:continue
                try:
                    w=window(m,tuple(c['tokens']),n,k)
                    if word_eligible(w,cfg['minimum_words_each_side']):candidates.append((c,w))
                except ValueError:pass
            candidates.sort(key=lambda cw:(usage[cw[0]['control_id']],hash_order(seed,bid+'|'+cw[0]['control_id'])))
            ordinary=candidates[0] if candidates else None
            if ordinary:usage[ordinary[0]['control_id']]+=1
            else:reasons.append('no_ordinary_match')
            pilot_shuffle=None;pilot_identical=0
            if ordinary:
                ordered=sorted(candidates,key=lambda cw:hash_order(seed,'pilot_shuffle|'+bid+'|'+cw[0]['control_id']))
                for c,w in ordered:
                    if c['payload_name']==ordinary[0]['payload_name']:continue
                    if w['right']==ordinary[1]['right']:pilot_identical+=1;continue
                    pilot_shuffle=(c,w);break
            if not pilot_shuffle:pilot_reason='no_nonidentical_ordinary_shuffle'
            else:pilot_reason=None
            actual_shuffle=None;identical=0
            group=shuffled_groups[(m,tuple(r['context_ids']),r['schedule'],n)]
            index=next(i for i,d in enumerate(group) if d['boundary_id']==bid)
            if own:
                for d in group[index+1:]+group[:index]:
                    if d['payload_name']==r['payload_name']:continue
                    try:
                        w=window(m,tuple(d['tokens']),n,k)
                        if len(re.findall(r'\b\w+\b',w['right']))<cfg['minimum_words_each_side']:continue
                        if w['right']==own['right']:identical+=1;continue
                        actual_shuffle=(d,w);break
                    except ValueError:pass
                if not actual_shuffle:reasons.append('no_nonidentical_actual_shuffle')
            def unit(arm,w,left_source,right_source):
                right_window=window(m,tuple(right_source['tokens']),n,k)
                return {**meta,'unit_id':bid+'__'+arm,'arm':arm,'left':w['left'],'right':w['right'],
                    'left_source_id':left_source.get('control_id',left_source.get('boundary_id')),
                    'right_source_id':right_source.get('control_id',right_source.get('boundary_id')),
                    'left_payload':left_source['payload_name'],'right_payload':right_source['payload_name'],
                    'left_source_sha256':left_source['source_record_sha256'],'right_source_sha256':right_source['source_record_sha256'],
                    'left_source_byte_start':w['left_byte_start'],'left_source_byte_stop':w['boundary_byte'],
                    'right_source_byte_start':right_window['boundary_byte'],'right_source_byte_stop':right_window['right_byte_stop'],
                    'joined_boundary_byte':len(w['left'].encode('utf8')),
                    'left_generator_tokens':w['left_generator_tokens'],'right_generator_tokens':k}
            units=[]
            if own:units.append(unit('actual',own,r,r))
            if ordinary:units.append(unit('ordinary',ordinary[1],ordinary[0],ordinary[0]))
            if own and actual_shuffle:
                w={**own,'right':actual_shuffle[1]['right']}
                units.append(unit('shuffled',w,r,actual_shuffle[0]))
            def add_eval(u):
                try:
                    u['evaluator_model_id']=cfg['evaluator_map'][m]
                    u['evaluator_inputs']=evaluator_inputs(models[u['evaluator_model_id']],u['prompt_text'],u['left'],u['right'],cfg['lm_context_limit'])
                    return None
                except ValueError as exc:return u['arm']+':'+str(exc)
            for u in units:
                error=add_eval(u)
                if error:reasons.append(error)
            if len(units)==3 and not reasons:full_units.extend(units)
            pilot=[]
            if ordinary and pilot_shuffle:
                a=unit('pilot_ordinary',ordinary[1],ordinary[0],ordinary[0])
                b=unit('pilot_shuffled',{**ordinary[1],'right':pilot_shuffle[1]['right']},ordinary[0],pilot_shuffle[0])
                pilot_errors=[e for u in [a,b] if (e:=add_eval(u))]
                if not pilot_errors:
                    for u in [a,b]:
                        u['anchor_payload']=u['payload_name'];u['payload_name']=ordinary[0]['payload_name'];u['payload_class']=ordinary[0]['payload_class'];u['trial_id']=ordinary[0]['control_id']
                    pilot=[a,b]
                else:pilot_reason='|'.join(pilot_errors)
            meta.update(eligible=not reasons and len(units)==3,exclusion_reasons=reasons,pilot_eligible=bool(pilot),pilot_reason=pilot_reason,
                        identical_actual_donors_skipped=identical,identical_pilot_donors_skipped=pilot_identical,
                        ordinary_donor_id=ordinary[0]['control_id'] if ordinary else None,
                        ordinary_donor_payload=ordinary[0]['payload_name'] if ordinary else None,
                        shuffled_donor_id=actual_shuffle[0]['boundary_id'] if actual_shuffle else None,
                        shuffled_donor_payload=actual_shuffle[0]['payload_name'] if actual_shuffle else None)
            boundary_rows.append(meta)
            if pilot:pilot_candidates.append((meta,pilot))
        cells={}
        for meta,pair in pilot_candidates:
            key=(meta['model_id'],meta['schedule'],meta['prompt_category'])
            if key not in cells or hash_order(seed,meta['boundary_id'])<hash_order(seed,cells[key][0]['boundary_id']):cells[key]=(meta,pair)
        pilot=[u for key in sorted(cells) for u in cells[key][1]]
        folder=OUT/'plans'/output_name;folder.mkdir(parents=True,exist_ok=True)
        write_jsonl(folder/'boundaries.jsonl',boundary_rows)
        write_jsonl(folder/'full_eligible_units.jsonl',full_units)
        write_jsonl(folder/'pilot_units.jsonl',pilot)
        manifest={'config_path':str(Path(config_path).relative_to(ROOT)),'config_sha256':file_hash(config_path),'instrumentation_source_sha256':file_hash(Path(__file__)),
            'source_hashes':source_hashes,'source_inventory_sha256':file_hash(ROOT/'results/revision_v4/source_tables/boundary_inventory.csv'),
            'structural_boundaries':len(actual),'segmented_trials':1440,'payload_groups':240,'eligible_boundaries':len(full_units)//3,
            'pilot_identities':len(pilot)//2,'pilot_strata':len(cells),'excluded_pilot_payloads':sorted(excluded_pilot_payloads),
            'files':{p.name:file_hash(p) for p in folder.glob('*.jsonl')},'metrics_scored':False,
            'exclusion_reason_counts':dict(Counter(reason for r in boundary_rows for reason in r['exclusion_reasons']))}
        atomic_json(folder/'inventory_manifest.json',manifest)
        return manifest
    finally:
        for model in models.values():model.close()
