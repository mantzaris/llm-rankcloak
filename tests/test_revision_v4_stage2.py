import json
from pathlib import Path
import pytest
import numpy as np
from rankcloak.revision_v4_coherence import window_from_ids, evaluator_inputs, word_eligible
from rankcloak.revision_v4_stage2_common import digest, load_cache, append_jsonl, immutable_json, hash_order
from rankcloak.revision_v4_stage2_scoring import score_ids


class ByteTokenizer:
    def tokenize(self,raw,add_bos=False,**kwargs):return ([999] if add_bos else [])+list(raw)
    def detokenize(self,ids):return bytes(ids)
    def token_bos(self):return 999


def test_exact_byte_windows_and_no_implicit_trimming():
    t=ByteTokenizer();ids=list(b' one two three four five six')
    w=window_from_ids(t,ids,14,13)
    assert w['left']==' one two three' and w['right']==' four five si'
    assert w['boundary_byte']==14 and w['right_byte_stop']==27
    assert word_eligible(w,3)


def test_utf8_split_is_an_explicit_exclusion():
    with pytest.raises(ValueError,match='utf8_split'):
        window_from_ids(ByteTokenizer(),list('éword'.encode()),1,3)


def test_identical_tail_token_path_in_both_contexts():
    t=ByteTokenizer();r=evaluator_inputs(t,'Prompt','one ','two three')
    assert r['target_ids']==list(b'two three')
    assert r['conditional_context_ids']==list(b'Promptone ')
    assert r['reference_context_ids']==list(b'Prompt')
    assert r['evaluator_boundary_token']==4


def test_evaluator_token_straddling_is_rejected():
    class MergeTokenizer(ByteTokenizer):
        def tokenize(self,raw,add_bos=False,**kwargs):return [1000] if raw==b'onetwo' else super().tokenize(raw,add_bos,**kwargs)
        def detokenize(self,ids):return b'onetwo' if ids==[1000] else super().detokenize(ids)
    with pytest.raises(ValueError,match='boundary_straddle'):
        evaluator_inputs(MergeTokenizer(),'Prompt','one','two')


def test_known_rendering_prefix_is_bound_and_other_changes_fail():
    class PrefixTokenizer(ByteTokenizer):
        def tokenize(self,raw,add_bos=False,**kwargs):return super().tokenize(b' '+raw,add_bos,**kwargs)
    r=evaluator_inputs(PrefixTokenizer(),'Prompt','one ','two')
    assert r['tokenizer_rendering_prefix_hex']=='20' and r['target_ids']==list(b'two')
    class BadTokenizer(PrefixTokenizer):
        def tokenize(self,raw,add_bos=False,**kwargs):return super().tokenize(b'x'+raw,add_bos,**kwargs)
    with pytest.raises(ValueError,match='roundtrip'):evaluator_inputs(BadTokenizer(),'Prompt','one ','two')


def test_checkpoint_rejects_identity_and_contract_corruption(tmp_path):
    p=tmp_path/'cache.jsonl';request={'kind':'test','input':'exact'}
    row={'request':request,'request_id':digest(request),'contract_sha256':'a'}
    append_jsonl(p,row);assert len(load_cache(p,'a'))==1
    with pytest.raises(ValueError,match='contract'):load_cache(p,'b')
    append_jsonl(p,row)
    with pytest.raises(ValueError,match='duplicate'):load_cache(p,'a')
    p.write_text(json.dumps({**row,'request_id':'altered'})+'\n')
    with pytest.raises(ValueError,match='identity'):load_cache(p,'a')


def test_immutable_freeze_refuses_plan_changes(tmp_path):
    p=tmp_path/'plan.json';immutable_json(p,{'seed':7});immutable_json(p,{'seed':7})
    with pytest.raises(ValueError):immutable_json(p,{'seed':8})


def test_bulk_score_row_index_matches_serial_path():
    class Model:
        def __init__(self):self.n_tokens=0;self.scores=np.zeros((30,10))
        def reset(self):self.n_tokens=0
        def eval(self,ids):
            for token in ids:
                self.scores[self.n_tokens]=np.arange(10,dtype=float)*(token+1)/10
                self.n_tokens+=1
    assert score_ids(Model(),[1,2],[3,4,5],True)==score_ids(Model(),[1,2],[3,4,5],False)


def complete_contract():
    from rankcloak.revision_v4_compatibility import REQUIRED_FIELDS
    return {k:{'identity':k} for k in sorted(REQUIRED_FIELDS)}


def test_compatibility_acceptance_serialization_and_unknown_fields():
    from rankcloak.revision_v4_compatibility import configuration_fingerprint,require_compatible
    c=complete_contract();header=configuration_fingerprint(c)
    assert configuration_fingerprint(dict(reversed(list(c.items()))))==header
    assert require_compatible(header,c)
    with pytest.raises(ValueError,match='complete'):configuration_fingerprint({**c,'extra':{}})
    header['configuration_sha256']='0'*64
    with pytest.raises(ValueError,match='corrupt'):require_compatible(header,c)


@pytest.mark.parametrize('field',['model','tokenizer','quantization','backend','prompt_rendering','bos_eos_policy','ordering','filter','codec_framing','gate','boundary_reset'])
def test_every_decoder_contract_dimension_rejects_mismatch(field):
    from rankcloak.revision_v4_compatibility import configuration_fingerprint,require_compatible
    c=complete_contract();header=configuration_fingerprint(c)
    c[field]={'identity':'changed'}
    with pytest.raises(ValueError,match='mismatch'):require_compatible(header,c)


def test_payload_estimand_does_not_weight_longer_trials_more():
    from rankcloak.revision_v4_stage2_analysis import payload_values,effect
    rows=[]
    for trial,values in [('t1',[0.,0.,0.]),('t2',[4.])]:
        for i,value in enumerate(values):
            for arm,score in [('a',value),('b',0.)]:rows.append({'boundary_id':trial+str(i),'trial_id':trial,'payload_name':'p','payload_class':'c','arm':arm,'metric':score})
    values,_=payload_values(rows,('a','b'),'metric');assert values=={'p':2.}
    assert effect(rows,('a','b'),'metric')['effect']==2.
    with pytest.raises(ValueError,match='unpaired'):payload_values(rows[:-1],('a','b'),'metric')


def test_shared_stratified_draws_reproduce_and_keep_class_sizes():
    from rankcloak.revision_v4_stage2_analysis import bootstrap_draws
    classes={'a':'x','b':'x','c':'y'};names,draws=bootstrap_draws(classes,count=2000)
    assert np.array_equal(draws,bootstrap_draws(dict(reversed(list(classes.items()))),count=2000)[1])
    assert np.all(np.sum(draws==names.index('c'),axis=1)==1)


def test_frozen_sample_matches_score_blind_hash_rule_and_complete_membership():
    from collections import defaultdict
    from rankcloak.revision_v4_stage2_common import OUT,read_json,read_jsonl
    freeze=read_json(OUT/'plans/coherence_study/freeze.json');struct=read_jsonl(OUT/'plans/amendment1/boundaries.jsonl')
    classes={r['payload_name']:r['payload_class'] for r in struct};strata=defaultdict(list)
    for p,c in classes.items():strata[c].append(p)
    expected={p for names in strata.values() for p in sorted(names,key=lambda p:hash_order(20260922,p))[:freeze['payloads_per_class']]}
    assert expected==set(freeze['selected_payload_names'])
    allunits=read_jsonl(OUT/'plans/amendment1/full_eligible_units.jsonl');chosen=read_jsonl(OUT/'plans/coherence_study/units.jsonl')
    assert {u['unit_id'] for u in chosen}=={u['unit_id'] for u in allunits if u['payload_name'] in expected}
    for p in expected:
        rows=[r for r in struct if r['payload_name']==p]
        assert len({(r['model_id'],r['schedule']) for r in rows})==6


def test_fresh_pilot_has_no_recipient_or_donor_payload_overlap():
    from rankcloak.revision_v4_stage2_common import OUT,read_jsonl
    sets=[]
    for plan in ['pilot_initial','pilot_amendment1']:
        rows=read_jsonl(OUT/'plans'/plan/'units.jsonl')
        assert {r['arm'] for r in rows}=={'pilot_ordinary','pilot_shuffled'}
        sets.append({r[k] for r in rows for k in ['left_payload','right_payload']})
    assert not sets[0]&sets[1]


def test_wrapper_removal_is_declared_and_does_not_restore_trimmed_bytes():
    from rankcloak.revision_v4_transport import transform,remove_wrapper
    assert transform('  hello  \r\nworld \n','declared_wrapper_removal')=='hello\nworld'
    assert transform('  hello  \nworld \n','prefix_only')=='>   hello  \n> world \n> '
    with pytest.raises(ValueError,match='absent'):remove_wrapper('> first\nsecond')


def test_quantization_endpoint_reconciles_populations_and_invalid_bytes():
    from rankcloak.revision_v4_stage2_common import OUT,read_json,read_jsonl
    manifest=read_json(OUT/'quantization/manifest.json');groups=[r for r in manifest['summary'] if r['codec']=='all']
    assert sum(r['positions'] for r in groups)==244440
    assert sum(r['observed_rank_changes'] for r in groups)==69528
    rows=read_jsonl(OUT/'quantization/q4_to_q8_fixed_message_decode.jsonl')
    assert len(rows)==960 and all(r['population']=='rankcloak' and r['supported'] for r in rows)
    assert sum(r['decoded']['success'] for r in rows)==194
    for r in rows:
        if r['decoded']['success']:
            assert r['decoded']['recovered_bytes']['hex']!=r['source_payload_hex']
        assert bool(r['invalid_rank_count'])==(not r['decoded']['success'])


def test_join_refuses_partial_planned_sample(tmp_path,monkeypatch):
    import rankcloak.revision_v4_stage2_analysis as a
    from rankcloak.revision_v4_stage2_common import atomic_json,write_jsonl
    monkeypatch.setattr(a,'OUT',tmp_path);(tmp_path/'raw').mkdir()
    atomic_json(tmp_path/'scoring_contracts.json',{'semantic':{},'lm':{}})
    write_jsonl(tmp_path/'units.jsonl',[{'unit_id':'x'}]);write_jsonl(tmp_path/'unit_requests.jsonl',[{'unit_id':'x','requests':{'left':'absent'}}])
    write_jsonl(tmp_path/'requests.jsonl',[{'request_id':'absent','request':{}}])
    with pytest.raises(ValueError,match='incomplete planned sample'):a.join_scores(tmp_path)


def test_score_join_validates_denominators_and_retained_token_scores():
    from rankcloak.revision_v4_stage2_analysis import validate_score_record
    row={'request':{'kind':'lm','context_ids':[1,2],'target_ids':[3,4]},'target_token_count':2,'context_token_count':2,'log_probabilities':[-1.,-3.],'mean_log_probability':-2.}
    validate_score_record(row)
    with pytest.raises(ValueError,match='denominator'):validate_score_record({**row,'target_token_count':1})
    with pytest.raises(ValueError,match='inconsistent'):validate_score_record({**row,'mean_log_probability':-1.})
    with pytest.raises(ValueError,match='nonfinite'):validate_score_record({**row,'mean_log_probability':float('nan')})


def test_stage2_quote_invariant_rejects_rephrasing_even_with_completed_answers():
    from scripts.validate_revision_v4_stage2 import verify_quotes
    from rankcloak.revision_v4_stage2_common import ROOT,read_json
    text=(ROOT/'paperV4/response/requests.txt').read_text();ledger=read_json(ROOT/'paperV4/response/review_comments.json');response=(ROOT/'paperV4/response/response_to_reviewers_v4.tex').read_text()
    assert verify_quotes(text,ledger,response)==13
    with pytest.raises(ValueError,match='verbatim'):verify_quotes(text,ledger,response.replace('Thank the authors for addressing most of my concerns.','Thank the authors.',1))


def test_gpu_ledger_rejects_overlap_and_missing_failed_attempt():
    from scripts.validate_revision_v4_stage2 import verify_gpu_accounting
    jobs=[{'name':'a','pid':1,'started_epoch':1,'finished_epoch':3,'charged_seconds':2},
          {'name':'b','pid':2,'started_epoch':3,'finished_epoch':5,'charged_seconds':2}]
    ledger={'ceiling_seconds':28800,'jobs':jobs}
    assert verify_gpu_accounting(ledger,{'invocations':[]})==4
    jobs[1]['started_epoch']=2
    with pytest.raises(ValueError,match='overlapping'):verify_gpu_accounting(ledger,{'invocations':[]})
    jobs[1]['started_epoch']=3
    with pytest.raises(ValueError,match='unsuccessful'):
        verify_gpu_accounting(ledger,{'invocations':[{'name':'refused','conservatively_charged_command_wall_seconds':.1}]})
