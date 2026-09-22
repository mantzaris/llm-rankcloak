import json
from pathlib import Path

import numpy as np
import pytest

from rankcloak.revision_v4_stage1 import false_runs, trace_diagnostics, select_comparators, aligned_windows, export_filter
from rankcloak.token_filters import is_safe_text_token_piece, choose_token_at_rank_with_optional_filter, rank_token_with_optional_filter, build_allowed_token_mask


def record(identity='a', complete=False):
    mask = [True, False, False, True]
    expected = [2,3] if complete else [2,3,4]
    g = {'embedding_token_ids':[0,1,2,3], 'embedding_text':'abcd',
         'embedding_entropies_bits':[2.,1.,0.,2.], 'embedding_eligible_mask':mask,
         'embedding_token_roles':['payload','ordinary_sampled_skip','ordinary_sampled_skip','payload'],
         'embedding_log_probabilities':[-1.,-2.,-1.,-3.], 'embedding_observed_ranks':[2,1,1,3],
         'embedding_rank_pressure_log_probability_gaps_nats':[1.,0.,0.,2.],
         'requested_payload_rank_count':len(expected),'consumed_payload_rank_count':2,
         'payload_completion':complete,'consumed_payload_rank_indices':[0,1], 'capacity_failure':None if complete else 'budget'}
    p={k:'x' for k in ['experimental_cell_id','model_id','payload_name','payload_class','representation_name','prompt_template_id']}
    p.update(plan_id=identity,gate_level='strict')
    return {'plan_id':identity,'plan_row':p,'generation':g,'threshold_bits':2.,'expected_ranks':expected,
            'fixed_payload':{'maximum_embedding_token_count':4},
            'saved_token_id_replay':{'replay':{'embedding_eligible_mask':mask,'ranks':[2,3]}}}


def test_inclusive_threshold_and_consumption_with_sampled_skips():
    summary, rows, runs = trace_diagnostics(record(), window=2, low_progress=0)
    assert [r['consumed_after'] for r in rows] == [1,1,1,2]
    assert [r['payload_rank'] for r in rows] == [2,None,None,3]
    assert summary['longest_below_start']==1 and summary['longest_below_stop']==3
    assert summary['first_low_progress_window_start']==1
    assert rows[0]['rolling_window_full'] is False
    assert runs == [{'plan_id':'a','start':1,'stop':3,'length':2}]


@pytest.mark.parametrize('mutation', ['role','mask','count','replay','rank','finite'])
def test_trace_corruption_fails_closed(mutation):
    r=record()
    if mutation=='role': r['generation']['embedding_token_roles'][1]='payload'
    if mutation=='mask': r['generation']['embedding_eligible_mask'][0]=False
    if mutation=='count': r['generation']['consumed_payload_rank_count']=3
    if mutation=='replay': r['saved_token_id_replay']['replay']['ranks']=[2,4]
    if mutation=='rank': r['generation']['embedding_observed_ranks'][-1]=4
    if mutation=='finite': r['generation']['embedding_entropies_bits'][0]=float('nan')
    with pytest.raises(ValueError): trace_diagnostics(r)


def test_run_edges_and_identity_selection_order_independence():
    assert false_runs([False,True,False,False])==[(0,1),(2,4)]
    a,b=record('a',True),record('b',True)
    assert select_comparators([a,b],[record()],7)==select_comparators([b,a],[record()],7)
    with pytest.raises(ValueError): select_comparators([], [record()],7)


def test_prefix_alignment_rejects_nonprefix_token_decoding():
    class Tokenizer:
        def detokenize(self, ids): return bytes(97+i for i in ids)
    r=record(); summary=trace_diagnostics(r)[0]
    offsets, windows=aligned_windows(Tokenizer(),r,summary,1)
    assert offsets==[0,1,2,3,4]
    assert windows[0]['text']=='abcd'
    class Broken(Tokenizer):
        def detokenize(self,ids): return b'x' if len(ids)==1 else super().detokenize(ids)
    with pytest.raises(ValueError,match='non-prefix'): aligned_windows(Broken(),r,summary)


@pytest.mark.parametrize('piece', ['', '\ufffd', '\r', '\x00', '\x1f', 'x`y', 'x\\rMy', 'WWW.site', 'name.COM', '&GT', '{\\x', '___','|||', '  #a', '\\a\\b','\\\\'])
def test_filter_uncovered_rejections(piece):
    assert not is_safe_text_token_piece(piece)


@pytest.mark.parametrize('piece', ['\n','\t','   ','\x7f','\u200b','word#tag','_','||','\\x'])
def test_filter_near_boundary_allowances(piece):
    assert is_safe_text_token_piece(piece)


def test_filter_mask_ties_and_error_contract():
    logits=np.array([2.,2.,3.,2.]); mask=np.array([False,True,False,True])
    assert choose_token_at_rank_with_optional_filter(logits,1,mask)==1
    assert rank_token_with_optional_filter(logits,3,mask)==2
    with pytest.raises(ValueError): choose_token_at_rank_with_optional_filter(logits,3,mask)
    with pytest.raises(ValueError): rank_token_with_optional_filter(logits,0,mask)
    with pytest.raises(ValueError): build_allowed_token_mask(object(),'unknown_filter')


def test_export_keeps_literal_order_and_all_historical_substrings(tmp_path):
    root=Path(__file__).resolve().parents[1]
    export_filter(root,tmp_path)
    contract=json.loads((tmp_path/'provenance/filter_contract.json').read_text())
    assert contract['blocked_substrings']==['```','`','\\section','\\frac','{\\','\\rm','http','www.','.com','&lt','&gt','&amp','</','<','>','[',']','___','|||']
    for fragment in contract['blocked_substrings']:
        assert not is_safe_text_token_piece('prefix'+fragment.upper()+'suffix')


def test_alignment_uses_declared_progress_window_and_labels_no_event():
    class Tokenizer:
        def detokenize(self, ids): return bytes(97+i for i in ids)
    r = record()
    summary = trace_diagnostics(r, window=2, low_progress=0)[0]
    _, windows = aligned_windows(Tokenizer(), r, summary, flank=0, progress_window=2)
    progress = next(w for w in windows if w['kind'] == 'first_low_progress')
    assert (progress['event_start'], progress['event_stop'], progress['text']) == (1, 3, 'bc')
    summary.update(longest_below_start=0, longest_below_stop=0, first_low_progress_window_start=None)
    _, windows = aligned_windows(Tokenizer(), r, summary, flank=0, progress_window=2)
    assert windows[0]['kind'] == 'initial_window_no_below_threshold_run'
    assert windows[0]['text'] == 'ab'


def test_review_quotes_preserve_full_letter_and_reject_changed_comment():
    from scripts.prepare_revision_v4_stage1 import review_blocks, escape
    from scripts.validate_revision_v4_stage1 import verify_quotes
    original = 'Editor letter\n\nReviewer 2\n\nAcknowledgment.\n\nReviewer 1\n\nKeep wording.\n' + ''.join(
        f'{i}- Exact comment & 20% detail.\n' for i in range(1, 11))
    ledger = {'blocks': review_blocks(original)}
    response = ''.join('% BEGIN QUOTE '+b['id']+'\n'+r'\begin{reviewcomment}'+'\n'+escape(b['text'].rstrip())+'\n'+r'\end{reviewcomment}'+'\n% END QUOTE '+b['id']+'\n' for b in ledger['blocks'])
    response += ('Direct answer pending. Evidence and change pending. Location pending.\n')*16
    assert verify_quotes(original, ledger, response) == 13
    with pytest.raises(ValueError, match='quotation changed'):
        verify_quotes(original, ledger, response.replace('Exact comment', 'Reworded comment', 1))
    with pytest.raises(ValueError, match='complete original'):
        verify_quotes(original+'Omitted note.', ledger, response)
