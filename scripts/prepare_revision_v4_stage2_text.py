"""Generate retained-data tables and exact text windows for the V4 document update."""
from pathlib import Path
import csv,json
from rankcloak.revision_v4_stage2_common import ROOT,OUT,read_json,read_jsonl,atomic_json,file_hash
from scripts.prepare_revision_v4_stage1 import escape
from rankcloak.revision_v4_stage2_analysis import csv_write

paper=ROOT/'paperV4/scientific_reports'
failures=list(csv.DictReader((ROOT/'results/revision_v4/source_tables/entropy_six_failures.csv').open()))
selected=list(csv.DictReader((ROOT/'results/revision_v4/source_tables/entropy_selected_runs.csv').open()))
windows=list(csv.DictReader((ROOT/'results/revision_v4/source_tables/entropy_text_windows.csv').open()))
rows=[];text=[r'\subsection*{Matched trajectories and exact local windows}',r'The following excerpts use pinned full-prefix byte alignment. The event interval is half open in embedding-token positions. Each displayed window retains up to 16 neighboring tokens on either side. Literal newline and tab markers expose formatting without interpreting it as a causal trigger. Full unabridged windows, token IDs, entropy margins, rank pressure and rolling progress are archived in the source tables.']
characteristics=read_json(OUT/'entropy/longest_run_characteristics.json')['cases']
assert {r['plan_id'] for r in characteristics}=={r['plan_id'] for r in failures}
text += [r'These excerpts are generated cover text, not accounts of activities conducted by the authors.',
    f"The longest below-threshold runs contain {min(r['distinct_token_ids'] for r in characteristics)} to {max(r['distinct_token_ids'] for r in characteristics)} distinct token IDs, with at most {max(r['longest_identical_id_streak'] for r in characteristics)} identical IDs consecutively. All {sum(r['positions'] for r in characteristics)} positions are recorded as ordinary sampled skips and consume no payload ranks. Rank-one observations range from {100*min(r['rank_one_fraction'] for r in characteristics):.1f} to {100*max(r['rank_one_fraction'] for r in characteristics):.1f}\\% within these runs. These observations describe the retained token paths and do not establish a causal linguistic trigger or change the sampled-skip policy.",
    r'\input{supplementary_tables/v4_entropy_runs.tex}']
elines=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{lrrrrr}\toprule',r'Trial suffix & Positions & Distinct IDs & Max repeat & Rank one & Percent \\ \midrule']
for r in characteristics:
    elines.append(r'\code{'+r['plan_id'].split('__')[-1]+'} & '+f"{r['positions']} & {r['distinct_token_ids']} & {r['longest_identical_id_streak']} & {r['rank_one_count']} & {100*r['rank_one_fraction']:.1f}"+r' \\')
elines += [r'\bottomrule\end{tabular}',r'\caption{Descriptive token behavior within each retained longest below-threshold interval. Max repeat is the longest consecutive run of the same token ID. Rank-one choices remain outcomes of the ordinary sampler. All aligned byte offsets advance, while payload progress is zero.}\label{tab:v4-entropy-runs}',r'\end{table}']
(paper/'supplementary_tables/v4_entropy_runs.tex').write_text('\n'.join(elines)+'\n')
for f in failures:
    key=['model_id','payload_name','representation_name','prompt_template_id']
    matched=[r for r in selected if all(r[k]==f[k] for k in key) and r['gate_level'] in ['ungated','moderate']]
    assert len(matched)==2
    u=next(r for r in matched if r['gate_level']=='ungated');m=next(r for r in matched if r['gate_level']=='moderate')
    rows.append({'failure_suffix':f['plan_id'].split('__')[-1],'model':f['model_id'],'requested':int(f['requested_ranks']),'strict_consumed':int(f['consumed_ranks']),'strict_tokens':int(f['tokens_used']),'ungated_tokens':int(u['tokens_used']),'moderate_tokens':int(m['tokens_used']),'strict_longest_below':int(f['longest_below_length']),'moderate_longest_below':int(m['longest_below_length']),'strict_mean_margin_bits':float(f['mean_entropy_margin_bits']),'moderate_mean_margin_bits':float(m['mean_entropy_margin_bits'])})
    w=next(r for r in windows if r['plan_id']==f['plan_id'] and r['kind']=='longest_below_threshold')
    assert w['alignment_status']=='verified_pinned_prefix_bytes' and w['utf8_replacement_in_display']=='False'
    # Full exact retained window, with whitespace represented explicitly for typesetting.
    display=w['text'].replace('\n',r'\n').replace('\t',r'\t')
    text += [r'\paragraph{Trial \code{'+f['plan_id'].split('__')[-1]+r'}.}',
        f"The strict trace consumed {f['consumed_ranks']} of {f['requested_ranks']} ranks in {f['tokens_used']} positions. The longest below-threshold run is [{f['longest_below_start']}, {f['longest_below_stop']}). The matched ungated and moderate runs completed in {u['tokens_used']} and {m['tokens_used']} positions. Their identities are "+r'\code{'+u['plan_id'].split('__')[-1]+r'} and \code{'+m['plan_id'].split('__')[-1]+r'}.',
        f"The exact display spans source bytes [{w['display_byte_start']}, {w['display_byte_stop']}) and tokens [{w['window_token_start']}, {w['window_token_stop']}).",
        r'\begin{quote}\small '+escape(display)+r'\end{quote}']
csv_write(OUT/'entropy/failure_comparators.csv',rows)
text += [r'\paragraph{Successful strict comparators.}',r'The three prospective identity-selected strict successes consumed 72/72, 128/128 and 48/48 ranks in 350, 254 and 137 positions. Their longest below-threshold runs were 35, 10 and 19 positions. Thus a low-entropy run, or even a negative mean threshold margin, was not sufficient for failure. All matched ungated and moderate cases completed, but their different token histories prevent interpreting these comparisons as isolated causal effects of a phrase.']
(paper/'v4_entropy_windows.tex').write_text('\n\n'.join(text)+'\n')
# Compact transport table retains the historical baseline and composite endpoint separately.
covers=read_jsonl(OUT/'transport/cover_results.jsonl');historical=read_jsonl(OUT/'transport/historical_outcomes.jsonl');ids=sorted({r['source_trial_id'] for r in covers});lines=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{lrrrrrrr}\toprule',r'Case & Visible & Composite & Prefix & Outer & Trailing & LF & Unwrap \\ \midrule']
transport_rows=[]
for index,trial in enumerate(ids,1):
    rs=[r for r in covers if r['source_trial_id']==trial];first=rs[0];md=next(r for r in historical if r['source_trial_id']==trial and r['transformation_id']=='markdown_copy_paste')
    values=[int(first['historical_visible_text_success']),int(md['exact_recovery'])]+[int(next(r for r in rs if r['arm']==a)['decoded']['exact_payload_recovery']) for a in ['prefix_only','outer_trim_only','trailing_trim_only','line_endings_only','declared_wrapper_removal']]
    model=first['model_id'].split('_')[0];label=('Llama' if model=='llama3' else 'Qwen' if model=='qwen2' else 'Mistral')+' '+('success' if values[0] else 'failure')
    lines.append(escape(label)+' & '+' & '.join(map(str,values))+r' \\')
    transport_rows.append({'case':label,'source_trial_id':trial,'baseline_work_id':first['historical_work_id'],'composite_work_id':md['work_id'],'values':values})
lines += [r'\bottomrule\end{tabular}',r'\caption{Illustrative exact byte recovery. One denotes success and zero failure. Visible baselines are executed historical transformations that left every source byte unchanged. The historical unmodified reference itself aliases saved-ID recovery. The five decompositions use saved offsets and no payload-informed repair. Mistral has no eligible successful visible baseline. These cases do not estimate population rates.}\label{tab:v4-transport}',r'\end{table}']
(paper/'supplementary_tables/v4_transport.tex').write_text('\n'.join(lines)+'\n');atomic_json(OUT/'transport/manuscript_table_sources.json',transport_rows)
print('Generated six entropy windows and five transport rows')
