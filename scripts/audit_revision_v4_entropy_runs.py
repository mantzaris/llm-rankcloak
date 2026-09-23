"""Describe token behavior within the six retained longest low-entropy runs."""
import csv,itertools
from collections import Counter
from statistics import mean
from rankcloak.revision_v4_stage2_common import ROOT,OUT,atomic_json,file_hash
from rankcloak.revision_v4_stage2_analysis import csv_write

source=ROOT/'results/revision_v4/source_tables'
with (source/'entropy_positions.csv').open() as f:positions=list(csv.DictReader(f))
with (source/'entropy_six_failures.csv').open() as f:failures=list(csv.DictReader(f))
rows=[]
for case in failures:
    start,stop=int(case['longest_below_start']),int(case['longest_below_stop'])
    run=sorted([r for r in positions if r['plan_id']==case['plan_id'] and start<=int(r['position'])<stop],key=lambda r:int(r['position']))
    if [int(r['position']) for r in run]!=list(range(start,stop)):raise ValueError('run position join mismatch')
    if any(r['eligible']!='False' or float(r['entropy_margin_bits'])>=0 or r['consumed_after']!=r['consumed_before'] for r in run):raise ValueError('longest below-threshold run inconsistent')
    ids=[int(r['token_id']) for r in run];counts=Counter(ids)
    ranks=[int(r['observed_rank']) for r in run if r['observed_rank']]
    rows.append({'plan_id':case['plan_id'],'model_id':case['model_id'],'start':start,'stop':stop,'positions':len(run),
        'distinct_token_ids':len(counts),'most_frequent_token_id':min(counts,key=lambda t:(-counts[t],t)),
        'most_frequent_token_count':max(counts.values()),'longest_identical_id_streak':max(sum(1 for _ in g) for _,g in itertools.groupby(ids)),
        'observed_rank_available':len(ranks),'rank_one_count':ranks.count(1),'rank_one_fraction':ranks.count(1)/len(ranks) if ranks else None,
        'zero_byte_increment_positions':sum(r['byte_start']==r['byte_stop'] for r in run if r['byte_start']),
        'byte_alignment_available':sum(bool(r['byte_start']) for r in run),
        'mean_entropy_bits':mean(float(r['entropy_bits']) for r in run),'mean_margin_bits':mean(float(r['entropy_margin_bits']) for r in run),
        'token_roles':dict(Counter(r['token_role'] for r in run)),'payload_ranks_consumed':0})
if len(rows)!=6:raise ValueError('failure denominator changed')
csv_write(OUT/'entropy/longest_run_characteristics.csv',rows)
atomic_json(OUT/'entropy/longest_run_characteristics.json',{'selection':'Exactly the six original strict failures and their retained longest half-open below-threshold interval. No new model execution. Descriptive post-outcome analysis.',
    'inputs':{str((source/name).relative_to(ROOT)):file_hash(source/name) for name in ['entropy_positions.csv','entropy_six_failures.csv']},
    'source_sha256':file_hash(__file__),'cases':rows,'interpretation':'Conditional low entropy does not require repetition of one token ID. Rank-one observations remain outcomes of the ordinary sampler, not evidence of a greedy-skip policy. These summaries do not identify causal linguistic triggers.'})
print(rows)
