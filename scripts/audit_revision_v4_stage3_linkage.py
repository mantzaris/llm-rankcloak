"""Check visible byte windows, control linkage, policies and detector AUC on CPU."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,json,hashlib,io
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision_v4/stage3/audit'


def main():
    sources={};inputs={}
    def load(path):
        data=path.read_bytes();inputs[path.relative_to(ROOT).as_posix()]=hashlib.sha256(data).hexdigest();return data
    for p in sorted(ROOT.glob('results/revision_v1/primary_v2/*/records.jsonl')):
        for line in load(p).splitlines():
            r=json.loads(line)
            if r['record_type']=='rankcloak_trial':
                for i,s in enumerate(r['segments']):sources[r['trial_id']+'__segment_'+str(i)]=(r,s)
            else:sources[r['control_id']]=(r,r['generation'])
    units=[json.loads(l) for l in load(ROOT/'results/revision_v4/stage2/plans/coherence_study/units.jsonl').splitlines()]
    ordinary_ids=set();actual_ids=set();identical=Counter();groups=defaultdict(dict)
    for u in units:
        contexts=[]
        for side in ['left','right']:
            r,s=sources[u[side+'_source_id']]
            text=s.get('full_text',s.get('text'))
            start,stop=u[side+'_source_byte_start'],u[side+'_source_byte_stop']
            if text.encode()[start:stop].decode()!=u[side]:raise ValueError('source byte window mismatch '+u['unit_id'])
            if r['payload_name']!=u[side+'_payload'] or r['model_id']!=u['model_id']:raise ValueError('source identity mismatch')
            contexts.append(s['context_token_ids'])
            if r['record_type']!='rankcloak_trial':
                ordinary_ids.add(r['control_id'])
                if (s['temperature'],s['top_p'],s['sampler'])!=(.8,.95,'numpy_pcg64_serial_top_p_v1_token_id_tiebreak'):raise ValueError('ordinary policy')
            else:actual_ids.add(r['trial_id'])
        if contexts[0]!=contexts[1]:raise ValueError('nonmatching source prompts')
        if u['arm']=='shuffled' and u['left_payload']==u['right_payload']:raise ValueError('same payload donor')
        groups[u['boundary_id']][u['arm']]=u
    for arms in groups.values():
        for a,b in [('actual','ordinary'),('actual','shuffled'),('ordinary','shuffled')]:
            if arms[a]['right']==arms[b]['right']:identical[a+'_'+b]+=1
    if identical:raise ValueError('identical right control pair')
    windows=list(csv.DictReader(io.StringIO(load(ROOT/'results/revision_v4/source_tables/entropy_text_windows.csv').decode())))
    raw_entropy={}
    for p in ROOT.glob('results/revision_v3/generation/raw/entropy/*/*.json'):
        if p.stem in {r['plan_id'] for r in windows}:raw_entropy[p.stem]=json.loads(load(p))
    for w in windows:
        data=raw_entropy[w['plan_id']]['generation']['full_text'].encode()
        excerpt=data[int(w['display_byte_start']):int(w['display_byte_stop'])].decode()
        # CSV uses explicit literal newline/tab display escapes.
        if excerpt!=w['text']:raise ValueError('entropy display bytes differ '+w['plan_id'])
        if w['alignment_status']!='verified_pinned_prefix_bytes':raise ValueError('alignment not verified')
    detectors={}
    for name in ['textcnn','deberta','surprisal']:
        data=list(csv.DictReader(load(ROOT/f'results/revision_v3/detector_predictions/{name}__matched.csv').decode().splitlines()))
        test=[r for r in data if r['evaluation_role']=='test'];pos=np.array([float(r['score']) for r in test if r['label']=='1']);neg=np.array([float(r['score']) for r in test if r['label']=='0'])
        auc=float(np.mean((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])))
        if len(pos)!=1491 or len(neg)!=1491:raise ValueError('detector test denominator')
        detectors[name]={'positive':len(pos),'negative':len(neg),'auc_pairwise_independent':auc}
    result={'status':'passed','boundary_source_byte_windows_checked':2*len(units),'same_prompt_pairs_checked':len(units),
            'exact_identical_right_pairs':dict(identical),'sampled_ordinary_source_ids':len(ordinary_ids),'actual_and_donor_source_trials':len(actual_ids),
            'greedy_policy_source':'rankcloak/revision_protocol.py tail loop chooses allowed rank one; source unchanged',
            'entropy_source_text_windows_checked':len(windows),'entropy_token_alignment_limit':'No tokenizer rerun. Declared byte offsets independently match the original visible text; pinned-token prefix alignment proof and token positions remain protected Stage 1 evidence.',
            'detector_auc':detectors,'input_hashes':inputs}
    (OUT/'linkage.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='input_hashes'})


if __name__=='__main__':main()
