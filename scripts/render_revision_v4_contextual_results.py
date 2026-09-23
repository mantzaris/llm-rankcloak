"""Render contextual-study tables and figure from retained offline estimates only."""
from pathlib import Path
import json
import shutil
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt
import numpy as np
from rankcloak.revision_v4_contextual_coherence import ROOT,OUT,CONFIG,RUBRIC,SCALES
from rankcloak.revision_v4_stage2_common import read_json,atomic_json,read_jsonl,hash_order
from scripts.prepare_revision_v4_stage1 import escape

SHORT={'llama3_8b_instruct_q4_k_m':'Llama','mistral_7b_instruct_v0_3_q4_k_m':'Mistral','qwen2_5_7b_instruct_q4_k_m':'Qwen','all':'Panel overall','segmented_hex_single_topic':'Single topic','segmented_hex_multi_topic':'Rotating topics'}

def name(v):return SHORT.get(v,v.replace('_',' '))
def group_name(row):
    prefix={'generator':'Source ', 'judge':'Judge '}.get(row['group'],'')
    return prefix+name(row['value'])
def interval(v,scale=100,digits=1):
    return f"{v['estimate']*scale:.{digits}f} [{v['lower']*scale:.{digits}f}, {v['upper']*scale:.{digits}f}]"
def stack(v,scale=100):
    return r'\shortstack{'+f"{v['estimate']*scale:.1f}"+r'\\{}'+f"[{v['lower']*scale:.1f}, {v['upper']*scale:.1f}]"+'}'

def main():
    s=read_json(OUT/'analysis/summary.json');cal=read_json(OUT/'analysis/calibration.json')
    missing=read_json(OUT/'analysis/missingness_details.json')
    if s['status']!='complete':raise ValueError('Do not render an incomplete convenience subset')
    dest=OUT/'manuscript';dest.mkdir(parents=True,exist_ok=True);paper=ROOT/'paperV4/scientific_reports'
    rows=[r for r in s['summaries'] if r['group'] in ['panel','generator']]
    colors={'encoded':'#2265a5','ordinary':'#bc622e'}
    fig,(a,b)=plt.subplots(1,2,figsize=(7.4,3.05),gridspec_kw={'width_ratios':[1.45,1]})
    for arm,offset in [('encoded',-.12),('ordinary',.12)]:
        vals=[r[arm] for r in rows];x=np.array([v['estimate']*100 for v in vals]);lo=np.array([v['lower']*100 for v in vals]);hi=np.array([v['upper']*100 for v in vals])
        a.errorbar(x,np.arange(len(rows))+offset,xerr=[x-lo,hi-x],fmt='o',capsize=3,color=colors[arm],label='RankCloak' if arm=='encoded' else 'Matched ordinary',markersize=4)
    a.set_yticks(range(len(rows)),[name(r['value']) for r in rows]);a.invert_yaxis();a.set_xlim(0,103);a.set_xlabel('Accepted by rubric (%)');a.grid(axis='x',alpha=.2)
    vals=[r['difference'] for r in rows];x=np.array([v['estimate']*100 for v in vals]);lo=np.array([v['lower']*100 for v in vals]);hi=np.array([v['upper']*100 for v in vals])
    b.errorbar(x,np.arange(len(rows)),xerr=[x-lo,hi-x],fmt='o',capsize=3,color='#333333',markersize=4);b.axvline(0,color='#aaaaaa',lw=1)
    b.set_yticks(range(len(rows)),[]);b.invert_yaxis();b.set_xlabel('RankCloak minus ordinary\n(percentage points)');b.grid(axis='x',alpha=.2)
    for ax in [a,b]:ax.spines[['top','right']].set_visible(False)
    fig.legend(*a.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,1.0),ncol=2,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,0,1,.89));fig.savefig(dest/'contextual_acceptability.pdf',metadata={'CreationDate':None});fig.savefig(dest/'contextual_acceptability.png',dpi=200);plt.close(fig)
    shutil.copyfile(dest/'contextual_acceptability.pdf',paper/'figures/contextual_acceptability.pdf')
    table=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{lrrrrrr}\toprule',r'Source & Pairs & RankCloak & Ordinary & Difference & \multicolumn{2}{c}{Disagreement}\\',r' & & \% [95\% CI] & \% [95\% CI] & pp [95\% CI] & Enc.\% & Ord.\%\\\midrule']
    for r in rows:
        table.append(name(r['value'])+' & '+str(r['complete_pairs'])+' & '+stack(r['encoded'])+' & '+stack(r['ordinary'])+' & '+stack(r['difference'])+' & '+f"{r['encoded_disagreement']['estimate']*100:.1f} & {r['ordinary_disagreement']['estimate']*100:.1f}"+r'\\')
    table += [r'\bottomrule\end{tabular}',r'\caption{Intact-message contextual acceptability. Acceptance is disruption level 0 or 1. The equally weighted two-judge panel is averaged within trial and payload. The selected sample contains 48 payload clusters, with 288 trials overall and 96 per generator. Pairs counts complete panel-paired messages. Paired percentile intervals use 2,000 class-stratified payload draws and condition on the fixed greedy controls. The Qwen row has 47 observed payload clusters. Disagreement uses the same hierarchy on available pairs of judges within each arm, with 562 encoded and 551 ordinary messages overall. These are automated ratings, not human reading outcomes.}\label{tab:v4-contextual}',r'\end{table}']
    text='\n'.join(table)+'\n';(dest/'primary_table.tex').write_text(text);(paper/'supplementary_tables/v4_contextual.tex').write_text(text)
    supplementary=[r'\section*{Supplementary Note S20. Contextual acceptability rubric and complete results}\label{si:v4-contextual}',
        'The replacement study evaluates intact delivered text. It was designed after the earlier fragment analyses and frozen before its own final judging. It reuses historical encoded messages, not newly generated RankCloak covers. None of the 48 selected payloads overlaps the former 92-payload scored cohort. This does not imply that the underlying original corpus was unseen during study development.',
        r'\subsection*{Complete judging rubric}',escape(RUBRIC),escape(SCALES),
        r'\subsection*{Calibration and model assignment}',
        'Llama-generated messages are evaluated by Mistral and Qwen, Mistral messages by Llama and Qwen, and Qwen messages by Llama and Mistral. Each physical judge therefore sees two source families. Judge and generator effects are not fully crossed. The pinned Q4 model revisions and weights are those in Table~\\ref{tab:models} and the replacement configuration. Exact embedded chat templates, their hashes and the llama-cpp-python 0.3.23 backend are recorded. GPU inference uses deterministic decoding and JSON-schema grammar. Plain JSON escaping preserves the complete message without marking its payload boundary.',
        r'\begin{table}[H]\centering\small\begin{tabular}{lrrrr}\toprule',r'Judge & Clear accepted /6 & Minor accepted /6 & Disrupted rejected /6 & Invalid\\\midrule']
    for m,r in cal['models'].items():
        g=r['acceptable_by_construction'];supplementary.append(name(m)+f" & {g['clear']['acceptable']} & {g['minor']['acceptable']} & {6-g['substantial']['acceptable']} & {r['invalid']}"+r'\\')
    supplementary += [r'\bottomrule\end{tabular}\caption{Fixed constructed calibration. Intended labels were declared before inference. Each model met the basic check, without a rubric amendment. These examples are not participant validation.}\label{tab:v4-contextual-calibration}\end{table}',
        'The minor-imperfection examples contain informal phrasing, mild repetition or local grammar errors whose meaning remains clear. The disrupted examples contain explicit contradictions or disconnected intrusions. Llama accepts an explicit contradiction about attending the same meeting time. Mistral accepts simultaneously dry and fully soaked soil, even though its explanation identifies the contradiction. Thus the declared calibration check is imperfect and does not guarantee consistent application of the rubric. Same-prompt shuffled tails are not assumed to be incoherent negatives. Both the intended labels and all raw calibration responses remain available.',
        r'\subsection*{Distributions, physical judges and schedules}',r'\begin{table}[H]\centering\small\begin{tabular}{llrrrrr}\toprule',r'Group & Arm & Level 0 & Level 1 & Level 2 & Level 3 & Connectedness\\\midrule']
    for r in s['summaries']:
        for arm in ['encoded','ordinary']:
            supplementary.append(escape(group_name(r))+' & '+('Encoded' if arm=='encoded' else 'Ordinary')+' & '+' & '.join(f"{r[arm+'_score_'+str(k)]['estimate']*100:.1f}" for k in range(4))+' & '+f"{r[arm+'_connectedness']['estimate']:.2f}"+r'\\')
    supplementary += [r'\bottomrule\end{tabular}\caption{Complete disruption distributions in percent and mean logical connectedness from 1 to 5. Panel and subgroup values use the declared message, trial and payload hierarchy. Physical-judge rows use only the source families assigned to that judge. Logical connectedness is secondary and does not change acceptance.}\label{tab:v4-contextual-distribution}\end{table}',
        r'\begin{table}[H]\centering\small\begin{tabular}{lrrrr}\toprule',r'Group & Message pairs & RankCloak \% [95\% CI] & Ordinary \% [95\% CI] & Difference pp [95\% CI]\\\midrule']
    for r in s['summaries']:
        if r['group'] not in ['judge','schedule']:continue
        supplementary.append(escape(group_name(r))+' & '+str(r['complete_pairs'])+' & '+stack(r['encoded'])+' & '+stack(r['ordinary'])+' & '+stack(r['difference'])+r'\\')
    supplementary += [r'\bottomrule\end{tabular}\caption{Physical-judge and schedule estimates with paired payload-cluster intervals. Pair counts are complete observations, from 384 assigned pairs per judge and 288 per schedule. The panel uses two judgments per item rather than treating judges as independent participants.}\label{tab:v4-contextual-strata}\end{table}',
        r'\subsection*{Stops, lengths, reuse and missingness}',
        f"The {s['encoded_messages']} message pairs create {s['logical_judge_units']} logical judge assignments and {s['unique_judge_requests']} distinct requests. Exact input caching retains their original inferential multiplicity. There are {s['control_reuse']['unique_requests']} distinct matched controls with maximum reuse {s['control_reuse']['maximum_reuse']}. The primary intervals condition on this fixed control set and do not include variation from independently generated replacement controls. There are {s['invalid_unique_responses']} invalid unique final responses, {s['invalid_logical_scores']} invalid logical scores and {s['retries']} retries. Both initial and retried responses are retained. Worst/best assignment of missing scores is provided in the machine-readable analysis; when none is missing these bounds equal the complete estimates.",
        'Complete panel pairs number 537 in 280 trials across all 48 payloads. The generator counts are 182 Llama, 191 Mistral and 164 Qwen pairs. The Qwen comparison contains 88 trials and 47 observed payloads. Nineteen final unique responses are truncated and one is malformed JSON. Encoded invalid logical slots are 2 for Llama as judge, 7 for Mistral and 9 for Qwen. Two distinct ordinary requests judged by Mistral account for 25 invalid mapped slots. All other ordinary judgments are valid. The 39 incomplete pairs have encoded mean length 128.4 tokens, compared with 50.5 among complete pairs. Availability is therefore not assumed random.',
        'Assigning missing judgments to either binary extreme gives all-slot encoded acceptance from 88.9 to 90.5 percent, ordinary acceptance from 94.8 to 97.0 percent and a paired difference from -8.1 to -4.3 percentage points. These are sensitivity bounds, not confidence intervals. The main estimates and intervals condition on complete paired availability. Binary disagreement uses the 562 encoded and 551 ordinary messages with both within-arm judges valid. Unweighted disagreement counts are 83 and 35, whereas the reported proportions use the declared trial and payload hierarchy.',
        'New controls use a 2,048-token context allocation, while the historical covers used 4,096. A separately declared technical check regenerated all 54 frozen controls at 4,096 after judging. All token IDs, text, stopping reasons and filter masks agreed. It changed no study output and establishes agreement for these controls only. Both generation routes use serial batch and microbatch size one, the pinned backend and CUDA execution. Judge batch and microbatch sizes are 128 with a 2,048-token context. No message was truncated for judging.',
        r'\begin{table}[H]\centering\small\begin{tabular}{llrrrrr}\toprule',r'Source & Arm & Messages & Mean tokens & Median & Range & Cap stops\\\midrule']
    for r in s['lengths']:
        supplementary.append(name(r['generator'])+' & '+('Encoded' if r['arm']=='encoded' else 'Ordinary')+f" & {r['messages']} & {r['mean_tokens']:.1f} & {r['median_tokens']:.1f} & {r['min_tokens']}--{r['max_tokens']} & {r['cap_stops']}"+r'\\')
    supplementary += [r'\bottomrule\end{tabular}\caption{Realized lengths under a shared assigned 264-token budget. Counts are original generator tokens. Controls are not truncated to match encoded length. Cap stops are retained rather than regenerated.}\label{tab:v4-contextual-lengths}\end{table}',
        r'\subsection*{Full-message examples}\label{si:v4-contextual-examples}',
        'Examples follow the predeclared identity-hash rule within jointly acceptable, judge-disagreement and jointly disrupted encoded cases. These labels denote model scores rather than reader judgments. The text below is complete and unedited. No boundary was marked for judging.']
    for r in read_json(OUT/'analysis/examples.json'):
        supplementary += [r'\paragraph{'+escape(r['category'].capitalize())+'}',r'\noindent\textbf{Source.} '+r'\code{'+r['message_id']+r'}. '+name(r['generator'])+'. '+escape(r['payload_name'])+'.',r'\noindent\textbf{Authentic prompt.} '+escape(r['prompt']['prompt_text']),
            r'\begin{Verbatim}[breaklines,breakanywhere,breaksymbolleft={},breaksymbolright={}]'+'\n'+r['text']+'\n'+r'\end{Verbatim}']
        for j in r['judgments']:
            text=name(j['judge_model_id'])+f" gives disruption {j['disruption']} and connectedness {j['connectedness']}."
            if j.get('quote'):
                text+=(' Literal quoted passage: ' if j.get('quote_status')=='literal_match' else ' Reported passage, not a literal source quotation: ')+j['quote']
            supplementary.append(escape(text))
    (dest/'supplement.tex').write_text('\n\n'.join(supplementary)+'\n')
    (paper/'v4_contextual_supplement.tex').write_text('\n\n'.join(supplementary)+'\n')
    atomic_json(dest/'render_inputs.json',{'summary':'results/revision_v4/coherence_replacement/analysis/summary.json','primary_figure':'figures/contextual_acceptability.pdf','primary_table':'supplementary_tables/v4_contextual.tex'})


if __name__=='__main__':main()
