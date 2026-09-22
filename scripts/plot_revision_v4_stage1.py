"""Render retained entropy trajectories and a six-case LaTeX table on CPU."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision_v4'


def main():
    failures=list(csv.DictReader((OUT/'source_tables/entropy_six_failures.csv').open()))
    positions=list(csv.DictReader((OUT/'source_tables/entropy_positions.csv').open()))
    fig, axes=plt.subplots(3,2,figsize=(10,9),layout='constrained')
    for ax,s in zip(axes.flat,failures):
        rows=[r for r in positions if r['plan_id']==s['plan_id']]
        x=[int(r['position'])+1 for r in rows]
        ax.plot(x,[float(r['entropy_bits']) for r in rows],color='#697c91',lw=.65,label='Entropy')
        ax.axhline(float(s['threshold_bits']),color='#b24b38',ls=':',lw=1.5,label='Gate threshold')
        twin=ax.twinx()
        twin.plot(x,[int(r['consumed_after'])/int(s['requested_ranks']) for r in rows],color='#087b72',lw=1.5,label='Payload fraction')
        twin.set_ylim(0,1.05); twin.set_ylabel('Consumed / requested ranks',fontsize=8,color='#087b72')
        model='Mistral' if 'mistral' in s['model_id'] else 'Qwen'
        ax.set_title(f"{model}  {s['plan_id'][-8:]}   {s['consumed_ranks']}/{s['requested_ranks']} ranks",fontsize=10)
        ax.set_xlabel('Embedding position'); ax.set_ylabel('Entropy (bits)'); ax.set_xlim(1,max(x))
        ax.grid(alpha=.15)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False)
    folder=OUT/'figures'; folder.mkdir(exist_ok=True)
    fig.savefig(folder/'entropy_failure_traces.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(folder/'entropy_failure_traces.png',dpi=150)
    plt.close(fig)
    table=[r'\begin{table}[H]',r'\centering\small',r'\begin{tabular}{lrrrrr}',r'\toprule',r'Trial suffix & Requested & Consumed & Budget & Longest run & Eligible fraction \\',r'\midrule']
    for s in failures:
        table.append(f"{s['plan_id'][-8:]} & {s['requested_ranks']} & {s['consumed_ranks']} & {s['tokens_used']} & {s['longest_below_length']} & {float(s['eligible_fraction']):.4f} \\\\")
    table += [r'\bottomrule',r'\end{tabular}',r'\caption{All six retained strict-gate completion failures. Longest run counts consecutive positions below threshold. Full identities and exact half-open ranges are in the accompanying CSV.}\label{tab:v4-failures}',r'\end{table}']
    (OUT/'manuscript_tables/entropy_six_failures.tex').write_text('\n'.join(table)+'\n')


if __name__=='__main__': main()
