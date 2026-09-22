"""Create a conservative V4 document scaffold and exact review ledger.

Existing differing target files are never overwritten. This is an initial
scaffolder, not a manuscript regeneration command after author editing.
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITLE = 'RankCloak Conceals the Surface Form of Synthetic Cryptographic Artifacts in Language Model Generated Text'


def put(path, content):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_text()!=content:
            raise RuntimeError(f'Existing target requires conservative manual merge: {path}')
        return
    path.write_text(content)


def escape(text):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(mapping.get(c,c) for c in text)


def review_blocks(text):
    starts=[('editor_letter',0),('reviewer2',text.index('Reviewer 2\n\n')),('reviewer1_general',text.index('Reviewer 1\n\n'))]
    starts += [(f'R1.{m.group(1)}',m.start()) for m in re.finditer(r'(?m)^(10|[1-9])- ',text)]
    if len(starts)!=13:
        raise ValueError('Expected editor letter, two reviewer introductions, ten numbered comments')
    return [{'id':name,'text':text[start:starts[i+1][1] if i+1<len(starts) else len(text)]} for i,(name,start) in enumerate(starts)]


def main():
    old=ROOT/'paperV3/scientific_reports'; new=ROOT/'paperV4/scientific_reports'
    for p in sorted(old.rglob('*')):
        if not p.is_file(): continue
        rel=p.relative_to(old)
        if p.suffix not in {'.cls','.sty','.bst','.ldf','.bib'} and rel.parts[0] not in {'figures','supplementary_tables','submission_figures'}:
            continue
        target=new/rel; target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes()!=p.read_bytes():
            raise RuntimeError(f'Existing asset requires merge: {target}')
        if not target.exists(): shutil.copy2(p,target)
    for stem in ['main','supplementary']:
        text=(old/f'{stem}3.tex').read_text()
        text=text.replace(r'\usepackage{microtype}',r'\usepackage{microtype}'+'\n'+r'\usepackage{xcolor}')
        text=text.replace(r'\maketitle',r'''\maketitle
\begin{center}
\fcolorbox{red}{white}{\parbox{0.91\linewidth}{\centering\bfseries\color{red}
V4 INTERNAL WORKING DRAFT\par
Stage 1 scaffold based on V3. New evidence and responses remain pending.\par
Not ready for journal submission.}}
\end{center}''',1)
        text='% V4 Stage 1 working copy. Historical V3 source remains unchanged.\n'+text
        if stem=='main':
            text=text.replace(r'\subsection*{RankCloak segmented multi-cover method}',r'\input{v4_filter_methods.tex}'+'\n\n'+r'\subsection*{RankCloak segmented multi-cover method}',1)
        else:
            text=text.replace(r'\bibliography{references}',r'\input{v4_stage1_entropy.tex}'+'\n\n'+r'\bibliography{references}',1)
        put(new/f'{stem}4.tex',text)
    put(new/'v4_filter_methods.tex',(ROOT/'results/revision_v4/manuscript_tables/filter_methods.tex').read_text())
    put(new/'v4_stage1_entropy.tex',r'''\section*{Supplementary Note S16. V4 Stage 1 diagnostic evidence}\label{si:v4-stage1}
\textbf{Internal working draft.} The initial offline analysis reuses the retained V3 records. All six strict-gate failures remain in the denominator of 120 strict attempts. The ungated and moderate conditions each retain 120 attempts. Twenty-one trajectories include the six failures, their twelve matched ungated and moderate runs, and three strict successes chosen by a fixed identity hash within the failed model, representation, and prompt strata.

The complete tables and pinned-tokenizer text windows are retained under \path{results/revision_v4/source_tables}. The accompanying manifest records every raw input hash and the selection rules. These trajectories describe the capacity accounting and observed local contexts. They do not establish a causal linguistic explanation. Integration of the interpretation, complementary boundary-coherence results, and final reviewer responses remains pending.

\input{supplementary_tables/entropy_six_failures.tex}
\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/v4_entropy_failure_traces.pdf}
\caption{Existing strict-gate failure trajectories. Entropy is measured before each token and compared with the frozen model threshold. Cumulative consumed ranks are normalized by the requested count. All six trajectories exhaust their token budgets before completion. This is exploratory evidence from retained records.}\label{fig:v4-entropy}
\end{figure}
''')
    request=ROOT/'paperV4/response/requests.txt'; original=request.read_text()
    blocks=review_blocks(original)
    put(ROOT/'paperV4/response/review_comments.json',json.dumps({'source':'paperV4/response/requests.txt','source_sha256':hashlib.sha256(request.read_bytes()).hexdigest(),'blocks':blocks},indent=2)+'\n')
    prior=(ROOT/'paperV3/response/response_to_reviewers_v3.tex').read_text()
    preamble=prior.split(r'\title{',1)[0]
    body=preamble+r'''\title{V4 response working draft}
\author{Alexander V. Mantzaris}
\date{}
\begin{document}
\maketitle
\begin{center}\color{red}\bfseries INTERNAL PENDING WORK\end{center}
This response is a Stage 1 scaffold for submission 0170c7b2-1065-4344-a5d5-b839b51ac22c.
The manuscript title is \emph{'''+TITLE+r'''}.
The original review letter is preserved below. Pending responses are internal drafting instructions and must be resolved before submission.

'''
    labels={'editor_letter':'Editor letter','reviewer2':'Reviewer 2 acknowledgment','reviewer1_general':'Reviewer 1 general assessment and wording instruction'}
    for block in blocks:
        identity=block['id']; label=identity.replace('.','-')
        body+=r'\section*{'+labels.get(identity,'Reviewer 1 comment '+identity.split('.')[-1])+r'}\label{resp:'+label+'}\n'
        body+='% BEGIN QUOTE '+identity+'\n'+r'\begin{reviewcomment}'+'\n'+escape(block['text'].rstrip())+'\n'+r'\end{reviewcomment}'+'\n'+'% END QUOTE '+identity+'\n\n'
        topics=['editor requirements','code deposit and archive coverage','editor technical assessment'] if identity=='editor_letter' else [identity]
        if identity=='reviewer1_general': topics=['general assessment','verbatim wording preservation']
        for topic in topics:
            body+=r'\textbf{Internal pending work for '+escape(topic)+r'.}'+'\n\n'
            body+=r'\textbf{Direct answer pending.} Draft a direct answer after the relevant evidence and limitations have been checked.'+'\n\n'
            body+=r'\textbf{Evidence and change pending.} Link retained or newly completed evidence and describe the actual change. Do not claim unperformed work is complete.'+'\n\n'
            body+=r'\textbf{Location pending.} Insert final manuscript, supplement, and evidence locations after integration and compilation.'+'\n\n'
    body+=r'\end{document}'+'\n'
    put(ROOT/'paperV4/response/response_to_reviewers_v4.tex',body)
    cover=(ROOT/'paperV3/cover_letter/cover_letter_v3.tex').read_text().split(r'\begin{document}',1)[0]
    cover=cover.replace(r'\usepackage{microtype}',r'\usepackage{microtype}'+'\n'+r'\usepackage{xcolor}')
    cover+=r'''\begin{document}
\begin{letter}{Scientific Reports}
\opening{Dear Dr.\ Yang,}
\begin{center}\color{red}\bfseries V4 INTERNAL WORKING DRAFT\end{center}
This is a working cover-letter scaffold for \emph{'''+TITLE+r'''} (submission 0170c7b2-1065-4344-a5d5-b839b51ac22c).

Stage 1 prepares the V4 documents, audits existing evidence, and analyzes retained entropy traces. The manuscript and response are not ready for submission. Boundary-coherence evaluation, targeted diagnostic decisions, manuscript integration, and archive version coverage remain pending.

\textbf{Internal pending work.} Replace this paragraph with an accurate account of the completed V4 changes after the response matrix is resolved. Address the editor and both current reviewers. State adverse findings and remaining limitations directly. Confirm the final deposited version and DOI coverage before making an availability claim.

RankCloak concerns surface-form concealment and recovery of synthetic cryptographic artifacts under a shared configuration. It does not establish cryptographic secrecy, statistical indistinguishability against the evaluated informed attacker, or reliable transmission under arbitrary text transformations.

\closing{Sincerely,}
Alexander V. Mantzaris\\
School of Data, Mathematical, and Statistical Sciences\\
University of Central Florida\\
alexander.mantzaris@ucf.edu
\end{letter}
\end{document}
'''
    put(ROOT/'paperV4/cover_letter/cover_letter_v4.tex',cover)
    print('V4 scaffold created with 13 exact review blocks and pending responses.')


if __name__=='__main__': main()
