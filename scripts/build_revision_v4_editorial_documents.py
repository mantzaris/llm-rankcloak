"""Build the current editorial edition without rewriting prose or historical evidence."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/revision_v4/editorial_restructure'
from rankcloak.revision_v4_stage2_common import atomic_json, file_hash, read_json
from scripts.build_revision_v4_stage4_bundle import manuscript_source, dependency_check, DEPENDENCIES


def build(directory, name, bibliography, logs):
    started = time.monotonic()
    commands = [['pdflatex', '-recorder', '-interaction=nonstopmode', '-halt-on-error', name+'.tex']]
    if bibliography:
        commands.append(['bibtex', name])
    commands += [['pdflatex', '-recorder', '-interaction=nonstopmode', '-halt-on-error', name+'.tex']]*2
    env=os.environ.copy()
    for key in ['TEXINPUTS','BIBINPUTS','BSTINPUTS']:env.pop(key,None)
    env['TEXMFHOME']=str(directory/'empty_texmf')
    for i, cmd in enumerate(commands):
        result = subprocess.run(cmd, cwd=directory, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        (logs/f'{name}_pass{i+1}.txt').write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f'{name} failed. See {logs}')
    if 'undefined references' in result.stdout or re.search(r'(Citation|Reference).*undefined',result.stdout):
        raise RuntimeError(name+' has unresolved references')
    info = subprocess.check_output(['pdfinfo',str(directory/(name+'.pdf'))],text=True)
    (logs/f'{name}_pdfinfo.txt').write_text(info)
    return {'document':name, 'pages':int(re.search(r'^Pages:\s+(\d+)',info,re.M)[1]),
        'source_sha256':file_hash(directory/(name+'.tex')), 'pdf_sha256':file_hash(directory/(name+'.pdf')),
        'elapsed_seconds':time.monotonic()-started,
        'warnings':[line for line in result.stdout.splitlines() if any(t in line for t in ['Warning:','Overfull','Underfull'])],
        'unresolved_references':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['scientific','correspondence','submission'],required=True);args=parser.parse_args()
    parent=OUT/'validation/latex';parent.mkdir(parents=True,exist_ok=True)
    previous=sorted(parent.glob(args.phase+'_*'));logs=parent/(args.phase+'_'+str(len(previous)+1).zfill(2));logs.mkdir()
    paper=ROOT/'paperV4/scientific_reports'
    if args.phase=='submission':
        source='% Editorial author-review edition. Generated from current main4 inputs.\n'+manuscript_source(paper)
        (paper/'main4_submission.tex').write_text(source)
        if re.search(r'\\(?:input|includegraphics|bibliography)\{',source):
            raise ValueError('Editable source retains external content')
        with tempfile.TemporaryDirectory(prefix='rankcloak_editorial_submission_') as name:
            clean=Path(name)
            shutil.copyfile(paper/'main4_submission.tex',clean/'main4_submission.tex')
            for dependency in DEPENDENCIES:
                shutil.copyfile(paper/dependency,clean/dependency)
            records=[build(clean,'main4_submission',False,logs)]
            records[0]['verified_input_paths']=dependency_check(clean/'main4_submission.fls',clean)
            records[0]['clean_dependency_build']=True
            records[0]['routine_pdf_retained']=False
    else:
        jobs=[(paper,'supplementary4',True),(paper,'main4',True)] if args.phase=='scientific' else [(ROOT/'paperV4/response','response_to_reviewers_v4',False),(ROOT/'paperV4/cover_letter','cover_letter_v4',False)]
        records=[build(*job,logs) for job in jobs]
    atomic_json(logs/'build_status.json',records)
    latest=parent/'latest_builds.json';mapping=read_json(latest) if latest.exists() else {}
    mapping[args.phase]={'logs':str(logs.relative_to(ROOT)),'records':records};atomic_json(latest,mapping)
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
