"""Compile the four V4 working documents without changing their sources."""
import json
import re
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    logs=ROOT/'results/revision_v4/validation/latex'; logs.mkdir(parents=True,exist_ok=True)
    jobs=[('paperV4/scientific_reports','main4',True),('paperV4/scientific_reports','supplementary4',True),('paperV4/response','response_to_reviewers_v4',False),('paperV4/cover_letter','cover_letter_v4',False)]
    records=[]
    for directory,name,bibliography in jobs:
        started=time.monotonic(); commands=[['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex']]
        if bibliography: commands.append(['bibtex',name])
        commands += [['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex']]*2
        for i,cmd in enumerate(commands):
            result=subprocess.run(cmd,cwd=ROOT/directory,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
            (logs/f'{name}_pass{i+1}.txt').write_text(result.stdout)
            if result.returncode: raise RuntimeError(f'{name} build failed, see {logs}/{name}_pass{i+1}.txt')
        if 'undefined references' in result.stdout or re.search(r'(Citation|Reference).*undefined', result.stdout):
            raise RuntimeError(f'{name} has unresolved references after final pass')
        warnings=[line for line in result.stdout.splitlines() if any(term in line for term in ['Warning:', 'Overfull', 'Underfull'])]
        info=subprocess.check_output(['pdfinfo',str(ROOT/directory/(name+'.pdf'))],text=True)
        (logs/f'{name}_pdfinfo.txt').write_text(info)
        records.append({'document':directory+'/'+name+'.pdf','status':'compiled','unresolved_references':False,'final_layout_warnings':warnings,'elapsed_seconds':time.monotonic()-started})
    (logs/'build_status.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
