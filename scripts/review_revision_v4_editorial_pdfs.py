"""Record all-page text/bounds checks and render changed or reflowed PDF pages."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision_v4/editorial_restructure'
from rankcloak.revision_v4_stage2_common import read_json,atomic_json,file_hash

DOCS=['paperV4/scientific_reports/main4.pdf','paperV4/scientific_reports/supplementary4.pdf',
      'paperV4/response/response_to_reviewers_v4.pdf','paperV4/cover_letter/cover_letter_v4.pdf']


def inspect(path):
    text=subprocess.check_output(['pdftotext','-layout',str(path),'-'],text=True)
    xml=subprocess.check_output(['pdftotext','-bbox',str(path),'-'],text=True)
    xml=re.sub('[\x00-\x08\x0b\x0c\x0e-\x1f]','',xml)
    root=ET.fromstring(xml);pages=root.findall('.//{*}page');fingerprints=[];issues=[]
    for number,page in enumerate(pages,1):
        words=page.findall('.//{*}word');width=float(page.attrib['width']);height=float(page.attrib['height'])
        if len(words)<10:issues.append({'page':number,'type':'sparse_or_blank','words':len(words)})
        for word in words:
            a=word.attrib
            if float(a['xMin'])<0 or float(a['yMin'])<0 or float(a['xMax'])>width+1 or float(a['yMax'])>height+1:
                issues.append({'page':number,'type':'outside_page','word':word.text,'bounds':a})
        content=[(w.text,*(round(float(w.attrib[k]),1) for k in ['xMin','yMin','xMax','yMax'])) for w in words]
        fingerprints.append(hashlib.sha256(json.dumps(content,ensure_ascii=False).encode()).hexdigest())
    for marker in ['INTERNAL PENDING WORK','Direct answer pending.','TODO','??']:
        if marker in text:issues.append({'type':'stale_or_unresolved_marker','marker':marker})
    return text,fingerprints,issues


def render(job):
    pdf,page,dest=job
    subprocess.run(['pdftoppm','-f',str(page),'-l',str(page),'-r','100','-png','-singlefile',str(pdf),str(dest)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    return dest.with_suffix('.png')


def main():
    dest=OUT/'validation/rendered';dest.mkdir(parents=True,exist_ok=True)
    baseline=read_json(OUT/'baseline.json')['head'];records=[]
    for relative in DOCS:
        path=ROOT/relative;stem=path.stem
        old=dest/(stem+'_baseline.pdf');old.write_bytes(subprocess.check_output(['git','show',baseline+':'+relative],cwd=ROOT))
        text,new_pages,issues=inspect(path);_,old_pages,_=inspect(old)
        changed=[i for i,h in enumerate(new_pages,1) if i>len(old_pages) or old_pages[i-1]!=h]
        (OUT/'validation'/(stem+'_text.txt')).write_text(text)
        jobs=[(path,i,dest/(stem+'_page_'+str(i).zfill(2))) for i in changed]
        with ThreadPoolExecutor(max_workers=3) as pool:images=list(pool.map(render,jobs))
        sheets=[]
        for start in range(0,len(images),4):
            tiles=images[start:start+4];loaded=[Image.open(p).convert('RGB') for p in tiles]
            w=max(im.width for im in loaded);h=max(im.height for im in loaded)+30
            sheet=Image.new('RGB',(w*2,h*2),'#dddddd');draw=ImageDraw.Draw(sheet)
            for j,(p,im) in enumerate(zip(tiles,loaded)):
                x=(j%2)*w;y=(j//2)*h;draw.text((x+12,y+8),p.stem,fill='black');sheet.paste(im,(x,y+30))
            out=dest/(stem+'_contact_'+str(start//4+1).zfill(2)+'.jpg');sheet.save(out,quality=63)
            sheets.append(str(out.relative_to(ROOT)))
        records.append({'document':relative,'sha256':file_hash(path),'pages':len(new_pages),'baseline_pages':len(old_pages),
            'text_or_geometry_changed_pages':changed,'all_page_automatic_issues':issues,'contact_sheets':sheets,
            'visual_inspection':'pending actual inspection'})
    atomic_json(OUT/'validation/pdf_review.json',{'baseline_commit':baseline,'documents':records,
        'rule':'Every current page is checked for text and page bounds. Contact sheets cover every page whose text or word geometry differs from the baseline, plus new pages. Visual inspection is recorded separately after viewing.'})
    print(json.dumps([{k:v for k,v in r.items() if k!='contact_sheets'} for r in records],indent=2))


if __name__=='__main__':main()
