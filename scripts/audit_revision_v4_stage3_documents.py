"""Document counts, source dependencies, full PDF text scan and visual contact sheets."""
import hashlib,json,re,subprocess,tempfile,xml.etree.ElementTree as ET
from pathlib import Path
from scripts.prepare_revision_v4_stage3_handoff import flatten

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision_v4/stage3/validation'
DOCS=['paperV4/scientific_reports/main4','paperV4/scientific_reports/supplementary4','paperV4/response/response_to_reviewers_v4','paperV4/cover_letter/cover_letter_v4']


def main():
    from PIL import Image,ImageDraw
    OUT.mkdir(parents=True,exist_ok=True)
    renders=OUT/'page_sheets';renders.mkdir(exist_ok=True)
    details=[]
    title='RankCloak Conceals the Surface Form of Synthetic Cryptographic Artifacts in Language Model Generated Text'
    for base in DOCS:
        path=ROOT/(base+'.pdf')
        raw=subprocess.check_output(['pdftotext','-layout',str(path),'-'],text=True)
        pages=raw.split('\f')
        if not pages[-1].strip():pages.pop()
        bbox=subprocess.check_output(['pdftotext','-bbox',str(path),'-']).decode()
        # Poppler may emit XML-illegal control-code glyph mappings from math fonts.
        # Drop only those codepoints for XML parsing; the full-text scan is retained.
        xml_controls=len(re.findall(r'[\x00-\x08\x0b\x0c\x0e-\x1f]',bbox))
        doc=ET.fromstring(re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]','',bbox)).findall('.//{http://www.w3.org/1999/xhtml}page')
        if len(doc)!=len(pages):raise ValueError('PDF page count disagreement')
        alltext='\n'.join(pages)
        (OUT/(path.stem+'_fulltext.txt')).write_text(alltext)
        if any(len(re.sub(r'\W|\d','',text))<25 for text in pages):raise ValueError('blank or near-blank PDF page '+base)
        normalized=' '.join(alltext.split()).replace('- ','')
        if title not in normalized:raise ValueError('title mismatch '+base)
        if alltext.count('V4 AUTHOR-REVIEW CANDIDATE')!=1:raise ValueError('candidate label count '+base)
        for term in ['INTERNAL PENDING','INTERNAL WORKING','Stage 2 evidence integrated','Direct answer pending','??']:
            if term in alltext:raise ValueError('stale or unresolved PDF text '+term)
        outside=[]
        for n,page in enumerate(doc,1):
            for block in page.findall('.//{http://www.w3.org/1999/xhtml}word'):
                x0,y0,x1,y1=[float(block.attrib[k]) for k in ['xMin','yMin','xMax','yMax']]
                if x0<0 or y0<0 or x1>float(page.attrib['width'])+.2 or y1>float(page.attrib['height'])+.2:outside.append(n)
        if outside:raise ValueError('text outside page '+str(outside))
        sheets=[]
        for start in range(0,len(doc),4):
            canvas=Image.new('RGB',(1840,2480),'#d0d0d0');draw=ImageDraw.Draw(canvas)
            for k in range(4):
                if start+k>=len(doc):continue
                with tempfile.TemporaryDirectory() as tmp:
                    prefix=str(Path(tmp)/'page')
                    subprocess.run(['pdftoppm','-f',str(start+k+1),'-l',str(start+k+1),'-singlefile','-scale-to-x','900','-scale-to-y','-1','-png',str(path),prefix],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
                    im=Image.open(prefix+'.png').convert('RGB')
                x=(k%2)*920+10;y=(k//2)*1240+28;canvas.paste(im,(x,y));draw.text((x,y-20),path.stem+' page '+str(start+k+1),fill='black')
            dest=renders/(path.stem+f'_{start+1:02d}-{min(start+4,len(doc)):02d}.jpg');canvas.save(dest,quality=88)
            sheets.append(dest.relative_to(ROOT).as_posix())
        details.append({'document':base+'.pdf','pages':len(doc),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'nonblank_pages':len(pages),'text_outside_page':0,'stale_markers':0,'poppler_xml_control_glyphs':xml_controls,'review_sheets':sheets})
    folder=ROOT/'paperV4/scientific_reports';expanded=flatten(folder/'main4.tex')
    body=expanded[expanded.index('\\section*{Introduction}'):expanded.index('\\section*{Methods}')]+expanded[expanded.index('\\section*{Results}'):expanded.index('\\section*{Data Availability}')]
    words={}
    for key,text in [('body',body),('complete',expanded)]:
        with tempfile.NamedTemporaryFile(suffix='.tex',mode='w') as f:
            f.write(text);f.flush();log=subprocess.check_output(['texcount','-utf8',f.name],text=True)
        log=re.sub(r'File: /tmp/[^\n]+','File: expanded '+key,log)
        (OUT/('texcount_'+key+'.txt')).write_text(log)
        words[key]={k:int(v) for k,v in re.findall(r'(Words in text|Words in headers|Words outside text \(captions, etc\.\)): (\d+)',log)}
    abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',expanded,re.S)[1]
    figures=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',expanded)
    for name in figures:
        if not (folder/name).is_file():raise ValueError('missing figure '+name)
    citations=set(k for group in re.findall(r'\\cite\{([^}]+)\}',expanded) for k in group.split(','))
    bbl=(folder/'main4.bbl').read_text();compiled=set(re.findall(r'\\bibitem\{([^}]+)\}',bbl))
    if not citations<=compiled:raise ValueError('missing bibliography entry')
    if 'Cai2025EntropyGuidedWatermarking' in compiled:raise ValueError('obsolete preprint actively cited')
    result={'documents':details,'word_counts_texcount':words,'title_words':len(title.split()),'abstract_words_whitespace':len(abstract.split()),
            'main_figures':len(figures),'main_tables':expanded.count('\\begin{table}'),'main_algorithms_reported_separately':expanded.count('\\begin{algorithm}'),
            'main_references':len(compiled),'main_figure_paths':figures,'supplement_figures':flatten(folder/'supplementary4.tex').count('\\begin{figure}'),
            'supplement_tables':sum(flatten(folder/'supplementary4.tex').count('\\begin{'+kind+'}') for kind in ['table','longtable']),
            'body_word_definition':'Introduction, Results, Discussion and limitations. TeXcount text words exclude abstract, Methods, availability, references, headers and captions. Mathematical expressions are counted separately by TeXcount.',
            'visual_inspection_status':'contact sheets generated, review recorded separately','pdf_rebuild_identity':'Packaged bytes checked exactly. Fresh-build PDF timestamps may differ.'}
    (OUT/'document_audit.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='documents'});print([(r['document'],r['pages']) for r in details])


if __name__=='__main__':main()
