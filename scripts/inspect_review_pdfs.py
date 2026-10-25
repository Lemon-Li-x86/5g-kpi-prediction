"""Render final report PDFs for visual inspection and record page/text checks."""
from pathlib import Path
import json
import re
import hashlib
import pypdfium2 as pdfium
from pypdf import PdfReader
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'deliverables/review_20261025'
OUT=ROOT/'resources/final_audit_20261025/document_previews'
OUT.mkdir(parents=True,exist_ok=True)
summary=[]
for n,p in enumerate(sorted((FOLDER/'pdf').glob('*.pdf'))):
    doc=pdfium.PdfDocument(str(p));reader=PdfReader(p)
    rows=[];thumbs=[]
    for i,page in enumerate(doc):
        im=page.render(scale=1.3).to_pil().convert('RGB');dest=OUT/f'doc{n:02d}-page{i+1:02d}.png';im.save(dest)
        text=reader.pages[i].extract_text() or ''
        rows.append(dict(page=i+1,text_characters=len(text.strip()),path=str(dest.relative_to(ROOT))))
        thumb=im.copy();thumb.thumbnail((380,535));ImageDraw.Draw(thumb).text((4,4),f'Doc{n} page{i+1}',fill='red');thumbs.append(thumb)
    for start in range(0,len(thumbs),6):
        group=thumbs[start:start+6];sheet=Image.new('RGB',(1140,1070),'#ddd')
        for k,im in enumerate(group):sheet.paste(im,((k%3)*380,(k//3)*535))
        sheet.save(OUT/f'contact-doc{n:02d}-{start//6}.png')
    full=''.join((x.extract_text() or '') for x in reader.pages)
    summary.append(dict(file=str(p.relative_to(ROOT)),pages=len(doc),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                        page_checks=rows,mentions_latest_tests='57项' in re.sub(r'\s+','',full),
                        blank_page_numbers=[r['page'] for r in rows if r['text_characters']<30],
                        renderer='Installed Microsoft Word ExportAsFixedFormat -> pypdfium2 PNG; bundled render_docx failed: LibreOffice absent',
                        visual_inspection='pending'))
(ROOT/'reports/final_audit_20261025/document_qa.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print([(r['file'],r['pages'],r['blank_page_numbers'],r['mentions_latest_tests']) for r in summary])
