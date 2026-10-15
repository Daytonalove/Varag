"""Opt-in synthetic ingestion capacity check. No GigaChat or user documents."""
import argparse
import io
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from document_rag.bank.store import Store
from document_rag.bank.retrieval import Retriever


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text-pages',type=int,default=100)
    parser.add_argument('--ocr-pages',type=int,default=5)
    parser.add_argument('--report',type=Path,default=Path('evaluation/results/capacity.json'))
    args=parser.parse_args()
    if not 0<=args.text_pages<=5000 or not 0<=args.ocr_pages<=500:
        raise ValueError('Expected text-pages <=5000 and ocr-pages <=500.')
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.utils import ImageReader
    from PIL import Image,ImageDraw,ImageFont
    font=next(p for p in [Path('/System/Library/Fonts/Supplemental/Arial.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')] if p.exists())
    pdfmetrics.registerFont(TTFont('Capacity',str(font)))
    raster_font=ImageFont.truetype(str(font),44)
    report=dict(synthetic_only=True,real_ocr=True,external_calls=0,embedding_enabled=False,reranking_enabled=False,
                expected_text_pages=args.text_pages,expected_ocr_pages=args.ocr_pages,documents=[],pages=0,ocr_pages=0)
    started=time.monotonic()
    def save():
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with tempfile.TemporaryDirectory(prefix='bank-capacity-') as temporary:
        store=Store(Path(temporary)/'capacity.sqlite3')
        project=store.create_project('Synthetic capacity','tester')['id']
        for mode,count in [('text',args.text_pages),('ocr',args.ocr_pages)]:
            for start in range(0,count,100):
                batch=min(100,count-start)
                output=io.BytesIO();canvas=Canvas(output)
                for number in range(start,start+batch):
                    lines=[f'Проверка {mode}, страница {number+1}.',
                           '4.2 Лимит кредитного комитета составляет 10 млн руб.',
                           'Мониторинг заёмщика проводится ежемесячно.']
                    if mode=='text':
                        canvas.setFont('Capacity',11)
                        for i,line in enumerate(lines):canvas.drawString(42,760-i*25,line)
                    else:
                        image=Image.new('RGB',(1600,500),'white');draw=ImageDraw.Draw(image)
                        for i,line in enumerate(lines):draw.text((40,50+i*100),line,font=raster_font,fill='black')
                        canvas.drawImage(ImageReader(image),30,520,width=535,height=167)
                    canvas.showPage()
                canvas.save()
                uploaded=store.upload(project,'tester',f'{mode}-{start}.pdf',output.getvalue())
                t=time.monotonic();store.process_one()
                with store.connect() as conn:
                    row=dict(conn.execute('SELECT status,page_count,warnings FROM versions WHERE id=?',(uploaded['version_id'],)).fetchone())
                    ocr=conn.execute("SELECT count(*) FROM pages WHERE version_id=? AND extraction_method='tesseract'",(uploaded['version_id'],)).fetchone()[0]
                    unreadable=conn.execute("SELECT count(*) FROM pages WHERE version_id=? AND (text_exact='' OR quality_flags LIKE '%extraction_failed%')",(uploaded['version_id'],)).fetchone()[0]
                item=dict(name=f'{mode}-{start}.pdf',expected=batch,pages=row['page_count'],ocr_pages=ocr,status=row['status'],
                          unreadable=unreadable,seconds=round(time.monotonic()-t,2))
                report['documents'].append(item);report['pages']+=row['page_count'];report['ocr_pages']+=ocr
                save();print(json.dumps(item),flush=True)
        t=time.monotonic();result=Retriever(store).search(project,'tester','Какой лимит кредитного комитета?',top_k=10)
        report.update(search_seconds=round(time.monotonic()-t,3),retrieved_sources=len(result['sources']),
                      searched_documents=len(result['coverage_manifest']['searched_document_ids']),
                      elapsed_seconds=round(time.monotonic()-started,2),database_bytes=store.database.stat().st_size)
        report['passed']=report['pages']==args.text_pages+args.ocr_pages and report['ocr_pages']==args.ocr_pages and all(d['unreadable']==0 and d['status'].startswith('ready') for d in report['documents']) and bool(result['sources'])
        save()
    print(json.dumps({k:v for k,v in report.items() if k!='documents'}),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
