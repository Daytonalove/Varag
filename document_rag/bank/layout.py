"""Physical word coordinates from Poppler/Tesseract; no invented confidence."""
import csv
import io
import subprocess
import xml.etree.ElementTree as ET


def pdf_layout(path):
    result = subprocess.run(['pdftotext', '-bbox-layout', '-enc', 'UTF-8', str(path), '-'],
                            check=True, capture_output=True, timeout=120)
    root = ET.fromstring(result.stdout)
    pages = []
    for page in root.iter():
        if page.tag.rsplit('}',1)[-1] != 'page':
            continue
        regions, offset, text = [], 0, ''
        width, height = float(page.attrib['width']), float(page.attrib['height'])
        for block in page.iter():
            if block.tag.rsplit('}',1)[-1] != 'block':
                continue
            lines = []
            for line in block:
                words = [w.text or '' for w in line if w.tag.rsplit('}',1)[-1]=='word']
                if words:
                    lines.append(' '.join(words))
            value = '\n'.join(lines)+'\n'
            if not value.strip():
                continue
            box = [float(block.attrib[k]) for k in ('xMin','yMin','xMax','yMax')]
            regions.append(dict(start=offset,end=offset+len(value),box=box, width=width,height=height,
                                kind='footnote' if box[1]>height*.85 and value.lstrip()[:1] in '*¹²³' else 'paragraph'))
            offset += len(value)
            text += value
        words=[dict(text=w.text or '',box=[float(w.attrib[k]) for k in ('xMin','yMin','xMax','yMax')])
               for w in page.iter() if w.tag.rsplit('}',1)[-1]=='word']
        from .tables import measure_tables
        pages.append(dict(text_exact=text,regions=regions,words=words,tables=measure_tables(words)))
    return pages


def ocr_layout(path):
    from PIL import Image
    result = subprocess.run(['tesseract', str(path), 'stdout', '-l', 'rus+eng', 'tsv'],
                            check=True, capture_output=True, timeout=60)
    with Image.open(path) as image:
        width,height = image.size
    groups = {}
    raw_words=[]
    for row in csv.DictReader(io.StringIO(result.stdout.decode('utf-8')), delimiter='\t', quoting=csv.QUOTE_NONE):
        if row.get('level')!='5' or not row.get('text','').strip():
            continue
        raw_words.append(dict(text=row['text'],box=[int(row['left']),int(row['top']),int(row['left'])+int(row['width']),int(row['top'])+int(row['height'])],confidence=float(row['conf'])))
        groups.setdefault((row['block_num'],row['par_num'],row['line_num']),[]).append(row)
    text, regions = '', []
    for words in groups.values():
        value = ' '.join(w['text'] for w in words)+'\n'
        box = [min(int(w['left']) for w in words), min(int(w['top']) for w in words),
               max(int(w['left'])+int(w['width']) for w in words), max(int(w['top'])+int(w['height']) for w in words)]
        regions.append(dict(start=len(text),end=len(text)+len(value),box=box,width=width,height=height,
                            confidence=min(float(w['conf']) for w in words),kind='paragraph'))
        text += value
    from .tables import measure_tables
    return dict(text_exact=text,regions=regions,words=raw_words,tables=measure_tables(raw_words))


def span_boxes(page, start, end):
    regions = [r for r in page.get('regions',[]) if r['start']<end and r['end']>start]
    if not regions:
        return None
    # Coordinates are page-relative, top-left origin. A union could highlight
    # unrelated columns, so preserve individual rectangles instead.
    return dict(rects=[r['box'] for r in regions], width=regions[0]['width'], height=regions[0]['height'],
                origin='top-left', precision='layout_region',
                confidence=min((r['confidence'] for r in regions if 'confidence' in r),default=None))
