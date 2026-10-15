"""Page-level extraction and structural blocks; originals are never rewritten."""
import hashlib
import logging
import json
import os
import re
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Protocol

from document_rag.extraction import ocr_image

PARSER_VERSION = 'bank-structure-2-layout'
SUPPORTED = {'.pdf', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.txt', '.md'}


def extract_isolated(path, on_stage=lambda stage: None, timeout_seconds=900):
    """Bound the whole parser, not just its external OCR command."""
    import signal
    import time
    with tempfile.TemporaryDirectory(prefix='bank-parser-') as temporary:
        result = Path(temporary) / 'result.json'
        process = subprocess.Popen([sys.executable, '-m', 'document_rag.bank.extract_task', str(Path(path).resolve()), str(result)],
                                   cwd=Path(__file__).resolve().parents[2], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
        deadline, last_stage = time.monotonic() + timeout_seconds, None
        try:
            while process.poll() is None:
                if time.monotonic() > deadline:
                    raise ValueError('Превышено время обработки документа.')
                stage_path = result.with_suffix('.stage')
                if stage_path.exists():
                    stage = stage_path.read_text()
                    if (stage in ('parsing', 'ocr') or stage.startswith('progress:')) and stage != last_stage:
                        on_stage(stage)
                        last_stage = stage
                time.sleep(.1)
            if process.returncode or not result.exists():
                raise ValueError('Документ не удалось разобрать. Проверьте формат, шифрование и целостность файла.')
            return json.loads(result.read_text())
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


class OCRProvider(Protocol):
    def recognize(self, path: Path) -> str: ...


class TesseractOCR:
    def recognize(self, path):
        return ocr_image(path, {'enabled': True, 'languages': 'rus+eng', 'page_timeout_seconds': 60})


def extract_pages(path, ocr=None, on_stage=lambda stage: None):
    """Keep a record even when a single page fails. No invented OCR confidence."""
    path = Path(path)
    ocr = ocr or TesseractOCR()
    on_stage('parsing')
    suffix = path.suffix.lower()
    if suffix in {'.txt', '.md'}:
        text = path.read_text(encoding='utf-8')
        return [dict(page_index=0, page_number='1', text_exact=text,
                     extraction_method='text', quality_flags=[] if text.strip() else ['unreadable'])]
    if suffix == '.pdf':
        from pypdf import PdfReader
        logging.getLogger('pypdf').addHandler(logging.NullHandler())
        logging.getLogger('pypdf').propagate = False
        reader = PdfReader(path)
        if reader.is_encrypted or len(reader.pages) > 5000:
            raise ValueError('Нужен незашифрованный PDF, максимум 5000 страниц.')
        pages = []
        layouts = []
        if shutil.which('pdftotext'):
            try:
                from .layout import pdf_layout
                layouts = pdf_layout(path)
            except Exception:
                pass
        for index, page in enumerate(reader.pages):
            text, method, flags, regions = '', 'pdf_text', [], []
            try:
                text = page.extract_text() or ''
                if index < len(layouts) and layouts[index]['text_exact'].strip():
                    text, regions = layouts[index]['text_exact'], layouts[index]['regions']
                usable = sum(c.isalnum() for c in text) >= 25 and text.count('\ufffd') < 3
                if not usable:
                    on_stage('ocr')
                    method, flags = 'tesseract', ['ocr_unverified', 'layout_unverified']
                    if not shutil.which('pdftoppm'):
                        raise ValueError('Poppler недоступен')
                    with tempfile.TemporaryDirectory() as temp:
                        prefix = Path(temp) / 'page'
                        subprocess.run(['pdftoppm', '-f', str(index + 1), '-l', str(index + 1),
                                        '-singlefile', '-scale-to', '2400', '-png', str(path), str(prefix)],
                                       check=True, capture_output=True, timeout=60)
                        if isinstance(ocr,TesseractOCR):
                            from .layout import ocr_layout
                            layout = ocr_layout(prefix.with_suffix('.png'))
                            text,regions = layout['text_exact'],layout['regions']
                        else:
                            text = ocr.recognize(prefix.with_suffix('.png'))
            except Exception:
                # Parser/OCR diagnostics can contain document text. Keep only safe codes.
                flags.append('extraction_failed')
            if not text.strip():
                flags.append('unreadable')
            if re.search(r'таблиц|формул', text, re.I):
                flags.append('layout_unverified')
            if not regions:
                flags.append('coordinates_unavailable')
            # Physical page number; printed pagination is not guessed.
            pages.append(dict(page_index=index, page_number=str(index + 1), text_exact=text,
                              extraction_method=method, quality_flags=sorted(set(flags)), regions=regions))
            on_stage('progress:'+str(index+1)+'/'+str(len(reader.pages)))
        return pages
    from PIL import Image, ImageSequence
    pages = []
    on_stage('ocr')
    with Image.open(path) as original:
        for index, frame in enumerate(ImageSequence.Iterator(original)):
            if index >= 500:
                raise ValueError('Максимум 500 кадров изображения.')
            text, flags, regions = '', ['ocr_unverified', 'layout_unverified'], []
            try:
                with tempfile.TemporaryDirectory() as temp:
                    image = Path(temp) / 'page.png'
                    frame.convert('RGB').save(image)
                    if isinstance(ocr,TesseractOCR):
                        from .layout import ocr_layout
                        layout = ocr_layout(image)
                        text,regions = layout['text_exact'],layout['regions']
                    else:
                        text = ocr.recognize(image)
            except Exception:
                flags.append('extraction_failed')
            if not text.strip():
                flags.append('unreadable')
            pages.append(dict(page_index=index, page_number=str(index + 1), text_exact=text,
                              extraction_method='tesseract', quality_flags=flags,regions=regions))
            on_stage('progress:'+str(index+1)+'/'+str(original.n_frames))
    return pages


def structure(pages, version_id):
    """Split on paragraphs/numbered clauses, retain exact substrings and hierarchy."""
    result, parents, section = [], {}, ''
    for page in pages:
        offset = 0
        # Delimiters are retained in source text of each block, so no extracted line is lost.
        spans = re.split(r'(?<=\n)(?=\s*(?:\d+(?:\.\d+)*[.)]?\s|Приложение\s))|(?<=\n)(?=\n)',
                         page['text_exact'])
        for text in spans:
            if not text:
                continue
            from .layout import span_boxes
            bbox = span_boxes(page,offset,offset+len(text))
            offset += len(text)
            clause = re.match(r'\s*(\d+(?:\.\d+)*)(?:[.)])?\s+\S', text)
            appendix = re.match(r'\s*(Приложение\s+\S+)', text, re.I)
            if appendix:
                section, parents = appendix[1], {}
            elif clause:
                section = clause[1]
            block_id = hashlib.sha256(f'{version_id}:{len(result)}'.encode()).hexdigest()[:32]
            parent = parents.get(section.rsplit('.', 1)[0]) if '.' in section else None
            if not clause and section in parents:
                parent = parents[section]
            result.append(dict(id=block_id, version_id=version_id, ordinal=len(result),
                               page_index=page['page_index'], page_number=page['page_number'],
                               section_path=section, parent_block_id=parent,
                               block_type='clause' if clause else ('appendix' if appendix else 'paragraph'),
                               text_exact=text, text_normalized=' '.join(text.lower().replace('ё', 'е').split()),
                               bbox=bbox, extraction_method=page['extraction_method'],
                               quality_flags=page['quality_flags'],
                               content_hash=hashlib.sha256(text.encode()).hexdigest(), parser_version=PARSER_VERSION))
            if clause or appendix:
                parents[section] = block_id
    return result
