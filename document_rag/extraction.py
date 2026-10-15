import logging
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile

SUPPORTED = {'.pdf', '.docx', '.txt', '.md', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}
MAX_BYTES = 25 * 1024 * 1024


def chunks(text, size=700):
    text = re.sub(r'[ \t]+', ' ', text).strip()
    while text:
        if len(text) <= size:
            yield text
            break
        cut = text.rfind(' ', size // 2, size)
        if cut < 0:
            cut = size
        yield text[:cut].strip()
        text = text[max(1, cut - min(100, size // 5)):].strip()


def ocr_image(path, cfg):
    if not cfg.get('enabled') or not shutil.which('tesseract'):
        raise ValueError('Для OCR установите Tesseract с языками rus и eng; см. README.md.')
    try:
        result = subprocess.run(['tesseract', str(path), 'stdout', '-l', cfg['languages']],
            capture_output=True, check=True, timeout=cfg.get('page_timeout_seconds', 90))
        return result.stdout.decode('utf-8')
    except (subprocess.SubprocessError, UnicodeError):
        raise ValueError('OCR не выполнен: проверьте изображение и языки tesseract --list-langs.') from None


def extract(path, cfg):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('Максимум 25 MiB на файл.')
    suffix = path.suffix.lower()
    warnings, parts = [], []
    if suffix in ('.txt', '.md'):
        return [('текст', path.read_text(encoding='utf-8'))], []
    if suffix == '.docx':
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo('word/document.xml')
            if info.file_size > 30 * 1024 * 1024:
                raise ValueError('XML документа слишком велик.')
            root = ET.fromstring(archive.read(info))
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        body = root.find('w:body', ns)
        if body is None:
            raise ValueError('DOCX не содержит основного текста.')
        for number, block in enumerate(body, 1):
            text = ' '.join(node.text or '' for node in block.findall('.//w:t', ns))
            if text.strip():
                parts.append((f'блок {number}', text))
        return parts, ['DOCX: номера блоков, не страниц. Рисунки, сноски и сложные таблицы требуют проверки.']
    if suffix == '.pdf':
        from pypdf import PdfReader
        logger = logging.getLogger('pypdf')
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        reader = PdfReader(path)
        if reader.is_encrypted or len(reader.pages) > 200:
            raise ValueError('Нужен незашифрованный PDF не более 200 страниц.')
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ''
            if cfg.get('force_pdf_ocr') or len(text.strip()) < 25:
                if cfg.get('enabled'):
                    if not shutil.which('pdftoppm'):
                        raise ValueError('Для PDF OCR установите Poppler: brew install poppler.')
                    with tempfile.TemporaryDirectory() as temp:
                        prefix = Path(temp) / 'page'
                        try:
                            subprocess.run(['pdftoppm', '-f', str(number), '-l', str(number), '-singlefile',
                                '-r', '150', '-png', str(path), str(prefix)], check=True, capture_output=True,
                                timeout=cfg.get('page_timeout_seconds', 90))
                        except subprocess.SubprocessError:
                            raise ValueError(f'Ошибка растрирования страницы {number}.') from None
                        text = ocr_image(prefix.with_suffix('.png'), cfg)
                    warnings.append(f'Страница {number}: OCR, проверьте числа и таблицы по оригиналу.')
                else:
                    warnings.append(f'Страница {number}: мало текста, OCR выключен.')
            if text.strip():
                parts.append((f'страница {number}', text))
            else:
                warnings.append(f'Страница {number}: текст не извлечён.')
        if any(re.search(r'таблиц[аы]|формул', text, re.I) for _, text in parts):
            warnings.append('Таблицы и формулы извлечены как текст. Порядок колонок, обозначения и формулы проверьте по оригиналу.')
        return parts, warnings
    if suffix in SUPPORTED:
        return [('OCR изображения', ocr_image(path, cfg))], ['Точность OCR проверьте по оригиналу.']
    raise ValueError('Формат не поддерживается. DOC преобразуйте в DOCX или PDF.')
