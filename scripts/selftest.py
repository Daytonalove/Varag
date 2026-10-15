"""Actual OCR + local semantic model, isolated temporary DB, no GigaChat calls."""
import copy
from pathlib import Path
import tempfile


def run(cfg):
    from PIL import Image, ImageDraw, ImageFont
    from document_rag.engine import Engine
    from document_rag.extraction import extract
    fonts = ['/System/Library/Fonts/Supplemental/Arial.ttf',
             '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
             '/usr/share/fonts/dejavu/DejaVuSans.ttf']
    font_path = next((p for p in fonts if Path(p).is_file()), None)
    if not font_path:
        raise ValueError('Для OCR selftest установите шрифт DejaVu Sans или Arial.')
    with tempfile.TemporaryDirectory(prefix='rag-selftest-') as temporary:
        root = Path(temporary)
        image = Image.new('RGB', (1800, 800), 'white')
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path, 48)
        for y, line in enumerate(['Заказ 7482', 'Доставка занимает 15 рабочих дней.', 'Delivery confirmed.']):
            draw.text((80, 80 + y * 110), line, font=font, fill='black')
        png, pdf = root / 'scan.png', root / 'scan.pdf'
        image.save(png)
        image.save(pdf, resolution=150)
        ocr_cfg = dict(cfg['ocr'], enabled=True)
        for path in (png, pdf):
            parts, warnings = extract(path, ocr_cfg)
            text = ' '.join(text for _, text in parts)
            if not all(word in text for word in ('7482', '15', 'рабочих')):
                raise ValueError('OCR selftest: не распознаны контрольные слова в ' + path.name)
            print('OCR ' + path.name + ': OK')
        local = copy.deepcopy(cfg)
        local['storage']['database'] = str(root / 'test.sqlite3')
        local['retrieval']['mode'] = 'hybrid'
        engine = Engine(local)
        docs = {'delivery.txt': 'Поставка осуществляется в течение пятнадцати рабочих дней после согласования заявки.',
                'cafe.txt': 'Столовая расположена на втором этаже. Обеденный перерыв начинается в полдень.',
                'vacation.txt': 'Ежегодный оплачиваемый отпуск сотрудника составляет двадцать восемь календарных дней.'}
        ids = []
        for name, content in docs.items():
            path = root / name
            path.write_text(content)
            ids.append(engine.ingest(path)['id'])
        query = 'Как долго ждать заказ?'
        hits = engine.search(query, ids)
        if not hits or hits[0]['document'] != 'delivery.txt':
            raise ValueError('Семантический selftest: ожидаемый документ не на первом месте.')
        # Same question has no exact keyword matches, so this tests semantic retrieval.
        engine.cfg['retrieval']['mode'] = 'lexical'
        if engine.search(query, ids):
            raise ValueError('Контрольная фраза неожиданно дала лексическое совпадение.')
        print('Семантический поиск без совпадений ключевых слов: OK')
        print('GigaChat не вызывался; рабочая база не изменялась.')
