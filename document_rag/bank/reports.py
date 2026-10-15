"""Self-contained printable report. Escape all document/model content."""
from html import escape


def pdf_report(answer):
    """Paginated, searchable Unicode PDF; all untrusted markup is escaped."""
    import io
    import os
    from pathlib import Path
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    candidates = [os.environ.get('BANK_REPORT_FONT', ''),
        '/System/Library/Fonts/Supplemental/Arial.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    font_path = next((p for p in candidates if p and Path(p).is_file()), None)
    if not font_path:
        raise ValueError('Укажите BANK_REPORT_FONT: TTF-шрифт с поддержкой кириллицы.')
    if 'BankUnicode' not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont('BankUnicode', font_path))
    normal = ParagraphStyle('body', fontName='BankUnicode', fontSize=10, leading=15,
                            spaceAfter=8, splitLongWords=True, alignment=TA_LEFT)
    title = ParagraphStyle('title', parent=normal, fontSize=18, leading=24, spaceAfter=16)
    heading = ParagraphStyle('heading', parent=normal, fontSize=13, leading=18, spaceBefore=14, keepWithNext=True)
    small = ParagraphStyle('small', parent=normal, fontSize=8, leading=11, textColor=colors.HexColor('#526174'))
    story = []
    def add(value, style=normal):
        # Escape before adding line breaks: source text cannot create PDF tags/links.
        story.append(Paragraph(escape(str(value)).replace('\n', '<br/>'), style))
    add('Ответ по документам внутреннего аудита', title)
    add(answer.get('question') or answer.get('search_question', ''), heading)
    add(f"Проверка: {answer['verification_status']} · Охват: {answer['coverage_status']} · Версия корпуса: {answer['index_version']}", small)
    add('Дата проверки: ' + str(answer.get('as_of') or 'не задана'), small)
    add(answer['answer_text'])
    add('Проверка различий', heading)
    for finding in answer['discrepancies']:
        add(finding['status'] + ': ' + finding['explanation'])
    add('Источники', heading)
    for index, source in enumerate(answer['sources'], 1):
        add(f"[{index}] {source['document_name']} · стр. {source['page_number']} · п. {source['section_path'] or '—'}", heading)
        add('Версия: ' + source['document_version_id'] + ' · Блок: ' + source['block_id'], small)
        add(source['text_exact'])
    add('Ограничения проверки', heading)
    for warning in answer['warnings']:
        add(warning)
    if answer.get('reviews'):
        add('Заключения аудиторов', heading)
        for review in answer['reviews']:
            add(review['verdict'] + ': ' + review['comment'])
    add('Идентификатор ответа: ' + answer['id'], small)
    output = io.BytesIO()
    def footer(canvas, document):
        canvas.setFont('BankUnicode', 8)
        canvas.drawString(42, 26, 'Внутренний аудит · Проверка по загруженным источникам')
        canvas.drawRightString(553, 26, str(document.page))
    SimpleDocTemplate(output, pagesize=(595.28, 841.89), leftMargin=42, rightMargin=42,
                      topMargin=38, bottomMargin=48, title='Ответ по документам').build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()


def html_report(answer):
    title = escape(answer.get('question', 'Ответ по документам'))
    parts = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Аудиторский ответ</title>',
        '<style>body{font:16px/1.5 system-ui;max-width:900px;margin:40px auto;color:#172033}h1{font-size:26px}pre{white-space:pre-wrap;overflow-wrap:anywhere}section{border-top:1px solid #ccc;margin-top:24px;padding-top:16px}small{color:#555}@media print{body{margin:0}section{break-inside:avoid}}</style>',
        '<body><h1>' + title + '</h1>', '<small>Версия корпуса: ' + str(answer['index_version']) +
        ' · Охват: ' + escape(answer['coverage_status']) + '</small>', '<pre>' + escape(answer['answer_text']) + '</pre>']
    parts.append('<section><h2>Проверка различий</h2>')
    for item in answer['discrepancies']:
        parts.append('<p><b>' + escape(item['status']) + '</b>: ' + escape(item['explanation']) + '</p>')
    parts.append('</section><section><h2>Источники</h2>')
    for index, source in enumerate(answer['sources'], 1):
        parts.append('<h3>[' + str(index) + '] ' + escape(source['document_name']) + ' · стр. ' + escape(str(source['page_number'])) + '</h3>')
        parts.append('<small>Пункт: ' + escape(source['section_path']) + ' · block_id: ' + escape(source['block_id']) + '</small>')
        parts.append('<pre>' + escape(source['text_exact']) + '</pre>')
    parts.append('</section><section><h2>Ограничения</h2><ul>')
    parts.extend('<li>' + escape(w) + '</li>' for w in answer['warnings'])
    parts.append('</ul></section></body></html>')
    return '\n'.join(parts)
