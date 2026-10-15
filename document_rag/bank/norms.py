"""Conservative, source-bound normalization and comparison of explicit norms.

No legal precedence is inferred from publication order. Unknown units, dates or
subjects remain uncertain; explicit changes need a matching target and date.
"""
from datetime import date, datetime
from decimal import Decimal
import re

VERSION = 'norms-2'
MONEY = re.compile(r'(?<![\w.])(\d+(?:[ \u00a0]\d{3})*(?:[.,]\d+)?)\s*(млн|миллион\w*|тыс\.?|тысяч\w*|млрд)?\s*(руб\w*\.?|₽|RUB|USD|доллар\w*|EUR|евро)', re.I)
DURATION = re.compile(r'(?<!\w)(\d+(?:[.,]\d+)?)\s*(рабоч\w*|календарн\w*)?\s*(дн\w*|день|дней|час\w*|месяц\w*|лет|год\w*)', re.I)
PERCENT = re.compile(r'(?<!\w)(\d+(?:[.,]\d+)?)\s*(?:%|процент\w*)', re.I)
DATE = re.compile(r'\b(?:\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})\b')
WRITTEN_DATE = re.compile(r'\b(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+(\d{4})(?:\s+года|\s+г\.)?',re.I)
CHANGE = re.compile(r'измен\w*|замен\w*|отмен\w*|увелич\w*|снижа\w*|излож\w*', re.I)
SUBJECT = re.compile(r'кредитн\w*\s+комитет\w*|валютн\w*\s+комитет\w*|правлени\w*|совет\w*\s+директор\w*|служб\w*\s+комплаенс\w*', re.I)
NORMATIVE = re.compile(r'лимит|срок|полномоч|предел|обязан|запрещ|разреш|не более|не менее|свыше|ставк|процент|хранен|требует|должен|должна', re.I)


def iso_date(value):
    return (datetime.strptime(value, '%d.%m.%Y').date() if '.' in value else date.fromisoformat(value)).isoformat()


def date_mentions(text):
    for match in DATE.finditer(text):
        try:
            yield match.span(),match[0],iso_date(match[0])
        except ValueError:
            continue
    months='января февраля марта апреля мая июня июля августа сентября октября ноября декабря'.split()
    for match in WRITTEN_DATE.finditer(text):
        try:
            yield match.span(),match[0],date(int(match[3]),months.index(match[2].lower())+1,int(match[1])).isoformat()
        except ValueError:
            continue


def operator(text, start):
    prefix = text[max(0, start - 50):start].lower()
    for pattern, op in [(r'не более|не превыша\w*|не позднее|до|максимум', 'lte'),
                        (r'не менее|не раньше|от|минимум', 'gte'),
                        (r'свыше|более|позднее', 'gt'), (r'менее|раньше', 'lt'),
                        (r'равен|равна|составляет|ровно|—|:', 'eq')]:
        if re.search(r'(?:' + pattern + r')\s*$', prefix):
            return op
    return 'unknown'


def amounts(text):
    values = []
    for match in MONEY.finditer(text):
        number, multiplier, currency = match.groups()
        factor = 1
        if multiplier:
            factor = 10**9 if multiplier.lower().startswith('млрд') else (10**6 if multiplier.lower().startswith(('млн', 'миллион')) else 1000)
        value = Decimal(number.replace(' ', '').replace('\u00a0', '').replace(',', '.')) * factor
        currency = 'RUB' if currency.lower().startswith(('руб', 'rub', '₽')) else ('USD' if currency.lower().startswith(('usd', 'доллар')) else 'EUR')
        values.append(dict(value=str(value.normalize()), currency=currency, operator=operator(text, match.start()), exact=match[0]))
    return values


def periods(text):
    """Explicit dates only; a missing bound is unknown, not an inferred repeal."""
    result = {'from': None, 'until': None}
    for span,_,value in date_mentions(text):
        prefix = text[max(0, span[0] - 30):span[0]].lower()
        if re.search(r'(?:\bс|начиная с|вступает в силу)\s*$', prefix):
            result['from'] = result['from'] or value
        elif re.search(r'(?:\bпо|действует до|действовал до)\s*$', prefix):
            result['until'] = value
    month = re.search(r'\bв\s+(январ\w*|феврал\w*|март\w*|апрел\w*|мае|июн\w*|июл\w*|август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)\s+(20\d{2})', text, re.I)
    if month:
        import calendar
        number = next(i for i, prefix in enumerate(('январ','феврал','март','апрел','ма','июн','июл','август','сентябр','октябр','ноябр','декабр'), 1) if month[1].lower().startswith(prefix))
        year = int(month[2])
        result = {'from': date(year, number, 1).isoformat(), 'until': date(year, number, calendar.monthrange(year, number)[1]).isoformat()}
    return result


def scope(text):
    low = text.lower().replace('ё', 'е')
    products = [label for pattern, label in [(r'ипотеч', 'mortgage'), (r'потребител', 'consumer'), (r'мал\w* бизнес', 'sme')] if re.search(pattern, low)]
    limit_types = [label for pattern, label in [(r'на (?:одну )?(?:сделк|операци)', 'transaction'),
                  (r'на (?:одного )?(?:клиент|заемщик)', 'customer'), (r'на портфел', 'portfolio')] if re.search(pattern, low)]
    subject = SUBJECT.search(low)
    return dict(products=products, limit_types=limit_types, subject=' '.join(w[:6] for w in subject[0].split()) if subject else None, period=periods(text))


def facts(text):
    result = []
    for value in amounts(text):
        result.append(dict(kind='money', value=value['value'], unit=value['currency'], operator=value['operator'], exact=value['exact']))
    money_spans = [m.span() for m in MONEY.finditer(text)]
    for regex, kind in [(DURATION, 'duration'), (PERCENT, 'percent')]:
        for m in regex.finditer(text):
            if kind == 'duration' and m[3].lower().startswith('год') and 1900 <= Decimal(m[1].replace(',', '.')) <= 2200:
                continue  # A calendar year is not a duration.
            if any(a <= m.start() < b for a, b in money_spans):
                continue
            if kind == 'duration':
                unit = 'day' if m[3].lower().startswith(('дн', 'ден')) else ('hour' if m[3].lower().startswith('час') else ('month' if m[3].lower().startswith('месяц') else 'year'))
                unit += ':working' if m[2] and m[2].lower().startswith('рабоч') else (':calendar' if m[2] else ':unspecified')
            else:
                unit = '%'
            result.append(dict(kind=kind, value=str(Decimal(m[1].replace(',', '.')).normalize()), unit=unit,
                               operator=operator(text, m.start()), exact=m[0]))
    return result


def amendment_basis(sources, left, right):
    """Find a third document explicitly changing this clause/value/subject."""
    a, b = facts(left['text_exact']), facts(right['text_exact'])
    if len(a) != 1 or len(b) != 1 or (a[0]['kind'], a[0]['unit']) != (b[0]['kind'], b[0]['unit']):
        return None
    subjects = {scope(s['text_exact'])['subject'] for s in (left, right)} - {None}
    for source in sources:
        if source['block_id'] in (left['block_id'], right['block_id']):
            continue
        identities = [s.get('document_id', s['block_id']) for s in (left, right, source)]
        if len(set(identities)) != 3:
            continue
        text = source['text_exact']
        when = periods(text)['from']
        if not when or not CHANGE.search(text):
            continue
        values = facts(text)
        keys = {(f['kind'], f['unit'], f['value']) for f in values}
        target = {(f['kind'], f['unit'], f['value']) for f in (a[0], b[0])}
        clause = any(s.get('section_path') and re.search(r'(?:пункт\w*|п\.)\s*' + re.escape(s['section_path']) + r'(?![\d.])', text, re.I) for s in (left, right))
        change_subjects = {' '.join(w[:6] for w in m[0].lower().split()) for m in SUBJECT.finditer(text)}
        same_subject = (bool(change_subjects & subjects) or (not change_subjects and clause)) if subjects else clause
        # Both old/new amounts or an explicit target clause and new amount are necessary.
        matches_values = target <= keys or (clause and any((f['kind'], f['unit'], f['value']) in keys for f in (a[0], b[0])))
        if NORMATIVE.search(text) and same_subject and matches_values:
            new_value = values[-1]['value'] if values else None
            return dict(source=source, effective_from=when, new_value=new_value)
    return None


def compare(sources, as_of=None):
    findings = []
    for i, left in enumerate(sources):
        if not NORMATIVE.search(left['text_exact']):
            continue
        la, ls = facts(left['text_exact']), scope(left['text_exact'])
        for right in sources[i + 1:]:
            if not NORMATIVE.search(right['text_exact']):
                continue
            ra, rs = facts(right['text_exact']), scope(right['text_exact'])
            if len(la) != 1 or len(ra) != 1 or la[0]['kind'] != ra[0]['kind']:
                continue
            a, b = la[0], ra[0]
            if (a['value'], a['unit'], a['operator']) == (b['value'], b['unit'], b['operator']):
                continue
            if ls['subject'] and rs['subject'] and ls['subject'] != rs['subject']:
                continue
            words = lambda t: {w[:6] for w in re.findall(r'[а-яa-z]{4,}', t.lower())} - {'состав','лимит','срок','рублей','миллио'}
            if not ls['subject'] and not (words(left['text_exact']) & words(right['text_exact'])):
                continue
            status, explanation = 'potential_unresolved_conflict', 'Найдены разные значения или несовместимые ограничения одного предмета. Основание выбора не установлено.'
            disjoint = any(ls[key] and rs[key] and set(ls[key]).isdisjoint(rs[key]) for key in ('products','limit_types'))
            lp, rp = ls['period'], rs['period']
            disjoint = disjoint or bool(lp['until'] and rp['from'] and lp['until'] < rp['from']) or bool(rp['until'] and lp['from'] and rp['until'] < lp['from'])
            if disjoint or (a['kind'] == 'money' and a['unit'] != b['unit']):
                status, explanation = 'different_scope_no_conflict', 'В источниках явно различаются продукт, вид лимита, валюта или непересекающиеся периоды действия.'
            elif a['unit'] != b['unit']:
                status, explanation = 'needs_review', 'Единицы/виды дней различаются; пересчёт без календаря и дополнительных условий невозможен.'
            ids = [left['block_id'], right['block_id']]
            evidence = [{'block_id': s['block_id'], 'quote': s['text_exact']} for s in (left, right)]
            basis = amendment_basis(sources, left, right)
            applicable = None
            if basis and as_of:
                status = 'explained_by_amendment'
                before = as_of < basis['effective_from']
                explanation = ('Дата проверки предшествует изменению; новая норма ещё не вступила в силу.' if before else 'Прямое изменение указанного положения вступило в силу к дате проверки.')
                explanation += ' Дата изменения: ' + basis['effective_from'] + '.'
                value = next((f for f in (a,b) if (f['value']!=basis['new_value'] if before else f['value']==basis['new_value'])),None)
                if value:
                    applicable = dict(value, as_of=as_of)
                explanation += ' Для даты проверки: ' + value['exact'].rstrip('.') + '.'
                ids.append(basis['source']['block_id'])
                evidence.append({'block_id': basis['source']['block_id'], 'quote': basis['source']['text_exact']})
            elif basis:
                status, explanation = 'needs_review', 'Найдено прямое изменение. Для определения действующей нормы нужна дата проверки.'
            if any(s.get('quality_flags') for s in (left, right)):
                status, explanation = 'needs_review', 'Качество исходного текста требует проверки оригинала до подтверждения чисел и условий.'
            findings.append(dict(source_block_ids=ids, status=status, explanation=explanation, evidence=evidence,
                                 normalized_values=[a,b], applicable_fact=applicable, method='deterministic', amendment=basis['effective_from'] if basis else None))
    return findings
