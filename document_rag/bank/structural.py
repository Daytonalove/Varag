"""Version-bound structural identifiers. Normalization never changes canonical text."""
import re

VERSION = 'structural-1'
HOMOGLYPHS = str.maketrans('ABCEHKMOPTXYI', 'АВСЕНКМОРТХУІ')
DIGITS = str.maketrans({'З':'3','О':'0','І':'1','I':'1','L':'1'})
CODE = re.compile(r'(?<![\w])([А-ЯЁA-Z]{2,6})[ \t\u00a0]*[.．·][ \t\u00a0]*([0-9ЗОІIL]{1,3})(?!\w|\.\d)', re.I)
NON_CODES = {'РУБ','СТР','РИС','ТАБ','ГОСТ'}


def occurrences(text):
    matches=[]
    for match in CODE.finditer(text):
        prefix=match[1].upper().translate(HOMOGLYPHS)
        if prefix in NON_CODES:
            continue
        suffix=match[2].upper().translate(DIGITS)
        if not suffix.isdigit():
            continue
        key=prefix+'.'+str(int(suffix))
        line_start=text.rfind('\n',0,match.start())+1
        leading=text[line_start:match.start()].strip(' \t|│;')
        definition=not leading or leading in ('Мера','мера')
        matches.append(dict(key=key,start=match.start(),end=match.end(),definition=definition,raw=match[0]))
    return matches


def codes(text):
    return list(dict.fromkeys(m['key'] for m in occurrences(text)))


def rows(text):
    found=occurrences(text)
    result=[]
    for i,m in enumerate(found):
        # A reference in running prose is searchable but never becomes a definition.
        end=next((n['start'] for n in found[i+1:] if n['definition']),len(text))
        if not m['definition']:
            end=text.find('\n',m['end'])
            if end<0:end=len(text)
        result.append(dict(m,span_end=end))
    return result


def query_pages(question):
    return list(dict.fromkeys(int(n) for n in re.findall(r'(?:страниц\w*|стр\.)\s*(\d{1,5})',question,re.I)))


def standards(text):
    return list(dict.fromkeys(s.replace('–','-') for s in re.findall(r'\b\d{4,6}\.\d+(?:[-–]\d{4})?\b',text)))


def indexed_version(normalizer):
    return normalizer+'+'+VERSION
