"""Reference-aware ranking; normalization affects search only, never stored quotes."""
import re

CONFUSABLES=str.maketrans({'A':'А','B':'В','C':'С','E':'Е','H':'Н','K':'К','M':'М','O':'О','P':'Р','T':'Т','X':'Х'})
CODE=re.compile(r'(?<!\w)([A-ZА-Я]{2,6})\s*\.\s*([0-9ЗО]{1,3})(?!\w)')
SECTION=re.compile(r'(?<![\w.])\d{1,2}(?:\.\d{1,3}){1,4}(?![\w.])')


def normalize_codes(text):
    return CODE.sub(lambda m:m[1].translate(CONFUSABLES)+'.'+m[2].replace('З','3').replace('О','0'),text.upper())


def references(question):
    codes=[m.group(0) for m in CODE.finditer(normalize_codes(question))]
    return list(dict.fromkeys(codes+SECTION.findall(question)))


def reference_score(text, refs):
    if not refs:return 0
    value=normalize_codes(text)
    score=0
    for ref in refs:
        escaped=re.escape(ref)
        if re.search(r'(?m)^\s*'+escaped+r'(?![\w.])',value):score=max(score,2)
        elif re.search(r'(?<![\w.])'+escaped+r'(?![\w.])',value):score=max(score,1)
    return score


def diversify(ranked, rows, limit, documents):
    if documents<=1:return ranked[:limit]
    selected=[]; counts={}
    for key in ranked:
        doc=rows[key]['doc_id']
        if counts.get(doc,0)<2:
            selected.append(key);counts[doc]=counts.get(doc,0)+1
            if len(selected)==limit:return selected
    # Preserve relevance when fewer distinct documents actually have candidates.
    return (selected+[key for key in ranked if key not in selected])[:limit]
