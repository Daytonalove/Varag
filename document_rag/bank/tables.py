"""Conservative geometry candidates for measure tables, never invented cells.

Only aligned code rows under at least two explicit numbered headers are mapped.
Merged/multi-page tables remain unverified; all raw words/boxes are retained.
"""
from statistics import median
from .structural import codes


def measure_tables(words):
    anchors=[]
    for word in words:
        found=codes(word['text'])
        if len(found)==1 and len(word['text'].strip())<=12:
            anchors.append((word,found[0]))
    anchors.sort(key=lambda pair:(pair[0]['box'][1],pair[0]['box'][0]))
    if not anchors:return []
    left=min(w['box'][0] for w,_ in anchors)
    anchors=[(w,c) for w,c in anchors if abs(w['box'][0]-left)<max(15,(w['box'][3]-w['box'][1])*2)]
    first_y=anchors[0][0]['box'][1]
    numbers=[w for w in words if w['text'].strip() in ('1','2','3','4') and w['box'][1]<first_y and w['box'][0]>anchors[0][0]['box'][2]]
    groups=[]
    for w in sorted(numbers,key=lambda w:w['box'][1]):
        if groups and abs(groups[-1][0]['box'][1]-w['box'][1])<=max(3,(w['box'][3]-w['box'][1])*.6):groups[-1].append(w)
        else:groups.append([w])
    candidates=[g for g in groups if len(g)>=2 and len({w['text'] for w in g})==len(g)]
    if not candidates:return []
    headers=sorted(candidates[-1],key=lambda w:w['box'][0])
    centers=[(h['box'][0]+h['box'][2])/2 for h in headers]
    gap=median([b-a for a,b in zip(centers,centers[1:])])
    if gap<=0:return []
    result=[]
    for i,(anchor,code) in enumerate(anchors):
        line_height=anchor['box'][3]-anchor['box'][1]
        y0=anchor['box'][1]-line_height*.4
        y1=anchors[i+1][0]['box'][1]-line_height*.4 if i+1<len(anchors) else anchor['box'][1]+line_height*8
        row_words=[w for w in words if y0<=w['box'][1]<y1 and w['box'][0]>=left-2]
        cells=[]
        for header,center in zip(headers,centers):
            values=[w for w in row_words if abs((w['box'][0]+w['box'][2])/2-center)<gap*.4]
            values.sort(key=lambda w:(w['box'][1],w['box'][0]))
            # Long prose or duplicate vertical values indicates a layout we cannot map safely.
            if len(values)!=1 or len(values[0]['text'])>8:
                cells=[];break
            cells.append({'header':header['text'],'header_box':header['box'],
                          'text':values[0]['text'],'box':values[0]['box']})
        if cells:
            result.append({'code':code,'code_text':anchor['text'],'code_box':anchor['box'],
                           'cells':cells,'method':'aligned_word_geometry','needs_review':True})
    return result
