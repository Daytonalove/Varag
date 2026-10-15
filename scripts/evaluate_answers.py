"""Opt-in GigaChat test; semantic correctness requires manual review."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from document_rag.config import load
from document_rag.engine import Engine
from document_rag.provider import ProviderError

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--database',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--live',action='store_true',help='Call GigaChat and consume account quota')
    p.add_argument('--ids',default='pair-1,pair-2,pair-4,pair-6,pair-7,gost-link,absent,injection')
    a=p.parse_args();manifest=json.loads(a.manifest.read_text())
    wanted=a.ids.split(',');lookup={q['id']:q for q in manifest['questions']}
    if not 1<=len(wanted)<=30 or any(i not in lookup for i in wanted):raise ValueError('Choose 1–30 known question IDs.')
    if not a.live:
        print('План:',', '.join(wanted))
        print('Обычный RAG-запрос по всей базе на каждый вопрос. Для выполнения добавьте --live.');return
    if not a.database.is_file():raise ValueError('Сначала выполните evaluate_corpus.py.')
    cfg=load();cfg['storage']['database']=str(a.database.resolve())
    engine=Engine(cfg);ids=[d['id'] for d in engine.documents()]
    if not ids:raise ValueError('База пуста.')
    if cfg['gateway']['model'] not in engine.provider.models():raise ValueError('Выбранной модели нет в /models.')
    report={'model':cfg['gateway']['model'],'scope':'all documents; normal retrieval, independent chats',
            'semantic_correctness':'requires manual review; valid quotes are not a correctness score','results':[]}
    a.report.parent.mkdir(parents=True,exist_ok=True)
    for qid in wanted:
        q=lookup[qid];start=time.perf_counter()
        try:
            answer=engine.ask(q['question'],ids)
            row={'id':qid,'expected_relation':q['kind'],'expected_documents':q['expected_documents'],
                 'seconds':round(time.perf_counter()-start,2),'answer':answer}
        except Exception as exc:
            row={'id':qid,'error':str(exc) if isinstance(exc,(ProviderError,ValueError)) else type(exc).__name__}
        report['results'].append(row)
        a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(qid,row.get('answer',{}).get('status',row.get('error')),flush=True)
    print('Вручную проверьте противоречия, области применения, совместимые сроки, отсутствие данных и недоверенное вложение.')

if __name__=='__main__':main()
