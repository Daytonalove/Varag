"""Synthetic structural-RAG regression and real local model input-bound checks.

--live sends ONLY these synthetic documents to GigaChat. No user database is read.
This is a regression set, not an independent expert acceptance corpus.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from document_rag.config import ROOT
from document_rag.bank.store import Store
from document_rag.bank.runtime import configured_retriever,configured_provider
from document_rag.bank.embeddings import local_encoder
from document_rag.bank.answering import AnswerService
from document_rag.bank.fast_generation import _select_sources,_guard_claims
from document_rag.bank.structural import codes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--report',type=Path,default=Path('evaluation/results/precision-synthetic.json'))
    args=parser.parse_args()
    encoder=local_encoder(ROOT/'models/multilingual-e5-small')
    provider=configured_provider() if args.live else None
    report={'synthetic_only':True,'independent_acceptance':False,'cases':[]}
    with tempfile.TemporaryDirectory() as temp:
        store=Store(Path(temp)/'evaluation.sqlite3',encoder)
        project=store.create_project('Synthetic structural holdout','tester')['id']
        docs=[('Синтетический стандарт 20210.1.txt','ГОСТ Р 20210.1\nКЗД.7 Защита архивных копий шифрованием.\n\nОПС.12 Разделение прав операторов.'),
              ('Синтетический стандарт 20210.2.txt','ГОСТ Р 20210.2\nКЗД.7 Проверка восстановления архивов.\n\nНКЛ.4 Уведомление о смене ключей.'),
              ('Помехи.txt','КЗД.70 Регистрация событий.\n\nОПС.120 Контроль оборудования.\n'*30)]
        for name,text in docs:store.upload(project,'tester',name,text.encode());store.process_one()
        retriever=configured_retriever(store,provider)
        queries=[('Найди КЗД.7 и ОПС.12',['КЗД.7','ОПС.12']),
                 ('Найди НКЛ.4',['НКЛ.4']),('Найди КЗД.999',[]),
                 ('ГОСТ 20210.1: КЗД.7',['КЗД.7']),('ГОСТ 20210.2: КЗД.7',['КЗД.7'])]
        service=AnswerService(store,provider,retriever=retriever,automatic_external=True)
        for question,expected in queries:
            start=time.monotonic();result=retriever.search(project,'tester',question)
            chosen,payload=_select_sources(result['sources'],question)
            claims,limits=_guard_claims([],chosen,question)
            found=set(c for item in claims for c in codes(item['text']))
            item={'question':question,'expected_codes':expected,'found_codes':sorted(found),
                  'retrieval_codes':sorted(set(c for s in result['sources'] for c in codes(s['text_exact']))),
                  'sent_codes':sorted(set(c for s in payload for c in codes(s['text']))),
                  'seconds':round(time.monotonic()-start,3),'passed':found==set(expected)}
            if args.live:
                answer=service.ask(project,'tester',question,True)
                with store.connect() as c:
                    logs=[dict(row) for row in c.execute('SELECT status,stage,metadata FROM llm_calls WHERE question_id=?',(answer['id'],))]
                completions=sum(row['stage']=='fast_answer' for row in logs)
                item.update(answer=answer['answer_text'],mode=answer['mode'],model=answer['model_version'],completion_calls=completions,
                    usage=[json.loads(row['metadata']).get('usage',{}) for row in logs])
                item['passed']=item['passed'] and completions<=1 and answer['mode']!='gigachat_failed'
            report['cases'].append(item);print(json.dumps(item,ensure_ascii=False),flush=True)
        lengths=[]
        input_limit=retriever.reranker.max_length
        torch_model=retriever.reranker.model.model
        forward=torch_model.forward
        def bounded(*a,**kw):
            lengths.append(int(kw['input_ids'].shape[-1]));assert lengths[-1]<=input_limit
            return forward(*a,**kw)
        long='Длинное описание правила идентификации и контроля доступа. '*400
        with patch.object(torch_model,'forward',side_effect=bounded):retriever.reranker.score('Проверить правило '*120,[long])
        report['reranker_max_input_tokens']=max(lengths)
        report['reranker_input_limit']=input_limit
        lengths=[]
        input_limit=encoder.model.max_seq_length
        torch_model=encoder.model[0].auto_model;forward=torch_model.forward
        with patch.object(torch_model,'forward',side_effect=bounded):encoder.encode([long],query=True)
        report['encoder_max_input_tokens']=max(lengths)
        report['encoder_input_limit']=input_limit
    report['passed']=all(r['passed'] for r in report['cases'])
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'reranker_max_input_tokens':report['reranker_max_input_tokens'],'encoder_max_input_tokens':report['encoder_max_input_tokens']}))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
