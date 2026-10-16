"""50-file synthetic upload/search regression; --live adds ONE GigaChat answer.

Only generated fixtures are sent to the provider. User databases are never read.
"""
import argparse
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from document_rag.bank.api import create_app
from document_rag.bank.answering import AnswerService
from document_rag.bank.context import limits
from document_rag.bank.embeddings import local_encoder
from document_rag.bank.fast_generation import _select_sources
from document_rag.bank.runtime import configured_provider,configured_retriever
from document_rag.bank.store import Store
from document_rag.config import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--report',type=Path,default=Path('evaluation/results/many-documents.json'))
    args=parser.parse_args()
    provider=configured_provider() if args.live else None
    if args.live and not provider:raise SystemExit('GigaChat is not configured')
    policy=limits(provider.cfg if provider else {'model':'GigaChat-2'})
    encoder=local_encoder(ROOT/'models/multilingual-e5-small')
    with tempfile.TemporaryDirectory(prefix='varag-many-') as temp:
        store=Store(Path(temp)/'test.sqlite3',encoder)
        retriever=configured_retriever(store,provider)
        service=AnswerService(store,provider,retriever=retriever,automatic_external=True)
        app=create_app(store,{'synthetic-test-token':'tester'},False,service,True)
        client=app.test_client();headers={'X-RAG-Token':'synthetic-test-token'}
        project=client.post('/api/v1/projects',headers=headers,json={'name':'Synthetic 50 documents'}).json['id']
        files=[]
        for number in range(1,51):
            text=(f'Документ {number}. Мониторинг обеспечивает наблюдение за состоянием системы. '
                  +'Результаты мониторинга фиксируются в журнале. Дежурный проверяет доступность системы и отслеживает изменения её состояния. '*20)
            if number==1:text+=' При сбое сигнал направляют дежурному по телефону.'
            if number==50:text+=' После восстановления связи проводят повторную проверку доступности.'
            files.append((io.BytesIO(text.encode()),f'Правило-{number:02}.txt'))
        uploaded=client.post(f'/api/v1/projects/{project}/documents',headers=headers,data={'files':files})
        assert uploaded.status_code==202 and len(uploaded.json)==50 and not any('error' in r for r in uploaded.json)
        doc_ids={r['document_id'] for r in uploaded.json}
        for i in range(50):
            store.process_one()
            if (i+1)%10==0:print('Indexed',i+1,'documents',flush=True)
        question='Как организован мониторинг в загруженных документах? Укажи общее требование, а также действия при сбое и после восстановления связи.'
        report={'synthetic_only':True,'user_data_modified':False,'uploaded_documents':50,'live':args.live}
        started=time.monotonic()
        if args.live:
            job=client.post(f'/api/v1/projects/{project}/questions',headers=headers,
                            json={'question':question,'model_id':provider.cfg['model']})
            assert job.status_code==202,job.json
            app.extensions['bank_questions'].process_one()
            state=client.get('/api/v1/questions/'+job.json['id'],headers=headers).json
            assert state['status']=='done',state
            answer=state['answer']
            report.update(mode=answer['mode'],answer=answer['answer_text'],failure=answer.get('failure'))
            context=answer['coverage_manifest'].get('generation_context',{})
            report['context']={k:v for k,v in context.items() if not k.endswith('_ids')}
            represented=set(context.get('selected_document_ids',[]))
            with store.connect() as conn:
                logs=[dict(r) for r in conn.execute('SELECT stage,status,metadata FROM llm_calls WHERE question_id=?',(answer['id'],))]
            report['calls']=[dict(stage=r['stage'],status=r['status'],
                                  usage=json.loads(r['metadata']).get('usage',{})) for r in logs]
            report['model_version']=answer['model_version']
            report['passed']=(answer['mode']=='gigachat_answer'
                and context.get('token_count_method')=='provider_components'
                and sum(r['stage']=='fast_answer' for r in logs)==1
                and sum(r['stage']=='tokens_count' for r in logs)==1
                and 'телефон' in answer['answer_text'].lower() and 'восстанов' in answer['answer_text'].lower())
        else:
            local=copy.copy(retriever)
            local.budget=policy['context_characters']*2;local.limit=policy['max_sources']*2
            local.rerank_limit=256;local.diversify_documents=True
            result=local.search(project,'tester',question)
            chosen,payload=_select_sources(result['sources'],question,{'model':'GigaChat-2'})
            represented={s['document_id'] for s in chosen}
            report.update(context_characters=sum(len(s['text']) for s in payload),selected_sources=len(chosen),
                          reranker=result['coverage_manifest']['reranker'],passed=True)
        report['selected_documents']=len(represented)
        report['elapsed_seconds']=round(time.monotonic()-started,2)
        report['passed']=report['passed'] and represented==doc_ids
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
