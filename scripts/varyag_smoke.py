"""Opt-in GigaChat smoke test using only synthetic text and a temporary database."""
import argparse
import io
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from document_rag.config import ROOT
from document_rag.bank.api import create_app
from document_rag.bank.answering import AnswerService
from document_rag.bank.embeddings import local_encoder
from document_rag.bank.runtime import configured_provider,configured_retriever
from document_rag.bank.store import Store


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--report',type=Path,default=Path('evaluation/results/varyag-live.json'))
    args=parser.parse_args()
    provider=configured_provider()
    if provider is None:
        raise RuntimeError('GigaChat is not configured')
    with tempfile.TemporaryDirectory(prefix='varyag-smoke-') as temporary:
        store=Store(Path(temporary)/'smoke.sqlite3',local_encoder(ROOT/'models/multilingual-e5-small'))
        service=AnswerService(store,provider,retriever=configured_retriever(store,provider),automatic_external=True)
        app=create_app(store,{'synthetic-token':'tester'},False,service,True)
        client=app.test_client();headers={'X-RAG-Token':'synthetic-token'}
        project=client.post('/api/v1/chats',json={},headers=headers).json['project_id']
        uploaded=client.post(f'/api/v1/projects/{project}/documents',headers=headers,data={
            'file':(io.BytesIO('Срок подготовки отчёта составляет 7 рабочих дней.'.encode()),'instruction.txt')})
        assert uploaded.status_code==202,uploaded.json
        store.process_one()
        queued=client.post(f'/api/v1/projects/{project}/questions',headers=headers,json={
            'question':'За сколько дней готовится отчёт?','model_id':args.model})
        assert queued.status_code==202,queued.json
        app.extensions['bank_questions'].process_one()
        job=client.get('/api/v1/questions/'+queued.json['id'],headers=headers).json
        answer=job.get('answer',{})
        report=dict(synthetic_only=True,persistent_user_data_modified=False,job_status=job['status'],
                    selected_model=args.model,model_version=answer.get('model_version'),mode=answer.get('mode'),
                    answer=answer.get('answer_text'),error=job.get('error'),
                    reranker=answer.get('coverage_manifest',{}).get('reranker'),
                    evidence=answer.get('evidence'),warnings=answer.get('warnings'))
        report['passed']=(job['status']=='done' and answer.get('mode')=='gigachat_answer'
            and str(answer.get('model_version','')).startswith(args.model)
            and answer.get('coverage_manifest',{}).get('reranker',{}).get('status')=='applied'
            and '7 рабочих дней' in answer.get('answer_text','') and bool(answer.get('evidence')))
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report,ensure_ascii=False))
        if not report['passed']:
            raise SystemExit(1)


if __name__=='__main__':main()
