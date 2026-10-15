"""Opt-in helper: temporary SQLite + real worker + API + GigaChat, synthetic files only."""
import io
from pathlib import Path
import tempfile
import time

from document_rag.bank.api import create_app
from document_rag.bank.answering import AnswerService
from document_rag.bank.embeddings import local_encoder
from document_rag.bank.store import Store
from document_rag.bank.runtime import configured_retriever
from document_rag.config import ROOT


def run(provider):
    with tempfile.TemporaryDirectory(prefix='bank-mvp-smoke-') as temporary:
        model_dir = ROOT / 'models/multilingual-e5-small'
        encoder = local_encoder(model_dir) if model_dir.is_dir() else None
        store = Store(Path(temporary) / 'audit.sqlite3', encoder)
        service = AnswerService(store, provider, True, configured_retriever(store, provider))
        app = create_app(store, {'synthetic-test-token': 'tester'}, service=service, demo_mode=True)
        client, headers = app.test_client(), {'X-RAG-Token': 'synthetic-test-token'}
        try:
            project = client.post('/api/v1/projects', headers=headers, json={'name': 'Изолированный синтетический MVP smoke'}).json['id']
            files = [(io.BytesIO(p.read_bytes()), p.name) for p in sorted((ROOT / 'fixtures/synthetic_documents').glob('*.txt'))]
            uploaded = client.post(f'/api/v1/projects/{project}/documents', headers=headers, data={'external_allowed': 'true', 'files': files})
            if uploaded.status_code != 202 or any('error' in r for r in uploaded.json):
                raise RuntimeError('Synthetic upload failed')
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                docs = client.get(f'/api/v1/projects/{project}/documents', headers=headers).json
                if all(d['status'] in ('ready', 'ready_with_warnings') for d in docs):
                    break
                time.sleep(.1)
            else:
                raise RuntimeError('Synthetic indexing did not finish')
            searched = client.post(f'/api/v1/projects/{project}/search', headers=headers,
                                   json={'query': 'Что такое мониторинг?', 'top_k': 5})
            if searched.status_code != 200 or searched.json['coverage_manifest']['reranker']['status'] != 'applied':
                raise RuntimeError('Real reranking did not succeed')
            if 'Мониторинг' not in searched.json['sources'][0]['text_exact']:
                raise RuntimeError('Definition is not ranked first')
            def ask(question, conversation=None):
                response = client.post(f'/api/v1/projects/{project}/questions', headers=headers,
                    json={'question': question, 'use_llm': True, 'conversation_id': conversation})
                if response.status_code != 202:
                    raise RuntimeError('Question submission failed')
                job = response.json
                deadline = time.monotonic() + 240
                while time.monotonic() < deadline:
                    state = client.get('/api/v1/questions/' + job['id'], headers=headers).json
                    if state['status'] in ('done', 'failed'):
                        if state['status'] != 'done':
                            raise RuntimeError('Question job failed: ' + state.get('error', 'unknown'))
                        return state['answer'], job['conversation_id']
                    time.sleep(.2)
                raise RuntimeError('Question timeout')
            first, conversation = ask('Какие лимиты установлены для кредитного комитета? Укажи оба значения, если документы расходятся.')
            second, _ = ask('Что такое мониторинг?', conversation)
            if first['mode'] != 'gigachat_answer' or second['mode'] != 'gigachat_answer' or not second['claims']:
                raise RuntimeError('Semantic answer did not pass validation: ' + str({'first': first['mode'], 'second': second['mode'], 'first_warnings': first['warnings'], 'second_warnings': second['warnings']}))
            conflict_values = first['answer_text']
            if '10' not in conflict_values or '15' not in conflict_values:
                raise RuntimeError('Conflict answer does not include both values')
            if 'наблюден' not in second['answer_text'].lower():
                raise RuntimeError('Definition not answered')
            with store.connect() as conn:
                attempts = conn.execute('SELECT count(*) FROM llm_calls').fetchone()[0]
            return {'documents': len(docs), 'retrieval': 'hybrid' if encoder else 'lexical', 'answers': 2,
                    'reranker': first['coverage_manifest']['reranker'], 'search_api': True,
                    'generated_claims': [len(first['claims']), len(second['claims'])],
                    'first_status': first['verification_status'], 'followup_answered': True,
                    'async_jobs': True, 'llm_attempts_recorded': attempts, 'model_version': second['model_version'],
                    'synthetic_only': True, 'persistent_user_data_modified': False}
        finally:
            app.extensions['bank_worker'].close()
