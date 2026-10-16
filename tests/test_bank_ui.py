"""Varyag UI journeys use the real API, queue, retrieval and source snapshots."""
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest

from document_rag.bank.api import create_app
from document_rag.bank.answering import AnswerService, GigaChatEvidence
from document_rag.bank.retrieval import Retriever
from document_rag.bank.reranking import RankScore
from document_rag.bank.store import Store, NotFound
from document_rag.config import ROOT, local_token


class Provider(GigaChatEvidence):
    seen = []
    def models(self):
        return ['GigaChat-Pro', 'GigaChat-Max', 'Embeddings']
    def select(self, question, sources):
        self.seen.append(self.cfg['model'])
        return ([{'block_id':s['block_id'],'quote':s['text_exact']} for s in sources],
                {'model':self.cfg['model']})


class BankUITests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.database=Path(temp.name)/'db.sqlite3'
        self.store=Store(self.database)
        self.project=self.store.create_project('Мои документы','local-auditor')['id']
        self.provider=Provider({'model':'GigaChat-Pro'})
        self.provider.seen=[]
        class Reranker:
            model_id='test-cross-encoder'
            calls=0
            def score(self,question,texts):
                self.calls+=1
                return [RankScore(10,0,len(t)) for t in texts]
        self.reranker=Reranker()
        self.service=AnswerService(self.store,self.provider,
            retriever=Retriever(self.store,reranker=self.reranker),automatic_external=True)
        self.token=local_token(self.database.parent)
        self.app=create_app(self.store,{self.token:'local-auditor','other':'other'},False,self.service,True)
        self.client=self.app.test_client()
        self.headers={'X-RAG-Token':self.token}

    def request(self,method,url,**kwargs):
        result=self.client.open(url.split('8766')[-1],method=method,headers=kwargs.get('headers'),json=kwargs.get('json'))
        if method=='POST' and url.endswith('/questions') and result.status_code==202:
            self.app.extensions['bank_questions'].process_one()
        response=requests.Response();response.status_code,response._content=result.status_code,result.data
        return response

    def add_document(self):
        result=self.store.upload(self.project,'local-auditor','rules.txt','Мониторинг заёмщика — наблюдение за финансовым состоянием.'.encode())
        self.store.process_one()
        return result['document_id']

    def test_chat_model_sources_and_new_chat_isolation(self):
        self.add_document()  # Stored with the old external_allowed=false: now allowed by Varyag.
        pending=self.store.upload(self.project,'local-auditor','additional.txt','Порядок резервного копирования.'.encode())
        with self.store.connect() as conn:
            conn.execute("UPDATE versions SET status='ocr' WHERE id=?",(pending['version_id'],))
        with patch.dict(os.environ,{'BANK_DATABASE':str(self.database),'BANK_API_URL':'http://127.0.0.1:8766'}),patch('requests.request',side_effect=self.request):
            ui=AppTest.from_file(str(ROOT/'bank_ui.py'),default_timeout=20).run()
            self.assertEqual(len(ui.exception),0)
            self.assertEqual(ui.title[0].value,'Варяг')
            self.assertEqual(len(ui.checkbox),0)
            self.assertEqual(len(ui.tabs),0)
            self.assertEqual(len(ui.caption),0)
            self.assertTrue(ui.chat_input[0].disabled)
            self.store.process_one()
            ui.run()
            self.assertFalse(ui.chat_input[0].disabled)
            ui.selectbox[0].set_value('GigaChat-Max').run()
            ui.chat_input[0].set_value('Что такое мониторинг заёмщика?').run()
            self.assertEqual(len(ui.exception),0)
            self.assertEqual(self.provider.seen,['GigaChat-Max'])
            self.assertGreater(self.reranker.calls,0)
            self.assertEqual(self.provider.cfg['model'],'GigaChat-Pro')
            self.assertTrue(any('финансовым состоянием' in t.value for t in ui.text))
            next(b for b in ui.button if b.label=='Открыть документ').click().run()
            self.assertEqual(len(ui.exception),0)
            next(b for b in ui.button if b.label=='＋ Новый чат').click().run()
            self.assertFalse(ui.chat_input[0].disabled)
            self.assertFalse(any('финансовым состоянием' in t.value for t in ui.text))
            self.assertEqual(len(ui.exception),0)
            next(b for b in ui.button if b.label=='Что такое мониторинг заёмщика?').click().run()
            self.assertTrue(any('финансовым состоянием' in t.value for t in ui.text))

    def test_upload_automatic_policy_and_durable_model_selection(self):
        result=self.client.post(f'/api/v1/projects/{self.project}/documents',headers=self.headers,
            data={'file':(io.BytesIO('Мониторинг заёмщика — наблюдение за финансовым состоянием.'.encode()),'new.txt')})
        self.assertEqual(result.status_code,202)
        self.store.process_one()
        job=self.client.post(f'/api/v1/projects/{self.project}/questions',headers=dict(self.headers,**{'Idempotency-Key':'one'}),
            json={'question':'Что такое мониторинг?','model_id':'GigaChat-Max'})
        self.assertEqual(job.status_code,202)
        from document_rag.bank.jobs import QuestionQueue
        queue=QuestionQueue(Store(self.database),self.service)
        queue.process_one()
        result=queue.get(job.json['id'],'local-auditor')
        self.assertEqual(result['status'],'done')
        self.assertEqual(result['model_id'],'GigaChat-Max')
        self.assertEqual(result['answer']['model_version'],'GigaChat-Max')
        repeated=self.client.post(f'/api/v1/projects/{self.project}/questions',headers=dict(self.headers,**{'Idempotency-Key':'one'}),
            json={'question':'Что такое мониторинг?','model_id':'GigaChat-Pro'})
        self.assertEqual(repeated.status_code,400)
        invalid=self.client.post(f'/api/v1/projects/{self.project}/questions',headers=self.headers,
            json={'question':'Что такое мониторинг?','model_id':'invented'})
        self.assertEqual(invalid.status_code,400)
        self.assertEqual(self.client.get('/api/v1/provider/models',headers=self.headers).json['models'],['GigaChat-Pro','GigaChat-Max'])
        self.assertEqual(self.client.get('/api/v1/chats',headers={'X-RAG-Token':'other'}).json,[])

    def test_automatic_external_keeps_acl_checks(self):
        document=self.add_document()
        with self.store.connect() as conn:
            conn.execute('UPDATE documents SET restricted=1 WHERE id=?',(document,))
        answer=self.service.ask(self.project,'local-auditor','Мониторинг?',True)
        self.assertEqual(answer['sources'],[])
        self.assertEqual(self.provider.seen,[])
        with self.assertRaises(NotFound):
            self.service.ask(self.project,'other','Мониторинг?',True)

    def test_delete_chat_is_user_scoped_and_keeps_other_chats_and_documents(self):
        document = self.add_document()
        with self.store.connect() as conn:
            for chat, user in [('first', 'local-auditor'), ('second', 'local-auditor'), ('foreign', 'other')]:
                conn.execute('INSERT INTO conversations(id,project_id,user_id,title) VALUES(?,?,?,?)',
                             (chat, self.project, user, chat))
        path = f'/api/v1/chats/{self.project}/first'
        self.assertEqual(self.client.delete(path, headers={'X-RAG-Token': 'other'}).status_code, 404)
        self.assertEqual(self.client.delete(f'/api/v1/chats/{self.project}/foreign', headers=self.headers).status_code, 404)
        self.assertEqual(self.client.delete(path, headers=self.headers).status_code, 200)
        self.assertEqual(self.client.delete(path, headers=self.headers).status_code, 200)
        chats = self.client.get('/api/v1/chats', headers=self.headers).json
        self.assertEqual([c['conversation_id'] for c in chats], ['second'])
        self.assertEqual(self.store.documents(self.project, 'local-auditor')[0]['id'], document)
        self.client.delete(f'/api/v1/chats/{self.project}/second', headers=self.headers)
        self.assertEqual(self.client.get('/api/v1/chats', headers=self.headers).json, [])

    def test_delete_last_empty_chat_in_sidebar(self):
        with patch.dict(os.environ, {'BANK_DATABASE': str(self.database), 'BANK_API_URL': 'http://127.0.0.1:8766'}), patch('requests.request', side_effect=self.request):
            ui = AppTest.from_file(str(ROOT / 'bank_ui.py'), default_timeout=20).run()
            next(b for b in ui.button if (b.key or '').startswith('delete_chat_')).click().run()
            self.assertEqual(len(ui.exception), 0)
            self.assertFalse(any(b.label == 'Мои документы' for b in ui.button))
            self.assertEqual(self.client.get('/api/v1/chats', headers=self.headers).json, [])
            next(b for b in ui.button if b.label == '＋ Новый чат').click().run()
            self.assertEqual(len(ui.exception), 0)
            self.assertEqual(len(self.client.get('/api/v1/chats', headers=self.headers).json), 1)


if __name__ == '__main__':
    unittest.main()
