"""Architecture audit regressions; no real API or user data."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import unittest

from document_rag.answering import parse_and_validate
import test_core


class AuditTests(unittest.TestCase):
    setUp = test_core.EngineTests.setUp

    def pending_answer(self):
        started, release = threading.Event(), threading.Event()
        class SlowProvider:
            def generate(self, system, question):
                started.set()
                if not release.wait(5):
                    raise AssertionError('Test provider timed out')
                return test_core.answer('S1', 'S2')
        self.engine.provider = SlowProvider()
        pool = ThreadPoolExecutor(max_workers=2)
        self.addCleanup(pool.shutdown, wait=True)
        self.addCleanup(release.set)
        future = pool.submit(self.engine.ask, 'Поставка', self.ids, True)
        self.assertTrue(started.wait(2))
        return pool, future, release

    def test_search_available_during_generation(self):
        pool, pending, release = self.pending_answer()
        search = pool.submit(self.engine.search, 'Поставка', self.ids)
        self.assertEqual(len(search.result(timeout=1)), 2)
        release.set()
        self.assertEqual(pending.result(timeout=2)['status'], 'answered')

    def test_replaced_document_during_generation_rejects_stale_answer(self):
        pool, pending, release = self.pending_answer()
        path = self.root / '0.txt'
        path.write_text('Поставка в течение 30 рабочих дней.')
        pool.submit(self.engine.ingest, path).result(timeout=1)
        release.set()
        with self.assertRaisesRegex(ValueError, 'изменились'):
            pending.result(timeout=2)
        self.assertEqual(self.engine.chats(), [])

    def test_deleted_chat_not_resurrected(self):
        self.engine.provider = test_core.Provider([test_core.answer('S1', 'S2')])
        chat_id = self.engine.ask('Поставка', self.ids, True)['chat_id']
        started, release = threading.Event(), threading.Event()
        class SlowProvider:
            def generate(self, system, question):
                started.set()
                if not release.wait(5):
                    raise AssertionError('Test provider timed out')
                return 'Поставка' if 'Переформулируй' in system else test_core.answer('S1', 'S2')
        self.engine.provider = SlowProvider()
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = pool.submit(self.engine.ask, 'А срок?', self.ids, True, chat_id)
            try:
                self.assertTrue(started.wait(2))
                pool.submit(self.engine.delete_chat, chat_id).result(timeout=1)
            finally:
                release.set()
            with self.assertRaisesRegex(ValueError, 'изменился|не найден'):
                pending.result(timeout=2)
        self.assertEqual(self.engine.chats(), [])

    def test_real_quote_does_not_prove_claim(self):
        # Documents a limitation, NOT an entailment safeguard.
        raw = json.dumps({'status': 'answered', 'claims': [{'text': 'Поставка через 999 дней.',
            'evidence': [{'source_id': 'S1', 'quote': test_core.SOURCES[0]['text']}]}]})
        result = parse_and_validate(raw, test_core.SOURCES)
        self.assertEqual(result['status'], 'answered')
        self.assertIn('999', result['claims'][0]['text'])

    def test_answer_exposes_verification_limits_and_revision(self):
        self.engine.provider = test_core.Provider([test_core.answer('S1', 'S2')])
        result = self.engine.ask('Поставка', self.ids, True)
        self.assertEqual(result['validation'], {'citations_checked': True, 'entailment_checked': False})
        revisions = {d['id']: d['revision'] for d in self.engine.documents()}
        for source in result['sources']:
            self.assertEqual(source['revision'], revisions[source['document_id']])

    def test_conflict_returns_http_409(self):
        from unittest.mock import patch
        from document_rag.api import create_app
        from document_rag.engine import StateConflict
        app = create_app(self.engine, 'test-secret')
        self.addCleanup(app.extensions['executor'].shutdown, wait=True)
        with patch.object(self.engine, 'ask', side_effect=StateConflict('Документы изменились')):
            result = app.test_client().post('/ask', headers={'X-RAG-Token': 'test-secret'},
                json={'question': 'Поставка', 'document_ids': self.ids})
        self.assertEqual(result.status_code, 409)

    def test_upload_batch_continues_after_bad_file(self):
        import io
        import os
        import requests
        from unittest.mock import patch
        from streamlit.testing.v1 import AppTest
        from document_rag.api import create_app
        from document_rag.config import ROOT, local_token
        config = self.root / 'config.json'
        config.write_text(json.dumps(self.cfg))
        app = create_app(self.engine, local_token(self.root))
        self.addCleanup(app.extensions['executor'].shutdown, wait=True)
        client = app.test_client()
        empty, good = io.BytesIO(b''), io.BytesIO('Рабочий документ после ошибочного'.encode())
        empty.name, good.name = 'empty.txt', 'batch-good.txt'
        def request(method, url, **kwargs):
            data = None
            if kwargs.get('files'):
                name, content = kwargs['files']['file']
                data = {'file': (io.BytesIO(content), name)}
            result = client.open(url.split('8765')[-1], method=method,
                headers=kwargs.get('headers'), json=kwargs.get('json'), data=data)
            response = requests.Response()
            response.status_code, response._content = result.status_code, result.data
            return response
        with patch.dict(os.environ, {'RAG_CONFIG': str(config), 'RAG_API_URL': 'http://127.0.0.1:8765'}), \
                patch('requests.request', side_effect=request), \
                patch('streamlit.file_uploader', return_value=[empty, good]):
            at = AppTest.from_file(str(ROOT / 'ui.py'), default_timeout=20).run()
            next(b for b in at.button if b.label == 'Загрузить и обработать').click().run()
            self.assertEqual(len(at.exception), 0)
            good_id = next(d['id'] for d in self.engine.documents() if d['name'] == good.name)
            self.assertIn(good_id, at.session_state['selected_docs'])
            self.assertTrue(any('empty.txt' in e.value for e in at.error))
