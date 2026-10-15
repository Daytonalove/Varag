import copy
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

from document_rag.answering import grounded_answer, parse_and_validate
from document_rag.config import ROOT, load
from document_rag.engine import Engine
from document_rag.extraction import extract
from document_rag.provider import GigaChat, ProviderError


class Provider:
    def __init__(self, responses):
        self.responses, self.calls = iter(responses), []
    def generate(self, system, question):
        self.calls.append((system, question))
        return next(self.responses)


def answer(*ids):
    quotes = {'S1': 'Поставка в течение 10 рабочих дней.', 'S2': 'Поставка в течение 15 рабочих дней.'}
    return json.dumps({'status': 'answered', 'claims': [{'text': 'Срок увеличился с 10 до 15 рабочих дней.',
        'evidence': [{'source_id': i, 'quote': quotes[i]} for i in ids]}]})


SOURCES = [dict(source_id='S1', document_id='a', text='Поставка в течение 10 рабочих дней.'),
           dict(source_id='S2', document_id='b', text='Поставка в течение 15 рабочих дней.')]


class AnswerTests(unittest.TestCase):
    def test_comparison_retry(self):
        provider = Provider([answer('S1'), answer('S1', 'S2')])
        result = grounded_answer(provider, 'Как изменился срок?', SOURCES, ['a', 'b'])
        self.assertEqual((result['status'], result['attempts']), ('answered', 2))
        self.assertIn('document_id: b', provider.calls[1][0])
    def test_no_coverage_rejected(self):
        result = grounded_answer(Provider([answer('S1')] * 2), 'Вопрос', SOURCES, ['a', 'b'])
        self.assertEqual(result['status'], 'rejected')
        self.assertEqual(result['claims'], [])
    def test_fabricated_quote(self):
        with self.assertRaises(ValueError):
            parse_and_validate(answer('S1').replace('10', '999'), SOURCES)
    def test_unknown_source(self):
        with self.assertRaises(ValueError):
            parse_and_validate(answer('S1').replace('S1', 'S9'), SOURCES)
    def test_invalid_json_rejected(self):
        result = grounded_answer(Provider(['not json']), 'Вопрос', SOURCES)
        self.assertEqual(result['status'], 'rejected')
    def test_whitespace_normalization(self):
        result = parse_and_validate(answer('S1'), [dict(SOURCES[0], text='Поставка\nв течение 10 рабочих дней.')])
        self.assertEqual(result['status'], 'answered')
    def test_insufficient(self):
        result = grounded_answer(Provider(['{"status":"insufficient","claims":[]}']), 'Вопрос', SOURCES, ['a', 'b'])
        self.assertEqual(result['status'], 'insufficient')


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = json.loads((ROOT / 'config.example.json').read_text())
        self.cfg['storage']['database'] = str(self.root / 'library.sqlite3')
        self.cfg['retrieval']['mode'] = 'lexical'
        self.cfg['ocr']['enabled'] = False
        self.engine = Engine(self.cfg)
        self.ids = []
        for i, source in enumerate(SOURCES):
            path = self.root / f'{i}.txt'; path.write_text(source['text'])
            self.ids.append(self.engine.ingest(path)['id'])
    def test_search_scope(self):
        hits = self.engine.search('Поставка', self.ids[:1])
        self.assertEqual({h['doc_id'] for h in hits}, set(self.ids[:1]))
    def test_invalid_scope(self):
        for ids in ([], ['missing']):
            with self.assertRaises(ValueError):
                self.engine.search('Поставка', ids)
    def test_persistence_original(self):
        new = Engine(self.cfg)
        self.assertEqual(new.original(self.ids[0])[1], SOURCES[0]['text'].encode())
        self.assertEqual(len(new.documents()), 2)
    def test_replace_no_duplicates(self):
        path = self.root / '0.txt'; path.write_text('Изменённый текст документа')
        self.engine.ingest(path)
        self.assertEqual(len(self.engine.documents()), 2)
        self.assertFalse(self.engine.search('Поставка', self.ids[:1]))
    def test_failed_replace_preserves_original(self):
        path = self.root / '0.txt'; path.write_text('')
        with self.assertRaises(ValueError):
            self.engine.ingest(path)
        self.assertEqual(self.engine.original(self.ids[0])[1], SOURCES[0]['text'].encode())
    def ask(self):
        self.engine.provider = Provider([answer('S1', 'S2')])
        return self.engine.ask('Поставка', self.ids, compare=True)
    def test_persist_chat_and_quote_snapshots(self):
        result = self.ask()
        saved = Engine(self.cfg).chat(result['chat_id'])
        self.assertEqual(saved['turns'][0]['answer']['status'], 'answered')
        self.assertEqual(len(saved['turns'][0]['answer']['sources']), 2)
    def test_delete_document_removes_related_chats(self):
        self.ask()
        self.engine.delete_document(self.ids[0])
        self.assertEqual(self.engine.chats(), [])
        self.assertEqual(len(self.engine.documents()), 1)
    def test_backup_is_complete(self):
        self.ask()
        backup = self.root / 'backup.sqlite3'; self.engine.backup(backup)
        cfg = copy.deepcopy(self.cfg); cfg['storage']['database'] = str(backup)
        restored = Engine(cfg)
        self.assertEqual(restored.original(self.ids[0])[1], SOURCES[0]['text'].encode())
        self.assertEqual(len(restored.chats()), 1)
    def test_signature_prevents_silent_mixed_index(self):
        self.cfg['retrieval']['chunk_characters'] = 900
        with self.assertRaises(ValueError):
            Engine(self.cfg)
    def test_history_scope_does_not_leak(self):
        first = self.ask()
        provider = Provider(['{"status":"insufficient","claims":[]}'])
        self.engine.provider = provider
        self.engine.ask('Поставка', self.ids[:1], chat_id=first['chat_id'])
        self.assertEqual(len(provider.calls), 1)  # no rewrite using previous, different scope
        self.assertNotIn('history', provider.calls[0][1])
    def test_history_used_in_same_scope(self):
        first = self.ask()
        provider = Provider(['Поставка', answer('S1', 'S2')])
        self.engine.provider = provider
        self.engine.ask('А новый срок?', self.ids, compare=True, chat_id=first['chat_id'])
        self.assertEqual(len(provider.calls), 2)
        self.assertIn('history', provider.calls[0][1])
    def test_replacement_invalidates_history_for_rewrite(self):
        first = self.ask()
        path = self.root / '0.txt'; path.write_text('Поставка в течение 30 рабочих дней.')
        self.engine.ingest(path)
        provider = Provider(['{"status":"insufficient","claims":[]}'])
        self.engine.provider = provider
        self.engine.ask('Поставка', self.ids, compare=True, chat_id=first['chat_id'])
        self.assertEqual(len(provider.calls), 1)
        self.assertNotIn('history', provider.calls[0][1])
    def test_docx_paragraph_and_table(self):
        path = self.root / 'table.docx'
        xml = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Договор</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>15 дней</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>'
        with zipfile.ZipFile(path, 'w') as file:
            file.writestr('word/document.xml', xml)
        parts, warnings = extract(path, self.cfg['ocr'])
        self.assertEqual(parts, [('блок 1', 'Договор'), ('блок 2', '15 дней')])
        self.assertTrue(warnings)
    def test_api_auth_scope_upload_and_download(self):
        from document_rag.api import create_app
        app = create_app(self.engine, 'test-secret')
        self.addCleanup(app.extensions['executor'].shutdown, wait=True)
        client = app.test_client(); headers = {'X-RAG-Token': 'test-secret'}
        self.assertEqual(client.get('/documents').status_code, 401)
        self.assertEqual(client.get('/documents', headers=headers).status_code, 200)
        self.assertEqual(client.post('/search', json={'question': 'test'}, headers=headers).status_code, 400)
        result = client.post('/documents', data={'file': (io.BytesIO('Новый документ'.encode()), 'new.txt')}, headers=headers)
        self.assertEqual(result.status_code, 202)
        for _ in range(100):
            job = client.get('/jobs/' + result.json['job_id'], headers=headers).json
            if job['status'] in ('done', 'failed'): break
            time.sleep(.01)
        self.assertEqual(job['status'], 'done')
        response = client.get('/documents/' + job['result']['id'] + '/original', headers=headers)
        self.assertEqual(response.data, 'Новый документ'.encode())
    def test_hybrid_fusion(self):
        class Encoder:
            def encode(self, texts, query=False):
                return [[1, 0] if '10' in text or query else [0, 1] for text in texts]
        cfg = copy.deepcopy(self.cfg); cfg['storage']['database'] = str(self.root / 'hybrid.sqlite3'); cfg['retrieval']['mode'] = 'hybrid'
        engine = Engine(cfg, encoder=Encoder())
        ids = [engine.ingest(self.root / f'{i}.txt')['id'] for i in range(2)]
        hits = engine.search('Когда привезут заказ', ids)
        self.assertEqual(hits[0]['doc_id'], ids[0])


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / 'config.example.json').read_text())['gateway']
        self.provider = GigaChat(self.cfg)
        self.key = patch.dict(os.environ, {'GIGACHAT_CREDENTIALS': 'secret-key'}); self.key.start(); self.addCleanup(self.key.stop)
    def test_token_cache_and_expiry_milliseconds(self):
        with patch.object(self.provider, 'request', return_value={'access_token': 'bearer', 'expires_at': (time.time() + 1800) * 1000}) as request:
            self.assertEqual(self.provider.token(), 'bearer')
            self.assertEqual(self.provider.token(), 'bearer')
            self.assertEqual(request.call_count, 1)
    def test_401_refreshes_once(self):
        responses = [{'access_token': 'first', 'expires_at': (time.time() + 1800) * 1000}, ProviderError('unauthorized', 401),
                     {'access_token': 'second', 'expires_at': (time.time() + 1800) * 1000}, {'data': [{'id': 'GigaChat-2'}]}]
        with patch.object(self.provider, 'request', side_effect=responses) as request:
            self.assertEqual(self.provider.models(), ['GigaChat-2'])
            self.assertEqual(request.call_count, 4)
    def test_expired_oauth_rejected(self):
        with patch.object(self.provider, 'request', return_value={'access_token': 'bearer', 'expires_at': 1}):
            with self.assertRaises(ProviderError): self.provider.token()
    def test_redirects_disabled(self):
        from document_rag.provider import NoRedirect
        self.assertIsNone(NoRedirect().redirect_request(None, None, 302, '', {}, 'https://elsewhere.invalid'))
    def test_http_error_body_not_exposed(self):
        import urllib.error
        error = urllib.error.HTTPError('https://example.invalid', 404, 'error', {}, io.BytesIO(b'{"message":"secret-key private document"}'))
        with patch('urllib.request.OpenerDirector.open', side_effect=error):
            with self.assertRaises(ProviderError) as caught:
                self.provider.request('https://example.invalid/models', {})
        self.assertNotIn('secret-key', str(caught.exception))
        self.assertIn('404', str(caught.exception))
    def test_generation_truncated_rejected(self):
        with patch.object(self.provider, 'call', return_value={'choices': [{'finish_reason': 'length', 'message': {'content': 'partial'}}]}):
            with self.assertRaises(ProviderError): self.provider.generate('system', 'question')
    def test_insecure_url_rejected(self):
        with self.assertRaises(ProviderError): self.provider.request('http://example.invalid', {})


if __name__ == '__main__':
    unittest.main()
