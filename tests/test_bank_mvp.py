"""SQLite MVP: actual orchestration, immutable corrections, semantic contracts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from document_rag.bank.api import create_app
from document_rag.bank.answering import AnswerService
from document_rag.bank.generation import GigaChatRAG, validate_analysis
from document_rag.bank.ingest import extract_isolated
from document_rag.bank.jobs import QuestionQueue
from document_rag.bank.retrieval import Retriever
from document_rag.bank.store import Store, NotFound
from document_rag.provider import ProviderError


def response(data):
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(data, ensure_ascii=False)}}],
            'model': 'test-model:1', 'usage': {'total_tokens': 10}}


def source(block='b1', text='Лимит кредитного комитета составляет 10 млн руб.'):
    return dict(block_id=block, text_exact=text, document_name='Policy.txt', section_path='4.2', page_number='1')


def analysis(sources):
    return {'claims': [{'text': s['text_exact'], 'evidence': [{'block_id': s['block_id'], 'quote': s['text_exact']}]} for s in sources],
            'assessments': [], 'limitations': []}


class GenerationTests(unittest.TestCase):
    def test_local_encoder_accepts_path_object(self):
        from document_rag.bank.embeddings import local_encoder
        with patch('document_rag.engine.Encoder') as constructor:
            local_encoder(Path('models/local'))
        constructor.assert_called_once_with('models/local')

    def test_real_adapter_uses_generation_and_separate_verifier(self):
        provider = GigaChatRAG({'model': 'test-model'})
        sources = [source()]
        callbacks = []
        with patch.object(provider, 'call', side_effect=[response(analysis(sources)), response({'judgments': [{'index': 0, 'supported': True}]})]) as call:
            result = provider.analyze('Лимит?', sources, before_call=lambda *args: callbacks.append(args))
        self.assertEqual(len(result['claims']), 1)
        self.assertEqual([c['stage'] for c in result['calls']], ['answer', 'entailment'])
        self.assertEqual(len(callbacks), 2)
        self.assertEqual(call.call_count, 2)

    def test_verifier_rejects_unsupported_paraphrase(self):
        provider = GigaChatRAG({'model': 'test-model'})
        data = analysis([source()])
        data['claims'][0]['text'] = 'Кредитный комитет не имеет полномочий.'
        with patch.object(provider, 'call', side_effect=[response(data), response({'judgments': [{'index': 0, 'supported': False}]})]):
            result = provider.analyze('Полномочия?', [source()])
        self.assertEqual(result['claims'], [])
        self.assertTrue(result['errors'])

    def test_missing_assessment_judgment_does_not_discard_supported_fact(self):
        provider = GigaChatRAG({'model': 'test-model'})
        data = analysis([source()])
        data['assessments'] = [{'status': 'needs_review', 'explanation': 'Область применения требует уточнения.', 'evidence': data['claims'][0]['evidence']}]
        with patch.object(provider, 'call', side_effect=[response(data), response({'judgments': [{'index': 0, 'supported': True}]})]):
            result = provider.analyze('Лимит?', [source()])
        self.assertEqual(len(result['claims']), 1)
        self.assertEqual(result['assessments'], [])
        self.assertTrue(result['errors'])

    def test_numeric_hallucination_and_false_ids_rejected(self):
        for field, value in [('text', 'Лимит составляет 999 млн руб.'), ('block_id', 'invented')]:
            data = analysis([source()])
            if field == 'text':
                data['claims'][0]['text'] = value
            else:
                data['claims'][0]['evidence'][0]['block_id'] = value
            with self.assertRaises(ValueError):
                validate_analysis(json.dumps(data), [source()])

    def test_future_or_undated_amendment_cannot_resolve_priority(self):
        sources = [source(), source('b2', 'Лимит составляет 15 млн руб.'),
                   source('b3', 'С 01.07.2026 изменяется пункт 4.2: лимит составляет 15 млн руб.')]
        data = {'claims': [], 'limitations': [], 'assessments': [{'status': 'explained_by_amendment',
            'explanation': 'С 01.07.2026 лимит составляет 15 млн руб.',
            'evidence': [{'block_id': s['block_id'], 'quote': s['text_exact']} for s in sources]}]}
        for as_of in (None, '2026-06-01'):
            result = validate_analysis(json.dumps(data), sources, as_of)
            self.assertEqual(result['assessments'][0]['status'], 'needs_review')
        result = validate_analysis(json.dumps(data), sources, '2026-08-01')
        self.assertEqual(result['assessments'][0]['status'], 'explained_by_amendment')

    def test_422_splits_and_preserves_ids(self):
        sources = [source(), source('b2', 'Лимит кредитного комитета составляет 15 млн руб.')]
        provider = GigaChatRAG({'model': 'test-model'})
        with patch.object(provider, 'call', side_effect=[ProviderError('large', 422), response(analysis(sources[:1])),
             response(analysis(sources[1:])), response({'judgments': [{'index': 0, 'supported': True}, {'index': 1, 'supported': True}]})]):
            result = provider.analyze('Лимиты?', sources)
        self.assertEqual({e['block_id'] for c in result['claims'] for e in c['evidence']}, {'b1', 'b2'})
        self.assertEqual(len(result['examined_block_ids']), 2)

    def test_422_single_block_is_reported_not_success(self):
        provider = GigaChatRAG({'model': 'test-model'})
        with patch.object(provider, 'call', side_effect=ProviderError('large', 422)):
            result = provider.analyze('Лимит?', [source()])
        self.assertEqual(result['claims'], [])
        self.assertEqual(result['examined_block_ids'], [])
        self.assertTrue(result['errors'])

    def test_permission_revocation_before_verification_stops_call(self):
        provider = GigaChatRAG({'model': 'test-model'})
        checks = 0
        def permission(*args):
            nonlocal checks
            checks += 1
            if checks == 2:
                raise ValueError('Permission revoked')
        with patch.object(provider, 'call', return_value=response(analysis([source()]))) as call:
            with self.assertRaises(ValueError):
                provider.analyze('Лимит?', [source()], before_call=permission)
        self.assertEqual(call.call_count, 1)


class MVPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / 'db.sqlite3')
        self.project = self.store.create_project('Тест MVP', 'alice')['id']
        self.doc = self.store.upload(self.project, 'alice', 'A.txt', '4.2 Лимит кредитного комитета составляет 10 млн руб.'.encode())
        self.store.process_one()
        self.service = AnswerService(self.store)
        self.queue = QuestionQueue(self.store, self.service)

    def test_question_queue_idempotent_and_restart_after_publish(self):
        job = self.queue.submit(self.project, 'alice', 'Какой лимит?', False, idempotency_key='request-1')
        again = self.queue.submit(self.project, 'alice', 'Какой лимит?', False, idempotency_key='request-1')
        self.assertEqual(job['id'], again['id'])
        self.assertTrue(self.queue.process_one())
        with self.store.connect() as conn:
            conn.execute("UPDATE question_jobs SET status='running'")
        self.queue.recover()
        self.assertFalse(self.queue.process_one())
        self.assertEqual(self.queue.get(job['id'], 'alice')['status'], 'done')
        with self.store.connect() as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM answers').fetchone()[0], 1)
        with self.assertRaises(ValueError):
            self.queue.submit(self.project, 'alice', 'Другой вопрос', False, idempotency_key='request-1')

    def test_question_user_project_isolation(self):
        job = self.queue.submit(self.project, 'alice', 'Какой лимит?', False)
        with self.assertRaises(NotFound):
            self.queue.get(job['id'], 'bob')
        with self.assertRaises(NotFound):
            self.queue.submit(self.project, 'bob', 'Вопрос', False, conversation_id=job['conversation_id'])

    def test_conversation_fresh_retrieval(self):
        first = self.queue.submit(self.project, 'alice', 'Какой лимит кредитного комитета?', False)
        self.queue.process_one()
        extra = self.store.upload(self.project, 'alice', 'B.txt', 'Лимит кредитного комитета составляет 15 млн руб.'.encode())
        self.store.process_one()
        second = self.queue.submit(self.project, 'alice', 'А по другим документам?', False, conversation_id=first['conversation_id'])
        self.queue.process_one()
        a, b = self.queue.get(first['id'], 'alice')['answer'], self.queue.get(second['id'], 'alice')['answer']
        self.assertGreater(b['index_version'], a['index_version'])
        self.assertIn('кредитного комитета', b['search_question'])
        self.assertIn(extra['document_id'], b['coverage_manifest']['searched_document_ids'])

    def test_invalid_date_fails_at_submission(self):
        for value in ('2026-99-01', 'yesterday', 3):
            with self.assertRaises(ValueError):
                self.queue.submit(self.project, 'alice', 'Вопрос', False, as_of=value)

    def test_original_correction_is_new_version_with_same_original(self):
        before = self.service.ask(self.project, 'alice', 'Лимит?')
        original = self.store.original(self.doc['document_id'], 'alice')[1]
        corrected = self.store.correct_page(self.doc['document_id'], 'alice', 1, '4.2 Лимит составляет 12 млн руб.', self.doc['version_id'])
        self.assertNotEqual(corrected['version_id'], self.doc['version_id'])
        self.assertEqual(self.store.original(self.doc['document_id'], 'alice')[1], original)
        self.assertIn('10 млн', self.store.page(self.doc['document_id'], 1, 'alice', self.doc['version_id'])['text_exact'])
        self.assertIn('12 млн', self.store.page(self.doc['document_id'], 1, 'alice')['text_exact'])
        self.assertIn('10 млн', self.store.answer(before['id'], 'alice')['sources'][0]['text_exact'])
        with self.assertRaises(ValueError):
            self.store.correct_page(self.doc['document_id'], 'alice', 1, 'Обновление', self.doc['version_id'])

    def test_reprocess_preserves_original_and_previous_blocks(self):
        repeat = self.store.reprocess(self.doc['document_id'], 'alice')
        self.store.process_one()
        self.assertNotEqual(repeat['version_id'], self.doc['version_id'])
        self.assertEqual(self.store.page(self.doc['document_id'], 1, 'alice', repeat['version_id'])['text_exact'],
                         self.store.page(self.doc['document_id'], 1, 'alice', self.doc['version_id'])['text_exact'])

    def test_reindex_existing_canonical_blocks(self):
        class Encoder:
            def encode(self, texts, query=False):
                return [[1., 0.] for _ in texts]
        self.store.encoder = Encoder()
        old = self.store.page(self.doc['document_id'], 1, 'alice')['blocks'][0]['id']
        job = self.store.queue_reindex(self.doc['document_id'], 'alice')
        self.assertEqual(self.store.queue_reindex(self.doc['document_id'], 'alice')['id'], job['id'])
        self.store.process_maintenance()
        result = Retriever(self.store).search(self.project, 'alice', 'Какие полномочия?')
        self.assertEqual(result['sources'][0]['block_id'], old)
        self.assertIn('semantic', result['sources'][0]['reasons'])

    def test_backup_consistent_and_no_overwrite(self):
        target = Path(self.temp.name) / 'backup.sqlite3'
        self.store.backup(target)
        restored = Store(target)
        self.assertEqual(restored.original(self.doc['document_id'], 'alice')[1], self.store.original(self.doc['document_id'], 'alice')[1])
        with self.assertRaises(ValueError):
            self.store.backup(target)

    def test_semantic_service_saves_answer_and_attempt_metadata(self):
        with self.store.connect() as conn:
            conn.execute('UPDATE documents SET external_allowed=1')
        provider = GigaChatRAG({'model': 'test-model'})
        sources = Retriever(self.store).search(self.project, 'alice', 'Лимит?')['sources']
        with patch.object(provider, 'call', side_effect=[response(analysis(sources)), response({'judgments': [{'index': 0, 'supported': True}]})]):
            answer = AnswerService(self.store, provider, True).ask(self.project, 'alice', 'Какой лимит?', True)
        self.assertEqual(answer['mode'], 'gigachat_answer')
        self.assertIn('10 млн', answer['answer_text'])
        self.assertIn('[1]', answer['answer_text'])
        with self.store.connect() as conn:
            rows = conn.execute('SELECT * FROM llm_calls').fetchall()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r['status'] == 'succeeded' for r in rows))
        self.assertNotIn('Лимит', ''.join(r['metadata'] for r in rows))

    def test_async_api_and_openapi_contract(self):
        app = create_app(self.store, {'token': 'alice'}, start_worker=False, demo_mode=True)
        client, headers = app.test_client(), {'X-RAG-Token': 'token'}
        response = client.post(f'/api/v1/projects/{self.project}/questions', headers=headers, json={'question': 'Лимит?'})
        self.assertEqual(response.status_code, 202)
        app.extensions['bank_questions'].process_one()
        answer = client.get('/api/v1/questions/' + response.json['id'], headers=headers)
        self.assertEqual(answer.json['status'], 'done')
        spec = client.get('/api/v1/openapi.json', headers=headers)
        self.assertEqual(spec.status_code, 200)
        self.assertEqual(set(spec.json['paths']['/api/v1/projects/{project}/questions']), {'get', 'post'})
        report = client.get('/api/v1/answers/' + response.json['id'] + '/export?format=html', headers=headers)
        self.assertEqual(report.status_code, 200)
        self.assertIn('10 млн', report.data.decode())

    def test_long_block_not_silently_dropped(self):
        long = ('Вводный текст. ' * 1000) + '\nМониторинг — наблюдение за портфелем.' + (' Дополнение.' * 1500)
        self.store.upload(self.project, 'alice', 'long.txt', long.encode())
        self.store.process_one()
        result = Retriever(self.store).search(self.project, 'alice', 'Что такое мониторинг?')
        self.assertTrue(any('Мониторинг' in s['text_exact'] for s in result['sources']))
        self.assertIn('long_block_excerpt', result['coverage_manifest']['limitations'])

    def test_parser_timeout_kills_process_group(self):
        with patch('document_rag.bank.ingest.subprocess.Popen') as process, patch('os.killpg') as kill:
            process.return_value.poll.return_value = None
            process.return_value.pid = 123456
            with self.assertRaises(ValueError):
                extract_isolated(Path(self.temp.name) / 'input.pdf', timeout_seconds=0)
        kill.assert_called_once()


if __name__ == '__main__':
    unittest.main()
