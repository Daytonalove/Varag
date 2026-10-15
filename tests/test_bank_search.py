"""Search contract, evidence preservation, failure reporting and concurrent ACL."""
import math
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from document_rag.bank.api import create_app
from document_rag.bank.answering import AnswerService
from document_rag.bank.reranking import LocalCrossEncoder, RankScore, verify_model
from document_rag.bank.retrieval import Retriever
from document_rag.bank.store import Store, NotFound


class Scorer:
    model_id = 'test-cross-encoder@1'

    def __init__(self, target='наблюдение', callback=None):
        self.target, self.callback, self.inputs = target, callback, []

    def score(self, question, texts):
        self.inputs = texts
        if self.callback:
            self.callback()
        return [RankScore(10 if self.target in text else -5, max(0, text.find(self.target)), len(text)) for text in texts]


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / 'search.sqlite3')
        self.project = self.store.create_project('Поиск', 'alice')['id']
        self.noise = self.upload('A.txt', 'Мониторинг серверов: запись ошибок сетевого подключения.')
        self.target = self.upload('Z.txt', 'Мониторинг заёмщика — систематическое наблюдение за его финансовым состоянием.')
        self.scorer = Scorer()
        self.retriever = Retriever(self.store, reranker=self.scorer)

    def upload(self, name, text):
        uploaded = self.store.upload(self.project, 'alice', name, text.encode())
        self.store.process_one()
        return uploaded['document_id']

    def search(self, **kwargs):
        return self.retriever.search(self.project, 'alice', 'Что такое мониторинг?', **kwargs)

    def test_reranks_before_context_budget_with_canonical_provenance(self):
        self.retriever.budget = 450
        result = self.search(top_k=1)
        self.assertEqual(result['sources'][0]['document_id'], self.target)
        self.assertEqual(result['sources'][0]['rerank_score'], 10)
        with self.store.connect() as conn:
            original = conn.execute('SELECT text_exact FROM blocks WHERE id=?', (result['sources'][0]['block_id'],)).fetchone()[0]
        self.assertEqual(original, result['sources'][0]['text_exact'])
        manifest = result['coverage_manifest']
        self.assertEqual(manifest['reranker']['status'], 'applied')
        self.assertNotIn('reranker_disabled', manifest['limitations'])
        self.assertEqual(manifest['retrieved_candidate_count'], manifest['examined_candidate_count'] + manifest['unexamined_candidate_count'])

    def test_document_filter_applied_before_model_and_accounting(self):
        result = self.search(document_ids=[self.target])
        self.assertEqual(result['coverage_manifest']['accessible_document_ids'], [self.target])
        self.assertEqual(len(self.scorer.inputs), 1)
        self.assertNotIn('серверов', self.scorer.inputs[0])

    def test_filter_does_not_allow_foreign_project_or_private_document(self):
        other = self.store.create_project('Чужой', 'bob')['id']
        foreign = self.store.upload(other, 'bob', 'secret.txt', b'secret')['document_id']
        for document in [foreign, 'missing']:
            with self.assertRaises(NotFound):
                self.search(document_ids=[document])
        with self.store.connect() as conn:
            conn.execute('UPDATE documents SET restricted=1 WHERE id=?', (self.target,))
        with self.assertRaises(NotFound):
            self.search(document_ids=[self.target])
        self.search()
        self.assertFalse(any('заёмщика' in text for text in self.scorer.inputs))

    def test_acl_revocation_during_rerank_prevents_release(self):
        def revoke():
            with self.store.connect() as conn:
                conn.execute('UPDATE documents SET restricted=1 WHERE id=?', (self.target,))
        self.scorer.callback = revoke
        with self.assertRaises(NotFound):
            self.search()

    def test_corpus_change_during_inference_rejected(self):
        def change():
            with self.store.connect() as conn:
                conn.execute('UPDATE projects SET index_version=index_version+1 WHERE id=?', (self.project,))
        self.scorer.callback = change
        with self.assertRaisesRegex(ValueError, 'Корпус изменился'):
            self.search()

    def test_failed_or_nonfinite_model_uses_explicit_fallback(self):
        for response in [RuntimeError('private diagnostic'), [RankScore(math.nan, 0, 1)] * 2, []]:
            with patch.object(self.scorer, 'score', side_effect=response if isinstance(response, Exception) else None,
                              return_value=response):
                result = self.search()
            self.assertEqual(len(result['sources']), 2)
            self.assertEqual(result['coverage_manifest']['reranker']['status'], 'failed')
            self.assertNotIn('private diagnostic', str(result))
            self.assertIn('reranker_failed', result['coverage_manifest']['limitations'])

    def test_disabled_is_explicit_in_manifest_and_answer(self):
        self.retriever.reranker = None
        service = AnswerService(self.store, retriever=self.retriever)
        result = service.ask(self.project, 'alice', 'Мониторинг?')
        self.assertIn('reranker_disabled', result['coverage_manifest']['limitations'])
        self.assertTrue(any('Реранкер отключён' in w for w in result['warnings']))

    def test_rerank_cap_accounted(self):
        self.retriever.rerank_limit = 1
        result = self.search()
        self.assertEqual(result['coverage_manifest']['reranker']['candidates'], 1)
        self.assertIn('reranker_candidate_budget', result['coverage_manifest']['limitations'])
        self.assertEqual(result['coverage_manifest']['unexamined_candidate_count'], 1)

    def test_long_block_uses_best_window_not_first_keyword(self):
        text = 'Мониторинг серверов. ' * 600 + 'Нужное наблюдение за состоянием заёмщика.'
        self.store.upload(self.project, 'alice', 'Z.txt', text.encode(), self.target)
        self.store.process_one()
        result = self.search(document_ids=[self.target])
        self.assertIn('Нужное наблюдение', result['sources'][0]['text_exact'])
        self.assertIn('long_block_excerpt', result['coverage_manifest']['limitations'])
        source = result['sources'][0]
        start, end = source['excerpt_range']
        self.assertEqual(text[start:end], source['text_exact'])

    def test_paraphrased_norm_retains_two_values_and_exception(self):
        first = self.upload('limit-a.txt', 'Лимит кредитного комитета составляет 10 млн руб. на сделку.')
        second = self.upload('limit-b.txt', 'Предел полномочий кредитного комитета составляет 15 млн руб. на сделку.')
        exception = self.upload('exception.txt', 'Исключение: кредитный комитет не рассматривает ипотечные сделки.')
        result = self.retriever.search(self.project, 'alice', 'Какую сумму может самостоятельно одобрить кредитный комитет?')
        self.assertTrue(result['normative'])
        self.assertTrue({first, second, exception}.issubset({s['document_id'] for s in result['sources']}))

    def test_search_api_validation_auth_and_no_llm_call(self):
        service = AnswerService(self.store, provider=object(), external_approved=False, retriever=self.retriever)
        app = create_app(self.store, {'test-token': 'alice'}, False, service, True)
        client, headers = app.test_client(), {'X-RAG-Token': 'test-token'}
        path = f'/api/v1/projects/{self.project}/search'
        self.assertEqual(client.post(path, json={'query': 'Мониторинг?'}).status_code, 401)
        for payload in [{'query': ''}, {'query': 'x', 'top_k': True}, {'query': 'x', 'top_k': None}, {'query': 'x', 'top_k': 51},
                        {'query': 'x', 'document_ids': []}, {'query': 'x', 'document_ids': 'all'},
                        {'query': 'x', 'surprise': True}]:
            self.assertEqual(client.post(path, headers=headers, json=payload).status_code, 400, payload)
        response = client.post(path, headers=headers, json={'query': 'Мониторинг?', 'top_k': 1})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['sources'][0]['document_id'], self.target)
        self.assertTrue(client.get('/api/v1/health', headers=headers).json['reranker']['enabled'])
        with self.store.connect() as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM llm_calls').fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT count(*) FROM audit_events WHERE action='search.completed'").fetchone()[0], 1)

    def test_incomplete_index_visible(self):
        pending = self.store.upload(self.project, 'alice', 'pending.txt', b'not ready')['document_id']
        result = self.search()
        self.assertIn(pending, [d['document_id'] for d in result['coverage_manifest']['skipped_documents']])
        self.assertEqual(result['coverage_manifest']['coverage_status'], 'partial')


class WindowTests(unittest.TestCase):
    def test_model_missing_or_corrupted_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'setup_reranker'):
                verify_model(directory)

    def test_long_text_tail_and_query_limits_reported(self):
        import numpy as np
        class Tokenizer:
            def __call__(self, text, **kwargs):
                return {'input_ids': [ord(c) for c in text], 'offset_mapping': [(i, i + 1) for i in range(len(text))]}
            def decode(self, ids, **kwargs):
                return ''.join(chr(c) for c in ids)
            def num_special_tokens_to_add(self, **kwargs):
                return 4
        class Model:
            tokenizer = Tokenizer()
            def predict(self, pairs, **kwargs):
                self.pairs = pairs
                return np.array([10 if 'NEEDLE' in text else -10 for _, text in pairs])
        provider = object.__new__(LocalCrossEncoder)
        provider.model, provider.lock = Model(), threading.Lock()
        provider.max_length, provider.max_windows, provider.batch_size = 384, 3, 2
        text = 'irrelevant ' * 2000 + 'NEEDLE'
        score = provider.score('question ' * 40, [text])[0]
        self.assertEqual(score.score, 10)
        self.assertTrue(text[score.start:score.end].endswith('NEEDLE'))
        self.assertLess(score.windows, score.total_windows)
        self.assertTrue(score.query_truncated)
        self.assertTrue(all(len(q) + len(t) + 4 <= 384 for q, t in provider.model.pairs))


if __name__ == '__main__':
    unittest.main()
