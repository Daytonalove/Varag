"""End-to-end regressions for general chat, summaries and content refusals."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from document_rag.bank.answering import AnswerService
from document_rag.bank.fast_generation import analyze_fast, _select_sources
from document_rag.bank.generation import GigaChatRAG
from document_rag.bank.retrieval import Retriever
from document_rag.bank.store import Store
from document_rag.provider import ProviderError


def reply(text='Здравствуйте! Чем могу помочь?', ids=None, finish='stop'):
    return {'model': 'test', 'choices': [{'finish_reason': finish, 'message': {'content': json.dumps({
        'claims': [{'text': text, 'source_ids': ids or []}], 'limitations': []})}}]}


class ChatTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.store = Store(Path(temp.name) / 'test.sqlite3')
        self.project = self.store.create_project('Chat', 'tester')['id']
        self.provider = GigaChatRAG({'model': 'test'})
        self.service = AnswerService(self.store, self.provider, automatic_external=True)

    def test_new_chat_answers_without_documents_and_without_false_citations(self):
        with patch.object(self.provider, 'call', return_value=reply()) as call:
            answer = self.service.ask(self.project, 'tester', 'Привет, как дела?', True)
        self.assertEqual(call.call_count, 1)
        self.assertIn('Здравствуйте', answer['answer_text'])
        self.assertEqual(answer['mode'], 'gigachat_general')
        self.assertEqual(answer['verification_status'], 'general_knowledge')
        self.assertEqual(answer['sources'], [])
        self.assertEqual(answer['evidence'], [])
        self.assertTrue(answer['answer_limitations'])

    def test_document_question_without_evidence_does_not_invent_answer(self):
        with patch.object(self.provider, 'call') as call:
            answer = self.service.ask(self.project, 'tester', 'Что написано в документе?', True)
        call.assert_not_called()
        self.assertEqual(answer['claims'], [])
        self.assertIn('не найдено', answer['answer_text'])

    def test_general_greeting_does_not_send_document_text(self):
        self.store.upload(self.project, 'tester', 'test.txt', 'Мониторинг — наблюдение за состоянием системы.'.encode())
        self.store.process_one()
        with patch.object(self.provider, 'call', return_value=reply()) as call:
            answer = self.service.ask(self.project, 'tester', 'Как дела у тебя?', True)
        sent = json.loads(call.call_args.args[1]['messages'][1]['content'])
        self.assertEqual(sent['sources'], [])
        self.assertEqual(answer['mode'], 'gigachat_general')

    def test_blacklist_keeps_retrieval_and_does_not_open_circuit_breaker(self):
        self.store.upload(self.project, 'tester', 'test.txt', 'Мониторинг — наблюдение за состоянием системы.'.encode())
        self.store.process_one()
        with patch.object(self.provider, 'call', return_value=reply(finish='blacklist')) as call:
            for _ in range(3):
                answer = self.service.ask(self.project, 'tester', 'Что такое мониторинг?', True)
        self.assertEqual(call.call_count, 3)
        self.assertEqual(answer['mode'], 'gigachat_failed')
        self.assertTrue(answer['retrieved_sources'])
        self.assertFalse(answer['evidence'])
        self.assertEqual(self.provider.open_until, 0)

    def test_document_answer_requires_real_source_id(self):
        self.store.upload(self.project, 'tester', 'test.txt', 'Мониторинг — наблюдение за состоянием системы.'.encode())
        self.store.process_one()
        with patch.object(self.provider, 'call', return_value=reply('Выдуманный ответ.')):
            answer = self.service.ask(self.project, 'tester', 'Что в документе о мониторинге?', True)
        self.assertEqual(answer['failure']['code'], 'invalid_response')

    def test_summary_retrieves_beginning_and_end_without_query_word_matches(self):
        text = '\n\n'.join(f'Глава {i}. ' + ('Защита ресурсов организации. ' * 60) for i in range(100))
        self.store.upload(self.project, 'tester', 'long.txt', text.encode())
        self.store.process_one()
        retrieval = Retriever(self.store, context_characters=240000, candidate_limit=64).search(
            self.project, 'tester', 'Сделай краткое содержание документа')
        chosen, _ = _select_sources(retrieval['sources'], 'Сделай краткое содержание документа', {'model': 'GigaChat-2'})
        combined = '\n'.join(s['text_exact'] for s in chosen)
        self.assertIn('Глава 0.', combined)
        self.assertIn('Глава 99.', combined)
        self.assertIn('summary_coverage', retrieval['coverage_manifest']['retrieval_passes'])

    def test_plain_text_citations_are_bound_and_unknown_ids_rejected(self):
        from document_rag.bank.fast_generation import _parse_answer
        sources = [{'block_id': 'b', 'text_exact': 'Мониторинг — наблюдение за состоянием системы.'}]
        claims, notes = _parse_answer('Мониторинг — наблюдение. [Q1]\n\nВыдуманная ссылка. [Q99]', sources)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]['evidence'][0]['block_id'], 'b')
        self.assertNotIn('[Q1]', claims[0]['text'])
        self.assertTrue(notes)

    def test_plain_general_answer_needs_no_json(self):
        raw = {'model': 'test', 'choices': [{'finish_reason': 'stop', 'message': {'content': 'Привет! Чем могу помочь?'}}]}
        with patch.object(self.provider, 'call', return_value=raw) as call:
            answer = self.service.ask(self.project, 'tester', 'Привет!', True)
        self.assertIn('Чем могу помочь?', answer['answer_text'])
        self.assertEqual(call.call_args.args[1]['response_format'], {'type': 'text'})

    def test_summary_refusal_is_labelled_local_excerpt_fallback(self):
        self.store.upload(self.project, 'tester', 'test.txt', 'Мониторинг — наблюдение за состоянием системы.'.encode())
        self.store.process_one()
        with patch.object(self.provider, 'call', return_value=reply(finish='blacklist')) as call:
            answer = self.service.ask(self.project, 'tester', 'Сделай краткое содержание документа', True)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(answer['mode'], 'local_summary')
        self.assertEqual(answer['failure']['code'], 'finish_blacklist')
        self.assertIn('локальная подборка', answer['answer_text'])
        self.assertTrue(answer['sources'])
        self.assertIsNone(answer['model_version'])
        for evidence in answer['evidence']:
            self.assertTrue(any(evidence['quote'] in s['text_exact'] for s in answer['sources']))

    def test_numbered_markdown_items_are_not_mistaken_for_document_facts(self):
        from document_rag.bank.fast_generation import _guard_claims, _parse_answer
        sources = [{'block_id': 'b', 'text_exact': 'Состав мер приведен в таблице 42.', 'section_path': ''}]
        claims, _ = _parse_answer('2. Меры приведены в таблице 42. [Q1]', sources)
        checked, notes = _guard_claims(claims, sources, 'Есть ли в документе таблица?')
        self.assertEqual(len(checked), 1)
        self.assertEqual(notes, [])
        bad, _ = _parse_answer('2. Меры приведены в таблице 99. [Q1]', sources)
        self.assertEqual(_guard_claims(bad, sources, 'Есть ли в документе таблица?')[0], [])
