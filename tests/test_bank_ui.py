"""Real Streamlit reruns routed to the bank Flask app with synthetic documents."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest

from document_rag.bank.api import create_app
from document_rag.bank.store import Store
from document_rag.config import ROOT, local_token


class BankUITests(unittest.TestCase):
    def test_local_search_filter_source_and_clear_on_new_version(self):
        from document_rag.bank.answering import AnswerService
        from document_rag.bank.retrieval import Retriever
        from document_rag.bank.reranking import RankScore
        class Reranker:
            model_id = 'ui-test'
            def score(self, question, texts):
                return [RankScore(10 if 'заёмщика' in t else -5, 0, len(t)) for t in texts]
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / 'audit.sqlite3'
            store = Store(database)
            project = store.create_project('Поиск', 'local-auditor')['id']
            document = store.upload(project, 'local-auditor', 'monitoring.txt', 'Мониторинг заёмщика — наблюдение за финансовым состоянием.'.encode())['document_id']
            store.process_one()
            service = AnswerService(store, retriever=Retriever(store, reranker=Reranker()))
            app = create_app(store, {local_token(database.parent): 'local-auditor'}, False, service, True)
            client = app.test_client()
            def request(method, url, **kwargs):
                result = client.open(url.split('8766')[-1], method=method, headers=kwargs.get('headers'), json=kwargs.get('json'))
                response = requests.Response()
                response.status_code, response._content = result.status_code, result.data
                return response
            with patch.dict(os.environ, {'BANK_DATABASE': str(database)}), patch('requests.request', side_effect=request):
                ui = AppTest.from_file(str(ROOT / 'bank_ui.py'), default_timeout=20).run()
                next(e for e in ui.text_input if e.label == 'Поисковый запрос').set_value('Что такое мониторинг?')
                ui.multiselect[0].set_value([document])
                next(b for b in ui.button if b.label == 'Найти фрагменты').click().run()
                self.assertEqual(len(ui.exception), 0)
                self.assertTrue(any('Реранкер обработал 1' in c.value for c in ui.caption))
                self.assertTrue(any('финансовым состоянием' in t.value for t in ui.text))
                next(b for b in ui.button if b.label == 'Открыть оригинал').click().run()
                self.assertEqual(len(ui.exception), 0)
                self.assertTrue(any('Оригинал: monitoring.txt' in h.value for h in ui.subheader))
                store.upload(project, 'local-auditor', 'monitoring.txt', 'Новое определение мониторинга заёмщика.'.encode(), document)
                store.process_one()
                ui.run()
                self.assertTrue(any('Выполните поиск заново' in i.value for i in ui.info))
                self.assertEqual(len(ui.exception), 0)

    def test_project_question_sources_review_and_project_switch(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / 'audit.sqlite3'
            store = Store(database)
            project = store.create_project('Кредитные полномочия', 'local-auditor')['id']
            second = store.create_project('Другой проект', 'local-auditor')['id']
            for index, amount in enumerate((10, 15)):
                store.upload(project, 'local-auditor', f'{index}.txt', f'4.2 Лимит кредитного комитета составляет {amount} млн руб. на сделку.'.encode())
                store.process_one()
            app = create_app(store, {local_token(database.parent): 'local-auditor'}, start_worker=False, demo_mode=True)
            client = app.test_client()
            def request(method, url, **kwargs):
                result = client.open(url.split('8766')[-1], method=method, headers=kwargs.get('headers'), json=kwargs.get('json'))
                if method == 'POST' and url.endswith('/questions') and result.status_code == 202:
                    app.extensions['bank_questions'].process_one()
                response = requests.Response()
                response.status_code, response._content = result.status_code, result.data
                return response
            with patch.dict(os.environ, {'BANK_DATABASE': str(database), 'BANK_API_URL': 'http://127.0.0.1:8766'}), patch('requests.request', side_effect=request):
                ui = AppTest.from_file(str(ROOT / 'bank_ui.py'), default_timeout=20).run()
                self.assertEqual(len(ui.exception), 0)
                ui.selectbox(key='selected_project').set_value(project).run()
                ui.chat_input[0].set_value('Какой лимит кредитного комитета?').run()
                self.assertEqual(len(ui.exception), 0)
                self.assertTrue(any('Потенциальное противоречие' in m.value for m in ui.markdown))
                self.assertTrue(any('10 млн' in t.value for t in ui.text))
                self.assertTrue(any('15 млн' in t.value for t in ui.text))
                next(b for b in ui.button if b.label == 'Сохранить заключение').click().run()
                self.assertEqual(len(ui.exception), 0)
                with store.connect() as conn:
                    self.assertEqual(conn.execute('SELECT count(*) FROM reviews').fetchone()[0], 1)
                next(b for b in ui.button if b.label == 'Открыть оригинал').click().run()
                self.assertEqual(len(ui.exception), 0)
                ui.selectbox(key='selected_project').set_value(second).run()
                self.assertFalse(any('10 млн' in t.value for t in ui.text))
                self.assertEqual(len(ui.exception), 0)


if __name__ == '__main__':
    unittest.main()
