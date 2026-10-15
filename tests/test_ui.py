"""Actual Streamlit reruns against Flask's test client, without external API."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest
from document_rag.api import create_app
from document_rag.config import ROOT, local_token
from document_rag.engine import Engine


class UIFlowTests(unittest.TestCase):
    def test_search_chat_resume_scope_and_delete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = json.loads((ROOT / 'config.example.json').read_text())
            cfg['storage']['database'] = str(root / 'library.sqlite3')
            cfg['retrieval']['mode'] = 'lexical'
            cfg['ocr']['enabled'] = False
            config = root / 'config.json'; config.write_text(json.dumps(cfg))
            engine = Engine(cfg)
            for file in sorted((ROOT / 'sample-documents').glob('*.txt')):
                engine.ingest(file)
            class Provider:
                def generate(self, system, question):
                    return json.dumps({'status': 'answered', 'claims': [{'text': 'Срок изменился с 10 до 15 рабочих дней.', 'evidence': [
                        {'source_id': 'S1', 'quote': 'Поставка осуществляется в течение 10 рабочих дней после согласования заявки.'},
                        {'source_id': 'S2', 'quote': 'поставка осуществляется в течение 15 рабочих дней после согласования заявки.'}]}]})
            engine.provider = Provider()
            app = create_app(engine, local_token(root)); client = app.test_client()
            self.addCleanup(app.extensions['executor'].shutdown, wait=True)
            def request(method, url, **kwargs):
                result = client.open(url.split('8765')[-1], method=method, headers=kwargs.get('headers'), json=kwargs.get('json'))
                response = requests.Response(); response.status_code = result.status_code
                response._content = result.data
                return response
            with patch.dict(os.environ, {'RAG_CONFIG': str(config), 'RAG_API_URL': 'http://127.0.0.1:8765'}), patch('requests.request', side_effect=request):
                at = AppTest.from_file(str(ROOT / 'ui.py'), default_timeout=20).run()
                self.assertEqual(len(at.exception), 0)
                self.assertEqual(len(at.multiselect[0].value), 2)
                at.text_input[0].set_value('Поставка')
                next(b for b in at.button if b.label == 'Найти фрагменты').click().run()
                self.assertTrue(any('10 рабочих' in t.value for t in at.text), str(at))
                at.checkbox(key='compare').check().run()
                at.chat_input[0].set_value('Поставка: как изменился срок?').run()
                self.assertEqual(len(at.exception), 0)
                self.assertEqual(len(engine.chats()), 1)
                self.assertTrue(any('Срок изменился' in m.value for m in at.markdown))
                chat_id = engine.chats()[0]['id']
                next(b for b in at.button if b.label == 'Новый диалог').click().run()
                self.assertIsNone(at.session_state['chat_id'])
                at.button(key='resume_' + chat_id).click().run()
                self.assertEqual(at.session_state['chat_id'], chat_id)
                self.assertTrue(at.checkbox(key='compare').value)
                at.multiselect[0].set_value(at.multiselect[0].value[:1]).run()
                self.assertIsNone(at.session_state['chat_id'])
                self.assertFalse(at.checkbox(key='compare').value)
                at.button(key='remove_chat_' + chat_id).click().run()
                self.assertEqual(engine.chats(), [])
                self.assertEqual(len(at.exception), 0)


if __name__ == '__main__':
    unittest.main()
