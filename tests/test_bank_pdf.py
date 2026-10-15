"""Actual PDF + OCR ingestion, using the repository's synthetic evaluation fixtures."""
import io
from pathlib import Path
import shutil
import tempfile
import time
import unittest

from pypdf import PdfReader, PdfWriter

from document_rag.bank.api import create_app
from document_rag.bank.store import Store
from document_rag.config import ROOT


@unittest.skipUnless(shutil.which('tesseract') and shutil.which('pdftoppm'), 'Local Tesseract and Poppler required')
class BankPDFTests(unittest.TestCase):
    def test_30_text_pdfs_and_two_page_scan_async(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / 'audit.sqlite3')
            project = store.create_project('31 синтетический PDF', 'alice')['id']
            app = create_app(store, {'test-token': 'alice'}, demo_mode=True)
            self.addCleanup(app.extensions['bank_worker'].close)
            client, headers = app.test_client(), {'X-RAG-Token': 'test-token'}
            page = PdfReader(ROOT / 'evaluation/corpus/21-long-01.pdf').pages[0]
            files = []
            for index in range(30):
                writer, output = PdfWriter(), io.BytesIO()
                writer.add_page(page)
                writer.add_metadata({'/Title': f'Synthetic policy {index}'})
                writer.write(output)
                output.seek(0)
                files.append((output, f'policy-{index}.pdf'))
            files.append((io.BytesIO((ROOT / 'evaluation/corpus/41-scan-01.pdf').read_bytes()), 'scan.pdf'))
            response = client.post(f'/api/v1/projects/{project}/documents', headers=headers, data={'files': files})
            self.assertEqual(response.status_code, 202)
            self.assertEqual(len(response.json), 31)
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                docs = client.get(f'/api/v1/projects/{project}/documents', headers=headers).json
                if all(d['status'] in ('ready', 'ready_with_warnings', 'failed') for d in docs):
                    break
                time.sleep(.1)
            self.assertEqual(len(docs), 31)
            self.assertTrue(all(d['status'] in ('ready', 'ready_with_warnings') for d in docs), [(d['name'], d['status']) for d in docs])
            scan = next(d for d in docs if d['name'] == 'scan.pdf')
            self.assertEqual(scan['page_count'], 2)
            for number in (1, 2):
                result = client.get(f'/api/v1/documents/{scan["id"]}/pages/{number}', headers=headers)
                self.assertEqual(result.status_code, 200)
                self.assertGreater(len(result.json['text_exact'].strip()), 20)
                self.assertIn('ocr_unverified', result.json['quality_flags'])
                self.assertEqual(result.json['page_index'], number - 1)
            answer = client.post(f'/api/v1/projects/{project}/questions', headers=headers, json={'question': 'Какой порядок проверки?', 'async': False})
            self.assertEqual(answer.status_code, 201)
            self.assertEqual(len(answer.json['coverage_manifest']['searched_document_ids']), 31)
            self.assertEqual(answer.json['coverage_status'], 'partial')


if __name__ == '__main__':
    unittest.main()
