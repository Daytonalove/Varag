import unittest
from document_rag.retrieval import references, reference_score, diversify
import test_core

class ReferenceTests(unittest.TestCase):
    def test_section_not_standard_number(self):
        self.assertEqual(references('пункт 6.13 ГОСТ 57580.2-2018'), ['6.13'])
    def test_no_prefix_match(self):
        self.assertEqual(reference_score('6.13.1 Другое требование', ['6.13']), 0)
        self.assertEqual(reference_score('16.13 Другое требование', ['6.13']), 0)
    def test_ocr_code_aliases(self):
        self.assertEqual(reference_score('УЗП.З Контроль учетных записей', references('УЗП.3')), 2)
        self.assertEqual(reference_score('MAC. 8 Централизованный сбор', references('МАС.8')), 2)
    def test_heading_over_cross_reference(self):
        self.assertGreater(reference_score('6.13 Источники свидетельств', ['6.13']),reference_score('См. пункт 6.13', ['6.13']))
    def test_diversity(self):
        rows={i:{'doc_id':'a' if i<8 else 'b'} for i in range(10)}
        self.assertIn(8,diversify(list(range(10)),rows,8,2))
        self.assertEqual(diversify(list(range(10)),rows,8,1),list(range(8)))

class CoverageTests(unittest.TestCase):
    setUp=test_core.EngineTests.setUp
    def test_missing_sources_reported(self):
        self.engine.provider=test_core.Provider(['{"status":"insufficient","claims":[]}'])
        answer=self.engine.ask('10',self.ids)
        self.assertEqual(answer['retrieval_coverage']['selected_documents'],2)
        self.assertEqual(answer['retrieval_coverage']['represented_documents'],1)
    def test_compare_fifty_rejected(self):
        ids=self.ids[:]
        for i in range(48):
            p=self.root/f'extra-{i}.txt';p.write_text('Поставка и проверка')
            ids.append(self.engine.ingest(p)['id'])
        with self.assertRaisesRegex(ValueError,'от 2 до 8'):
            self.engine.ask('Поставка',ids,compare=True)
