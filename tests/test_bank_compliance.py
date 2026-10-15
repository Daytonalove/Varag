"""Regressions for defects recorded in the requirements audit."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,MagicMock

from document_rag.bank.api import create_app
from document_rag.bank.answering import AnswerService
from document_rag.bank.store import Store,NotFound
from document_rag.bank.retrieval import Retriever
from document_rag.bank.operations import encrypted_backup,restore_encrypted_backup,scan_upload
from document_rag.bank.ingest import extract_pages,structure
from document_rag.bank.norms import compare,facts


class ComplianceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.store=Store(self.root/'db.sqlite3')
        self.project=self.store.create_project('Проверка','owner')['id']
        self.app=create_app(self.store,{'owner-token':'owner','reader-token':'reader'},False,demo_mode=True)
        self.client=self.app.test_client()
        self.headers={'X-RAG-Token':'owner-token'}

    def upload(self,text,name='norm.txt',document=None):
        result=self.store.upload(self.project,'owner',name,text.encode(),document)
        self.store.process_one()
        return result

    def request(self,method,path,**kwargs):
        return self.client.open('/api/v1'+path,method=method,headers=self.headers,**kwargs)

    def test_original_audit_probes(self):
        from scripts.audit_bank_compliance import local_checks
        results=local_checks()
        self.assertEqual([r['id'] for r in results if not r['passed']],[])

    def test_pdf_unicode_long_source_and_invalid_format(self):
        self.upload('4.2 Лимит кредитного комитета составляет 10 млн руб.\n'+'Сверьте с оригиналом документа. '*300)
        answer=AnswerService(self.store).ask(self.project,'owner','Лимит кредитного комитета?')
        response=self.request('GET','/answers/'+answer['id']+'/export?format=pdf')
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.mimetype,'application/pdf')
        from pypdf import PdfReader
        reader=PdfReader(io.BytesIO(response.data))
        self.assertGreater(len(reader.pages),1)
        self.assertIn('Лимит кредитного комитета',''.join(p.extract_text() for p in reader.pages))
        self.assertEqual(self.request('GET','/answers/'+answer['id']+'/export?format=docx').status_code,400)

    def test_registry_filters_claim_index_and_lexical_rebuild(self):
        self.store.claims_enabled=True
        uploaded=self.upload('4.2 С 01.07.2026 лимит составляет 10 млн руб.')
        doc=uploaded['document_id']
        result=self.request('PATCH','/documents/'+doc+'/metadata',json={'tags':['проверен'],'product':'МСБ','verification_status':'confirmed'})
        self.assertEqual(result.status_code,200)
        card=self.request('GET','/documents/'+doc+'/registry').json
        self.assertEqual(card['proposal']['effective_from'],'2026-07-01')
        found=Retriever(self.store).search(self.project,'owner','пункт 4.2',filters={'product':'МСБ','tags':['проверен']})
        self.assertTrue(found['sources'])
        self.assertIn('exact_index',found['sources'][0]['reasons'])
        self.assertNotIn('claims_index_disabled',found['coverage_manifest']['limitations'])
        self.assertFalse(Retriever(self.store).search(self.project,'owner','Лимит',filters={'product':'Ипотека'})['sources'])
        with self.store.connect() as conn:
            original=conn.execute('SELECT id,text_exact FROM blocks').fetchall()
            conn.execute('DELETE FROM block_fts')
        self.store.queue_reindex(doc,'owner')
        self.store.process_maintenance()
        with self.store.connect() as conn:
            self.assertEqual([tuple(r) for r in original],[tuple(r) for r in conn.execute('SELECT id,text_exact FROM blocks')])
            self.assertGreater(conn.execute('SELECT count(*) FROM block_fts').fetchone()[0],0)
            self.assertGreater(conn.execute('SELECT count(*) FROM claims').fetchone()[0],0)

    def test_relations_resolve_explicit_target_and_cannot_cross_project(self):
        target=self.upload('4.2 Специальная норма действует для филиала.','policy.txt')
        origin=self.upload('С 01.07.2026 изменяется пункт 4.2 положения.','change.txt')
        with self.store.connect() as conn:
            block=conn.execute('SELECT id FROM blocks WHERE version_id=?',(origin['version_id'],)).fetchone()[0]
        payload=dict(source_document_id=origin['document_id'],target_document_id=target['document_id'],kind='amends',source_block_id=block,effective_from='2026-07-01',confirmed=True)
        self.assertEqual(self.request('POST','/relations',json=payload).status_code,201)
        found=Retriever(self.store).search(self.project,'owner','поправки')
        self.assertEqual({s['document_id'] for s in found['sources']},{target['document_id'],origin['document_id']})
        other=self.store.create_project('Другой','owner')['id']
        wrong=self.store.upload(other,'owner','other.txt',b'other')
        payload['target_document_id']=wrong['document_id']
        self.assertEqual(self.request('POST','/relations',json=payload).status_code,400)

    def test_member_acl_exports_and_last_owner(self):
        doc=self.upload('Лимит 10 млн руб.')['document_id']
        answer=AnswerService(self.store).ask(self.project,'owner','Лимит?')
        route='/projects/'+self.project+'/members/'
        self.assertEqual(self.request('PUT',route+'reader',json={'role':'viewer'}).status_code,200)
        reader={'X-RAG-Token':'reader-token'}
        self.assertEqual(self.client.get('/api/v1/answers/'+answer['id']+'/export',headers=reader).status_code,200)
        self.assertEqual(self.request('PUT','/documents/'+doc+'/acl',json={'restricted':True,'user_ids':['owner']}).status_code,200)
        self.assertEqual(self.client.get('/api/v1/answers/'+answer['id']+'/export',headers=reader).status_code,404)
        self.assertEqual(self.request('PUT',route+'owner',json={'role':None}).status_code,400)
        self.assertEqual(self.client.put('/api/v1'+route+'reader',headers=reader,json={'role':'owner'}).status_code,404)

    def test_antivirus_rejects_outage_and_infection(self):
        with patch.dict(os.environ,{'BANK_REQUIRE_AV':'true','BANK_CLAMAV_HOST':''}):
            with self.assertRaises(ValueError):
                scan_upload(b'data')
        with patch.dict(os.environ,{'BANK_CLAMAV_HOST':'localhost'}):
            with patch('socket.create_connection',side_effect=OSError):
                with self.assertRaises(ValueError): scan_upload(b'data')
            connection=MagicMock()
            connection.__enter__.return_value.recv.return_value=b'stream: Eicar-Test-Signature FOUND\0'
            with patch('socket.create_connection',return_value=connection):
                with self.assertRaises(ValueError): scan_upload(b'data')
            connection.__enter__.return_value.recv.return_value=b'stream: OK\0'
            with patch('socket.create_connection',return_value=connection):
                self.assertEqual(scan_upload(b'data'),'clean')

    def test_encrypted_backup_integrity_and_tampering(self):
        self.upload('Конфиденциальная тестовая норма')
        key=self.root/'key'
        key.write_bytes(os.urandom(32))
        backup=self.root/'backup.encrypted'
        encrypted_backup(self.store,backup,key)
        self.assertNotIn('Конфиденциальная'.encode(),backup.read_bytes())
        restored=self.root/'restored.sqlite3'
        restore_encrypted_backup(backup,restored,key)
        self.assertEqual(len(Store(restored).documents(self.project,'owner')),1)
        value=bytearray(backup.read_bytes());value[-1]^=1;backup.write_bytes(value)
        from cryptography.exceptions import InvalidTag
        with self.assertRaises(InvalidTag): restore_encrypted_backup(backup,self.root/'bad',key)
        self.assertFalse((self.root/'bad').exists())

    def test_pdf_layout_coordinates_and_tiff_frame_preview(self):
        from reportlab.pdfgen.canvas import Canvas
        path=self.root/'test.pdf'
        canvas=Canvas(str(path));canvas.drawString(80,720,'4.2 Credit committee transaction limit is 10 million RUB.');canvas.save()
        pages=extract_pages(path)
        blocks=structure(pages,'version')
        self.assertTrue(blocks[0]['bbox']['rects'])
        self.assertEqual(''.join(b['text_exact'] for b in blocks),pages[0]['text_exact'])
        from document_rag.bank.preview import render_page
        from PIL import Image
        original=io.BytesIO();Image.new('RGB',(100,100),'red').save(original,format='TIFF',save_all=True,append_images=[Image.new('RGB',(100,100),'blue')])
        frame=Image.open(io.BytesIO(render_page('scan.tiff',original.getvalue(),2)))
        self.assertEqual(frame.getpixel((10,10)),(0,0,255))

    def test_calendar_year_is_not_duration_and_percent_is_compared(self):
        self.assertEqual(facts('В январе 2026 года'),[])
        sources=[dict(block_id=str(i),document_id=str(i),text_exact=t,quality_flags=[]) for i,t in enumerate([
            'Процентная ставка кредита составляет 5 процентов.','Процентная ставка кредита составляет 10 процентов.'])]
        self.assertEqual(compare(sources)[0]['status'],'potential_unresolved_conflict')

    def test_written_dates_have_same_numeric_identity(self):
        from document_rag.bank.generation import numeric_facts
        self.assertEqual(numeric_facts('На 1 марта 2026 года'),numeric_facts('На 2026-03-01'))

    def test_unapproved_model_and_token_quota_block_publication(self):
        from document_rag.bank.generation import GigaChatRAG
        source=dict(block_id='b',document_name='norm',page_number='1',section_path='1',text_exact='Срок составляет 5 рабочих дней.')
        provider=GigaChatRAG({'model':'test','question_token_budget':1})
        with patch.object(provider,'call') as call:
            result=provider.analyze('Какой срок?',[source])
        self.assertFalse(result['claims'])
        self.assertFalse(call.called)
        provider=GigaChatRAG({'model':'test','allowed_model_versions':['approved']})
        response={'model':'changed','choices':[{'finish_reason':'stop','message':{'content':'{}'}}]}
        with patch.object(provider,'call',return_value=response):
            result=provider.analyze('Какой срок?',[source])
        self.assertFalse(result['claims'])
        self.assertTrue(any('Версия модели изменилась' in e for e in result['errors']))

    def test_retention_is_dry_by_default_and_removes_derived_copies(self):
        from document_rag.bank.operations import retention
        doc=self.upload('Лимит 10 млн руб.')['document_id']
        answer=AnswerService(self.store).ask(self.project,'owner','Лимит?')
        self.store.delete(doc,'owner')
        with self.store.connect() as conn:
            conn.execute("INSERT INTO audit_events(project_id,user_id,action,target_id,detail,created) VALUES(?,?,?,?,?,?)",(self.project,'owner','document.deleted',doc,'{}','2000-01-01 00:00:00'))
        self.assertEqual(retention(self.store,self.project,'owner')['document_ids'],[doc])
        retention(self.store,self.project,'owner',execute=True)
        with self.store.connect() as conn:
            for table in ('blobs','blocks','versions','answers','block_fts','exact_terms','claims'):
                self.assertEqual(conn.execute('SELECT count(*) FROM '+table).fetchone()[0],0)
            self.assertGreater(conn.execute('SELECT count(*) FROM audit_events').fetchone()[0],0)


if __name__=='__main__':
    unittest.main()
