"""Regressions for exact identifiers, fast evidence, budgets and provider failures."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from document_rag.bank.store import Store, NotFound
from document_rag.bank.retrieval import Retriever
from document_rag.bank.reranking import RankScore
from document_rag.bank.generation import GigaChatRAG
from document_rag.bank.fast_generation import analyze_fast, _select_sources, _guard_claims
from document_rag.bank.answering import AnswerService, _search_query_with_history
from document_rag.bank.structural import codes
from document_rag.provider import ProviderError, GigaChat


def source(text,block='b',document='d'):
    return dict(block_id=block,document_id=document,document_version_id=document+'v',document_name=document+'.txt',
                section_path='',page_number='1',page_index=0,text_exact=text,reasons=[],quality_flags=[])


def response(claims=None,finish='stop'):
    return {'model':'test','usage':{'prompt_tokens':100,'completion_tokens':20},'choices':[{
        'finish_reason':finish,'message':{'content':json.dumps({'claims':claims or [],'limitations':[]},ensure_ascii=False)}}]}


class PrecisionTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.path=Path(temp.name)/'db.sqlite3';self.store=Store(self.path)
        self.project=self.store.create_project('Regression','tester')['id']

    def upload(self,text,name='standard.txt',document=None):
        result=self.store.upload(self.project,'tester',name,text.encode(),document)
        self.store.process_one()
        return result

    def test_ocr_codes_and_independent_exact_lanes_survive_bad_reranking(self):
        self.assertEqual(codes('Найди CMЭ.З и ЗСВ.5, а не СМЭ.30.'),['СМЭ.3','ЗСВ.5','СМЭ.30'])
        self.upload('СМЭ.З Межсетевое экранирование вычислительных сетей.\n\nЗСВ.5 Идентификация пользователей виртуальных машин.')
        self.upload('РСБ.1 Регистрация событий.\n'*80,'distractor.txt')
        class BadReranker:
            model_id='test'
            def score(self,q,texts):
                return [RankScore(-100 if codes(t) and any(c in codes(t) for c in ('СМЭ.3','ЗСВ.5')) else 100,0,len(t)) for t in texts]
        result=Retriever(self.store,reranker=BadReranker()).search(self.project,'tester','Найди СМЭ.3 и ЗСВ.5',top_k=2)
        self.assertEqual(result['coverage_manifest']['missing_exact_codes'],[])
        self.assertEqual(set(c for s in result['sources'] for c in codes(s['text_exact'])),{'СМЭ.3','ЗСВ.5'})
        chosen,_=_select_sources(result['sources'],'Найди СМЭ.3 и ЗСВ.5')
        self.assertEqual(set(c for s in chosen for c in codes(s['text_exact'])),{'СМЭ.3','ЗСВ.5'})

    def test_code_at_end_of_long_block_is_not_lost(self):
        self.upload('Вводные пояснения. '*1000+'\nСМЭ.3 Межсетевое экранирование сетей.')
        result=Retriever(self.store).search(self.project,'tester','СМЭ.3')
        chosen,payload=_select_sources(result['sources'],'СМЭ.3')
        self.assertTrue(any('СМЭ.3' in s['text'] for s in payload))
        self.assertEqual(chosen[0]['text_exact'],payload[0]['text'])

    def test_source_budget_reserves_each_code_before_duplicate_versions(self):
        sources=[source('СМЭ.3 Межсетевое экранирование сетей.',str(i),str(i)) for i in range(10)]
        sources.append(source('ЗСВ.5 Идентификация виртуальных машин.','last','last'))
        with patch.dict('os.environ',{'FAST_RAG_MAX_SOURCES':'2'}):
            chosen,_=_select_sources(sources,'СМЭ.3 и ЗСВ.5')
        self.assertEqual(set(c for s in chosen for c in codes(s['text_exact'])),{'СМЭ.3','ЗСВ.5'})

    def test_versions_and_acl_remain_bound_to_exact_codes(self):
        first=self.upload('СМЭ.3 Исходная редакция защиты.')
        second=self.upload('СМЭ.3 Новая редакция защиты.',document=first['document_id'])
        current=Retriever(self.store).search(self.project,'tester','СМЭ.3')
        self.assertEqual({s['document_version_id'] for s in current['sources']},{second['version_id']})
        historical=Retriever(self.store).search(self.project,'tester','СМЭ.3',as_of='2025-01-01')
        self.assertEqual({s['document_version_id'] for s in historical['sources']},{first['version_id'],second['version_id']})
        with self.store.connect() as c:c.execute('UPDATE documents SET restricted=1')
        self.assertEqual(Retriever(self.store).search(self.project,'tester','СМЭ.3')['sources'],[])
        with self.assertRaises(NotFound):Retriever(self.store).search(self.project,'other','СМЭ.3')

    def test_missing_code_does_not_match_prefix(self):
        self.upload('СМЭ.30 Правило другой меры.')
        result=Retriever(self.store).search(self.project,'tester','СМЭ.3')
        self.assertEqual(result['coverage_manifest']['missing_exact_codes'],['СМЭ.3'])
        guarded,limits=_guard_claims([],result['sources'],'СМЭ.3')
        self.assertEqual(guarded,[])
        self.assertIn('не означает отсутствия',limits[0])

    def test_reference_is_not_a_definition_and_wrong_title_cannot_replace_row(self):
        s=source('СМЭ.3 Межсетевое экранирование сетей.\nРСБ.1 Регистрация событий безопасности.')
        claims=[{'text':'СМЭ.3 — регистрация событий.','evidence':[{'block_id':'b','quote':s['text_exact']}]}]
        checked,_=_guard_claims(claims,[s],'Найди СМЭ.3')
        self.assertIn('экранирование',checked[0]['text'])
        self.assertNotIn('Регистрация',checked[0]['text'])
        ref=source('Для регистрации событий см. СМЭ.3 в другом разделе.')
        self.assertEqual(_guard_claims(claims,[ref],'Найди СМЭ.3')[0],[])

    def test_fast_one_call_keeps_short_definitions_and_exact_sent_text(self):
        provider=GigaChatRAG({'model':'test'})
        sources=[source('Мониторинг — наблюдение за состоянием системы.')]
        raw=response([{'text':'Мониторинг — наблюдение за состоянием системы.','source_ids':['Q1']}])
        with patch.object(provider,'call',return_value=raw) as call:
            result=provider.analyze('Что такое мониторинг?',sources)
        self.assertEqual(call.call_count,1)
        sent=json.loads(call.call_args.args[1]['messages'][1]['content'])['sources'][0]['text']
        self.assertEqual(result['claims'][0]['evidence'][0]['quote'],sent)
        self.assertEqual(result['sent_sources'][0]['text_exact'],sent)
        self.assertEqual(result['verification_level'],'local_source_checks')

    def test_finish_reasons_timeout_and_no_hidden_retry(self):
        sources=[source('Мониторинг — наблюдение за состоянием системы.')]
        for finish in ('blacklist','length'):
            provider=GigaChatRAG({'model':'test'})
            with patch.object(provider,'call',return_value=response(finish=finish)) as call:
                with self.assertRaises(ProviderError) as error:analyze_fast(provider,'Мониторинг?',sources)
                self.assertEqual(error.exception.code,'finish_'+finish)
                self.assertEqual(call.call_count,1)
        provider=GigaChat({'model':'test','base_url':'https://example.invalid','retry_unauthorized':False})
        with patch.object(provider,'token',return_value='token'),patch.object(provider,'request',side_effect=ProviderError('denied',401)) as request:
            with self.assertRaises(ProviderError):provider.call('/chat/completions',{})
            self.assertEqual(request.call_count,1)

    def test_failure_has_no_candidate_citations_and_retains_diagnostics(self):
        self.upload('Мониторинг — наблюдение за состоянием системы.')
        provider=GigaChatRAG({'model':'test'})
        with patch.object(provider,'call',side_effect=ProviderError('Время ожидания истекло.',code='timeout')):
            answer=AnswerService(self.store,provider,automatic_external=True).ask(self.project,'tester','Что такое мониторинг?',True)
        self.assertEqual(answer['sources'],[])
        self.assertEqual(answer['evidence'],[])
        self.assertEqual(answer['failure']['code'],'timeout')
        self.assertTrue(answer['coverage_manifest']['retrieved_source_ids'])

    def test_budget_and_eight_item_list(self):
        source_text='\n'.join(f'{i}. Процесс защиты номер {i} предусматривает контроль ресурсов.' for i in range(1,9))
        provider=GigaChatRAG({'model':'test'})
        claims=[{'text':line,'source_ids':['Q1']} for line in source_text.splitlines()]
        with patch.object(provider,'call',return_value=response(claims)) as call:
            result=provider.analyze('Перечисли все восемь процессов',[source(source_text)])
        self.assertEqual(len(result['claims']),8)
        self.assertEqual(call.call_count,1)
        provider=GigaChatRAG({'model':'test','context_window_tokens':500})
        with patch.object(provider,'call') as call:
            with self.assertRaises(ProviderError):provider.analyze('Перечисли процессы',[source(source_text)])
            call.assert_not_called()

    def test_history_recheck_retains_topic_but_new_code_overrides_old(self):
        history=[{'question':'Найди СМЭ.3','answer_text':'Неверный ответ РСБ.1.'},{'question':'Проверь заново'}]
        self.assertEqual(_search_query_with_history('Ты ошибся, проверь',history),'Найди СМЭ.3')
        self.assertEqual(_search_query_with_history('Проверь ЗСВ.5',history),'Проверь ЗСВ.5')
        self.assertNotIn('РСБ.1',_search_query_with_history('А во втором документе?',history))
        provider=GigaChatRAG({'model':'test'})
        with patch.object(provider,'call',return_value=response()) as call:
            result=provider.analyze('Проверь ЗСВ.5',[source('ЗСВ.5 Идентификация виртуальных машин.')],history=history)
        self.assertIn('ЗСВ.5',result['claims'][0]['text'])
        self.assertNotIn('СМЭ.3',result['claims'][0]['text'])
        self.assertFalse(call.call_args.kwargs['retry_unauthorized'])

    def test_bad_numeric_claim_does_not_remove_independent_valid_claim(self):
        s=source('Срок хранения составляет 30 дней. Мониторинг обеспечивает наблюдение за системой.')
        claims=[{'text':t,'evidence':[{'block_id':'b','quote':s['text_exact']}]} for t in
                ('Срок хранения составляет 90 дней.','Мониторинг обеспечивает наблюдение за системой.')]
        accepted,limits=_guard_claims(claims,[s],'Какие требования?')
        self.assertEqual([c['text'] for c in accepted],[claims[1]['text']])
        self.assertTrue(limits)

    def test_explicit_standard_keeps_same_code_in_its_document(self):
        a=self.upload('ГОСТ Р 20210.1\nСМЭ.3 Межсетевое экранирование сетей.','Стандарт 20210.1.txt')
        self.upload('ГОСТ Р 20210.2\nСМЭ.3 Регистрация событий безопасности.','Стандарт 20210.2.txt')
        self.upload('ГОСТ Р 20210.10\nСМЭ.3 Регистрация чужого стандарта.','Стандарт 20210.10.txt')
        result=Retriever(self.store).search(self.project,'tester','ГОСТ 20210.1: СМЭ.3')
        self.assertEqual({s['document_version_id'] for s in result['sources']},{a['version_id']})

    def test_fast_failure_updates_shared_circuit_breaker(self):
        provider=GigaChatRAG({'model':'test'})
        with patch.object(provider,'call',side_effect=ProviderError('Timeout',code='timeout')) as call:
            for _ in range(4):
                with self.assertRaises(ProviderError):provider.analyze('Мониторинг?',[source('Мониторинг обеспечивает наблюдение за системой.')])
        self.assertEqual(call.call_count,3)
        self.assertEqual(provider.failures,3)

    def test_partial_measure_row_is_not_presented_as_complete_definition(self):
        s=source('СМЭ.3 Начало длинной строки с требованием')
        s.update(full_block_truncated=True,excerpt_range=[0,len(s['text_exact'])],full_block_length=9000)
        self.assertEqual(_select_sources([s],'СМЭ.3')[0],[])
        self.assertEqual(_guard_claims([],[s],'СМЭ.3')[0],[])
        complete=source('СМЭ.3 Межсетевое экранирование сетей.\nРСБ.1 Начало следующей строки')
        complete.update(full_block_truncated=True,excerpt_range=[0,len(complete['text_exact'])],full_block_length=9000)
        chosen,_=_select_sources([complete],'СМЭ.3')
        self.assertIn('экранирование',_guard_claims([],chosen,'СМЭ.3')[0][0]['text'])

    def test_existing_canonical_blocks_receive_structural_index(self):
        self.upload('УЗП.21 Разделение полномочий пользователей.')
        with self.store.connect() as conn:
            conn.execute('DELETE FROM structural_keys')
            conn.execute("UPDATE derived_versions SET normalizer_version='old'")
        reopened=Store(self.path)
        with reopened.connect() as conn:
            self.assertEqual(conn.execute("SELECT code FROM structural_keys").fetchone()[0],'УЗП.21')

    def test_geometry_cells_require_headers_and_retain_coordinates(self):
        from document_rag.bank.tables import measure_tables
        words=[dict(text=t,box=[x,y,x+20,y+10]) for t,x,y in [
            ('1',300,10),('2',350,10),('3',400,10),
            ('СМЭ.3',10,40),('Экранирование',100,40),('Н',300,40),('т',350,40),('т',400,40),
            ('ЗСВ.5',10,70),('Идентификация',100,70),('т',300,70),('т',350,70),('т',400,70)]]
        tables=measure_tables(words)
        self.assertEqual([r['code'] for r in tables],['СМЭ.3','ЗСВ.5'])
        self.assertEqual([(c['header'],c['text']) for c in tables[0]['cells']],[('1','Н'),('2','т'),('3','т')])
        self.assertTrue(tables[0]['needs_review'])
        self.assertEqual(measure_tables(words[3:]),[])

    def test_real_pdf_and_ocr_preserve_words_and_table_cells(self):
        import io
        from reportlab.pdfgen.canvas import Canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from document_rag.bank.ingest import extract_pages
        from PIL import Image,ImageDraw,ImageFont
        font=next(p for p in [Path('/System/Library/Fonts/Supplemental/Arial.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')] if p.exists())
        pdfmetrics.registerFont(TTFont('Precision',str(font)))
        output=io.BytesIO();pdf=Canvas(output,pagesize=(600,300));pdf.setFont('Precision',12)
        entries=[('Уровни защиты',300,30),('1',350,60),('2',420,60),('3',490,60),
                 ('СМЭ.3',30,100),('Экранирование',140,100),('Н',350,100),('т',420,100),('т',490,100),
                 ('ЗСВ.5',30,150),('Идентификация',140,150),('т',350,150),('т',420,150),('т',490,150)]
        for text,x,y in entries:pdf.drawString(x,300-y,text)
        pdf.showPage();pdf.setFont('Precision',12)
        pdf.drawString(30,200,'СМЭ.3 Требование второй страницы.')
        pdf.save()
        file=self.path.parent/'table.pdf';file.write_bytes(output.getvalue())
        pages=extract_pages(file)
        self.assertTrue(pages[0]['layout_data']['words'])
        self.assertEqual(pages[0]['layout_data']['tables'][0]['cells'][0]['text'],'Н')
        image=Image.new('RGB',(1800,900),'white');draw=ImageDraw.Draw(image);imagefont=ImageFont.truetype(str(font),36)
        for text,x,y in entries:draw.text((x*3,y*3),text,font=imagefont,fill='black')
        scan=self.path.parent/'table.png';image.save(scan)
        ocr=extract_pages(scan)[0]
        self.assertEqual(ocr['extraction_method'],'tesseract')
        self.assertTrue(ocr['layout_data']['words'])
        self.assertIn('СМЭ.3',codes(ocr['text_exact']))
        uploaded=self.store.upload(self.project,'tester','table.pdf',file.read_bytes())
        self.store.process_one()
        with self.store.connect() as conn:
            saved=json.loads(conn.execute('SELECT data FROM page_layouts WHERE version_id=?',(uploaded['version_id'],)).fetchone()[0])
        self.assertTrue(saved['tables'])
        result=Retriever(self.store).search(self.project,'tester','СМЭ.3 на странице 2')
        self.assertTrue(result['sources'])
        self.assertEqual({s['page_index'] for s in result['sources']},{1})


if __name__=='__main__':unittest.main()
