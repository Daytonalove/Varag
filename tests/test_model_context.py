"""Model switch, multi-document coverage, token-counter failure and budget tests."""
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from document_rag.bank.context import limits
from document_rag.bank.fast_generation import _select_sources
from document_rag.bank.generation import GigaChatRAG
from document_rag.bank.answering import AnswerService
from document_rag.bank.retrieval import Retriever
from document_rag.bank.store import Store
from document_rag.provider import ProviderError


def sources(count, repeat=1):
    return [dict(block_id=f'b{i}',document_id=f'd{i}',document_version_id=f'v{i}',
                 document_name=f'Правило {i}.txt',page_number='1',page_index=0,section_path='',
                 text_exact=('Мониторинг обеспечивает наблюдение за состоянием системы. '*repeat),
                 quality_flags=[],reasons=[]) for i in range(count)]


def completion():
    return {'model':'test','choices':[{'finish_reason':'stop','message':{'content':json.dumps({
        'claims':[{'text':'Мониторинг обеспечивает наблюдение за состоянием системы.','source_ids':['Q1']}],
        'limitations':[]},ensure_ascii=False)}}]}


class ModelContextTests(unittest.TestCase):
    def test_model_profiles_version_ids_and_admin_caps(self):
        for name in ('GigaChat-2','GigaChat-2-Pro','GigaChat-2-Max','GigaChat-2:2.0.30.01'):
            self.assertEqual(limits({'model':name})['context_tokens'],128000)
        self.assertEqual(limits({'model':'unknown'})['context_tokens'],32768)
        self.assertEqual(limits({'model':'GigaChat-2','context_window_tokens':999999})['context_tokens'],128000)
        self.assertEqual(limits({'model':'GigaChat-2','context_window_tokens':16000})['context_tokens'],16000)
        self.assertEqual(limits({'model':'special','model_context_limits':{'special':16000}})['context_tokens'],16000)

    def test_fifty_documents_are_not_cut_to_eight(self):
        original=sources(50,40)
        # A single document contributes many top-ranked passages.
        duplicates=[dict(original[0],block_id=f'duplicate{i}') for i in range(50)]
        chosen,payload=_select_sources(duplicates+original,'Перечисли требования документов',{'model':'GigaChat-2'})
        self.assertEqual({s['document_id'] for s in chosen},{s['document_id'] for s in original})
        self.assertGreater(sum(len(s['text']) for s in payload),10000)
        self.assertLessEqual(len(chosen),64)

    def test_provider_counted_context_and_one_completion(self):
        provider=GigaChatRAG({'model':'GigaChat-2','count_tokens':True})
        def call(path,body=None,**kwargs):
            if path=='/tokens/count':
                self.assertEqual(body['model'],'GigaChat-2')
                self.assertEqual(kwargs['timeout_seconds'],15)
                return [{'tokens':max(1,len(t)//3)} for t in body['input']]
            self.assertEqual(path,'/chat/completions')
            self.assertEqual(len(json.loads(body['messages'][1]['content'])['sources']),50)
            return completion()
        with patch.object(provider,'call',side_effect=call) as mocked:
            result=provider.analyze('Какие требования?',sources(50,40))
        self.assertEqual(mocked.call_count,2)
        self.assertEqual(result['context']['token_count_method'],'provider_components')
        self.assertLess(result['context']['input_tokens_estimate']+result['context']['output_tokens'],128000)
        self.assertEqual(len(result['sent_sources']),50)

    def test_counter_failure_uses_safe_bound_and_still_answers(self):
        provider=GigaChatRAG({'model':'GigaChat-2','count_tokens':True})
        def call(path,body=None,**kwargs):
            if path=='/tokens/count':raise ProviderError('unavailable',503)
            return completion()
        with patch.object(provider,'call',side_effect=call) as mocked:
            result=provider.analyze('Мониторинг?',sources(50,40))
        self.assertEqual(mocked.call_count,2)
        context=result['context']
        self.assertEqual(context['token_count_method'],'utf8_upper_bound')
        self.assertEqual(context['token_count_failure'],'http_503')
        self.assertLessEqual(context['input_tokens_estimate']+context['output_tokens'],128000)
        self.assertTrue(context['omitted_source_ids'])

    def test_invalid_counts_cannot_bypass_budget(self):
        provider=GigaChatRAG({'model':'test','count_tokens':True,'context_window_tokens':500})
        for bad in ([],[{'tokens':-1}],[{'tokens':True}],{'tokens':1}):
            with patch.object(provider,'call',return_value=bad) as call:
                with self.assertRaises(ProviderError) as error:provider.analyze('Мониторинг?',sources(1))
            self.assertEqual(error.exception.code,'context_budget')
            self.assertEqual([c.args[0] for c in call.call_args_list],['/tokens/count'])

    def test_counter_401_does_not_retry_or_generate(self):
        provider=GigaChatRAG({'model':'GigaChat-2','count_tokens':True})
        with patch.object(provider,'call',side_effect=ProviderError('unauthorized',401)) as call:
            with self.assertRaises(ProviderError):provider.analyze('Мониторинг?',sources(1))
        self.assertEqual(call.call_count,1)

    def test_policy_is_rechecked_after_counting_before_generation(self):
        provider=GigaChatRAG({'model':'GigaChat-2','count_tokens':True})
        calls=[]
        def call(path,body=None,**kwargs):
            calls.append(path)
            return [{'tokens':10} for _ in body['input']]
        def policy(*args):
            if calls:raise ValueError('Corpus changed')
        with patch.object(provider,'call',side_effect=call):
            with self.assertRaises(ValueError):provider.analyze('Мониторинг?',sources(1),before_call=policy)
        self.assertEqual(calls,['/tokens/count'])

    def test_small_model_skips_oversize_source_and_relabels_citations(self):
        provider=GigaChatRAG({'model':'small','model_context_limits':{'small':8192},'count_tokens':True})
        items=sources(3,40)
        def call(path,body=None,**kwargs):
            if path=='/tokens/count':return [{'tokens':100},{'tokens':2600},{'tokens':2600},{'tokens':100}]
            texts=json.loads(body['messages'][1]['content'])['sources']
            self.assertEqual([s['document_version_id'] for s in texts],['v0','v2'])
            self.assertEqual([s['source_id'] for s in texts],['Q1','Q2'])
            return completion()
        with patch.object(provider,'call',side_effect=call):result=provider.analyze('Мониторинг?',items)
        self.assertEqual(result['context']['omitted_source_ids'],['b1'])

    def test_request_model_switch_does_not_mutate_shared_retriever(self):
        with tempfile.TemporaryDirectory() as temp:
            store=Store(Path(temp)/'db.sqlite3')
            project=store.create_project('Models','tester')['id']
            provider=GigaChatRAG({'model':'GigaChat-2','count_tokens':False,
                                 'model_context_limits':{'GigaChat-small':8192}})
            seen=[]
            retriever=Retriever(store)
            original_search=Retriever.search
            def search(instance,*args,**kwargs):
                seen.append(instance.budget)
                return original_search(instance,*args,**kwargs)
            service=AnswerService(store,provider,retriever=retriever,automatic_external=True)
            with patch.object(provider,'models',return_value=['GigaChat-2','GigaChat-small']),patch.object(Retriever,'search',search):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results=list(pool.map(lambda m:service.ask(project,'tester','Мониторинг?',True,model_id=m),['GigaChat-2','GigaChat-small']))
            self.assertEqual(sorted(seen),[49152,480000])
            self.assertEqual(retriever.budget,16000)
            self.assertEqual(provider.cfg['model'],'GigaChat-2')
            self.assertEqual(len(results),2)

    def test_model_limits_api_matches_available_model_selection(self):
        from document_rag.bank.api import create_app
        with tempfile.TemporaryDirectory() as temp:
            store=Store(Path(temp)/'db.sqlite3')
            provider=GigaChatRAG({'model':'GigaChat-2','model_context_limits':{'GigaChat-small':8192}})
            service=AnswerService(store,provider,automatic_external=True)
            app=create_app(store,{'test-token':'tester'},False,service,True)
            with patch.object(provider,'models',return_value=['GigaChat-2','GigaChat-small','Embeddings']):
                result=app.test_client().get('/api/v1/provider/models',headers={'X-RAG-Token':'test-token'})
            self.assertEqual(result.status_code,200)
            self.assertEqual(result.json['limits']['GigaChat-small']['context_tokens'],8192)
            self.assertEqual(result.json['limits']['GigaChat-2']['context_tokens'],128000)
            self.assertNotIn('Embeddings',result.json['limits'])


if __name__=='__main__':unittest.main()
