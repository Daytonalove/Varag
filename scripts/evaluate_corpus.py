"""Repeatable local retrieval/OCR benchmark. No model API calls, no credentials needed."""
import argparse
from contextlib import closing
import io
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from document_rag.config import load
from document_rag.engine import Engine
from document_rag.api import create_app


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--corpus',required=True,type=Path)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--database',required=True,type=Path)
    parser.add_argument('--report',required=True,type=Path)
    parser.add_argument('--reuse',action='store_true')
    args=parser.parse_args()
    if args.database.exists() and not args.reuse:raise ValueError('Choose a new database or use --reuse.')
    manifest=json.loads(args.manifest.read_text())
    for record in manifest['documents']:
        path=args.corpus/record['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Corpus hash mismatch: '+record['file'])
    cfg=load();cfg['storage']['database']=str(args.database.resolve())
    class NoProvider:
        def generate(self,*a):raise AssertionError('This benchmark must not call GigaChat')
    engine=Engine(cfg,provider=NoProvider())
    if args.reuse:
        stored={d['name']:d['revision'] for d in engine.documents()}
        if any(stored.get(d['file'])!=d['sha256'] for d in manifest['documents']):
            raise ValueError('Existing index differs from manifest. Build a new evaluation database.')
    app=create_app(engine,token='benchmark-local-token'); client=app.test_client();headers={'X-RAG-Token':'benchmark-local-token'}
    report=dict(config={'retrieval':cfg['retrieval'],'platform':platform.platform(),'python':platform.python_version()},ingestion=[],queries=[],api_errors={},live_generation_tested=False)
    # Do not leak host-specific paths into the portable report.
    report['config']['retrieval']=dict(cfg['retrieval'],model_directory='local-model')
    start=time.perf_counter()
    try:
        if not args.reuse:
            for record in manifest['documents']:
                path=args.corpus/record['file'];begin=time.perf_counter()
                response=client.post('/documents',headers=headers,data={'file':(io.BytesIO(path.read_bytes()),path.name)})
                if response.status_code!=202:
                    row=dict(file=path.name,status='failed',error=response.json)
                else:
                    job_id=response.json['job_id']
                    while True:
                        job=client.get('/jobs/'+job_id,headers=headers).json
                        if job['status'] in ('done','failed'):break
                        time.sleep(.05)
                    row=dict(file=path.name,kind=record['kind'],status=job['status'],seconds=round(time.perf_counter()-begin,3),
                             chunks=job.get('result',{}).get('chunks',0),warnings=job.get('result',{}).get('warnings',[]),error=job.get('error'))
                report['ingestion'].append(row)
                print(row['file'],row['status'],row.get('chunks'),row.get('seconds'),flush=True)
                args.report.parent.mkdir(parents=True,exist_ok=True)
                args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        report['index_seconds']=round(time.perf_counter()-start,3)
        documents=engine.documents();ids=[d['id'] for d in documents];lookup={d['name']:d['id'] for d in documents}
        with closing(engine.connect()) as conn:
            report['documents']=len(documents);report['chunks']=conn.execute('SELECT count(*) FROM chunks').fetchone()[0]
        for question in manifest['questions']:
            begin=time.perf_counter();hits=engine.search(question['question'],ids)
            found={h['document'] for h in hits};expected=set(question['expected_documents'])
            normalized=' '.join(' '.join(h['text'].split()) for h in hits)
            row=dict(id=question['id'],question=question['question'],kind=question['kind'],expected_documents=sorted(expected),
                     retrieved_documents=sorted(found),all_expected_retrieved=expected<=found if expected else None,
                     expected_recall=len(expected&found)/len(expected) if expected else None,
                     snippets_found=all(s in normalized for s in question.get('snippets',[])),
                     seconds=round(time.perf_counter()-begin,4),hits=[dict(document=h['document'],locator=h['locator'],score=h['score']) for h in hits])
            # Secondary check: explicitly scoped retrieval, as the UI permits.
            scope=[lookup[name] for name in expected if name in lookup]
            if scope:
                focused=engine.search(question['question'],scope)
                row['focused_snippets_found']=all(s in ' '.join(h['text'] for h in focused) for s in question.get('snippets',[]))
            report['queries'].append(row)
            print(question['id'],row['all_expected_retrieved'],row['snippets_found'],flush=True)
        # Bad files are additional negative cases, not counted in the 50-file corpus.
        for name,data in [('broken.pdf',b'not a PDF'),('empty.txt',b''),('unsupported.exe',b'not executable')]:
            response=client.post('/documents',headers=headers,data={'file':(io.BytesIO(data),name)})
            if response.status_code==202:
                while True:
                    job=client.get('/jobs/'+response.json['job_id'],headers=headers).json
                    if job['status'] in ('done','failed'):break
                    time.sleep(.02)
                report['api_errors'][name]=dict(status=job['status'],error=job.get('error'))
            else:report['api_errors'][name]=dict(http=response.status_code,error=response.json)
        response=client.post('/ask',headers=headers,json={'question':'Сравните все документы','document_ids':ids,'compare':True})
        report['api_errors']['compare_50']=dict(http=response.status_code,error=response.json)
        measurable=[q for q in report['queries'] if q['all_expected_retrieved'] is not None]
        report['summary']=dict(retrieval_cases=len(measurable),complete_source_coverage=sum(q['all_expected_retrieved'] for q in measurable),
            mean_expected_document_recall=statistics.mean(q['expected_recall'] for q in measurable),
            median_search_seconds=statistics.median(q['seconds'] for q in report['queries']),max_search_seconds=max(q['seconds'] for q in report['queries']),
            db_mib=round(engine.db.stat().st_size/1048576,2),peak_process_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1048576 if sys.platform=='darwin' else 1024),1))
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps(report['summary'],ensure_ascii=False),flush=True)
    finally:app.extensions['executor'].shutdown(wait=True)


if __name__=='__main__':main()
