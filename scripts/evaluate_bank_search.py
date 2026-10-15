"""Offline retrieval evaluation on synthetic, document-level relevance labels."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from document_rag.config import ROOT
from document_rag.bank.embeddings import local_encoder
from document_rag.bank.reranking import LocalCrossEncoder
from document_rag.bank.retrieval import Retriever
from document_rag.bank.store import Store


def metrics(ranking, relevant):
    relevant = set(relevant)
    ranks = [i for i, doc in enumerate(ranking, 1) if doc in relevant]
    dcg = sum(1 / math.log2(i + 1) for i in ranks if i <= 10)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(10, len(relevant)) + 1))
    return {'recall_at_5': len(set(ranking[:5]) & relevant) / len(relevant),
            'mrr_at_10': 1 / ranks[0] if ranks and ranks[0] <= 10 else 0,
            'ndcg_at_10': dcg / ideal}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'evaluation/results/bank-search-reranked.json')
    args = parser.parse_args()
    dataset = json.loads((ROOT / 'fixtures/search_quality.json').read_text())
    encoder = local_encoder(ROOT / 'models/multilingual-e5-small')
    reranker = LocalCrossEncoder(ROOT / 'models/mmarco-mMiniLMv2-L12-H384-v1')
    results = []
    with tempfile.TemporaryDirectory(prefix='bank-search-eval-') as directory:
        store = Store(Path(directory) / 'eval.sqlite3', encoder)
        project = store.create_project('Синтетический поиск', 'evaluator')['id']
        ids = {}
        for doc in dataset['documents']:
            uploaded = store.upload(project, 'evaluator', doc['id'] + '.txt', doc['text'].encode())
            ids[uploaded['document_id']] = doc['id']
            store.process_one()
        if any(d['semantic_status'] != 'ready' for d in store.documents(project, 'evaluator')):
            raise RuntimeError('Semantic indexing failed')
        for index, case in enumerate(dataset['questions'], 1):
            row = dict(query=case['query'], relevant=case['relevant'])
            for mode, model in [('hybrid', None), ('reranked', reranker)]:
                retriever = Retriever(store, context_characters=56000, reranker=model)
                started = time.monotonic()
                found = retriever.search(project, 'evaluator', case['query'], top_k=10)
                latency = time.monotonic() - started
                ranking = list(dict.fromkeys(ids[s['document_id']] for s in found['sources']))
                if model and found['coverage_manifest']['reranker']['status'] != 'applied':
                    raise RuntimeError('Reranker silently degraded')
                row[mode] = dict(metrics(ranking, case['relevant']), ranking=ranking, seconds=round(latency, 3),
                                 required_preserved=set(case.get('required', [])).issubset(ranking))
            results.append(row)
            print(f'{index}/{len(dataset["questions"])} complete', flush=True)
    aggregate = {}
    for mode in ('hybrid', 'reranked'):
        seconds = sorted(r[mode]['seconds'] for r in results)
        aggregate[mode] = {name: round(statistics.mean(r[mode][name] for r in results), 4)
                           for name in ('recall_at_5', 'mrr_at_10', 'ndcg_at_10')}
        aggregate[mode].update(p50_seconds=round(statistics.median(seconds), 3), p95_seconds=seconds[math.ceil(.95 * len(seconds)) - 1],
                               required_preserved=all(r[mode]['required_preserved'] for r in results))
    quality = aggregate['reranked']
    passed = quality['recall_at_5'] >= .9 and quality['mrr_at_10'] >= .8 and quality['required_preserved']
    passed = passed and quality['ndcg_at_10'] >= aggregate['hybrid']['ndcg_at_10']
    report = {'synthetic_only': True, 'documents': len(dataset['documents']), 'questions': len(results),
              'model_id': reranker.model_id, 'metrics_level': 'document', 'passed': passed,
              'thresholds': {'recall_at_5': .9, 'mrr_at_10': .8, 'required_preserved': True, 'ndcg_not_below_baseline': True},
              'aggregate': aggregate, 'cases': results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
