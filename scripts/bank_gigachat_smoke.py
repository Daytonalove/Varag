"""Opt-in live test: sends ONLY the synthetic fixture below, never workspace files."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from document_rag.bank.answering import GigaChatEvidence
from document_rag.config import ROOT
from document_rag.provider import ProviderError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', required=True, choices=['GIGACHAT_API_PERS', 'GIGACHAT_API_B2B', 'GIGACHAT_API_CORP'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--key-file', default=str(ROOT / 'secrets/gigachat-bank.key'))
    parser.add_argument('--ca-file', default=str(ROOT / 'secrets/gigachat-ca.pem'))
    parser.add_argument('--report', type=Path)
    parser.add_argument('--diagnose-synthetic', action='store_true', help='Print only this synthetic fixture response on validation failure')
    parser.add_argument('--semantic', action='store_true', help='Exercise generated claims and separate entailment check')
    parser.add_argument('--workspace-e2e', action='store_true', help='Exercise temporary SQLite workspace, uploads and async follow-up questions')
    args = parser.parse_args()
    from document_rag.bank.generation import GigaChatRAG
    provider_class = GigaChatRAG if args.semantic or args.workspace_e2e else GigaChatEvidence
    provider = provider_class(dict(scope=args.scope, model=args.model, key_file=args.key_file, ca_file=args.ca_file,
        oauth_url='https://ngw.devices.sberbank.ru:9443/api/v2/oauth', base_url='https://api.giga.chat/v1',
        timeout_seconds=30, max_output_tokens=3000 if args.semantic or args.workspace_e2e else 1000))
    report = {'synthetic_only': True, 'scope': args.scope, 'requested_model': args.model, 'tls_verification': True,
              'oauth': False, 'model_available': False, 'structured_answer_validated': False}
    synthetic_response = {}
    synthetic_responses = []
    original_call = provider.call
    def capture(path, body=None, **kwargs):
        result = original_call(path, body, **kwargs)
        if path == '/chat/completions':
            synthetic_response.update(choices=result.get('choices'), model=result.get('model'))
            if args.diagnose_synthetic:
                synthetic_responses.append({'choices': result.get('choices'), 'model': result.get('model')})
        return result
    provider.call = capture
    try:
        models = provider.models()
        report['oauth'] = True
        report['model_available'] = args.model in models
        if not report['model_available']:
            raise ProviderError('Выбранная модель недоступна; автоматическая замена запрещена.')
        if args.workspace_e2e:
            from scripts.bank_mvp_smoke import run
            report.update(workspace_e2e=run(provider), structured_answer_validated=True)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            return 0
        sources = [{'block_id': 'synthetic-A', 'text_exact': 'Лимит учебного комитета составляет 10 млн руб.'},
                   {'block_id': 'synthetic-B', 'text_exact': 'Лимит учебного комитета составляет 15 млн руб.'}]
        if args.semantic:
            for source in sources:
                source.update(document_name=source['block_id'] + '.txt', section_path='1', page_number='1')
            analysis = provider.analyze('Какие лимиты установлены для учебного комитета? Приведи оба значения.', sources)
            evidence = [e for c in analysis['claims'] for e in c['evidence']]
            metadata = {'model': analysis['model'], 'usage': [c['usage'] for c in analysis['calls']]}
            report.update(semantic_claims=len(analysis['claims']), stages=[c['stage'] for c in analysis['calls']], limitations=analysis['errors'])
        else:
            evidence, metadata = provider.select('Какие лимиты установлены для учебного комитета? Приведи оба источника.', sources)
        if {e['block_id'] for e in evidence} != {'synthetic-A', 'synthetic-B'}:
            raise ProviderError('Ответ не содержит обе синтетические ссылки.')
        report.update(structured_answer_validated=True, model_version=metadata['model'], usage=metadata['usage'])
    except (ProviderError, RuntimeError) as exc:
        report['error'] = str(exc)
        if args.diagnose_synthetic:
            report['synthetic_response'] = synthetic_response
            report['synthetic_responses'] = synthetic_responses
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return 0 if report['structured_answer_validated'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
