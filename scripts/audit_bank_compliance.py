"""Reproducible compliance probes. Reports unmet expectations, does not fix code.

All documents are synthetic; databases are temporary. --live explicitly sends
only these synthetic texts to the configured GigaChat provider.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from document_rag.bank.api import create_app
from document_rag.bank.answering import AnswerService
from document_rag.bank.generation import SCHEMA, validate_analysis
from document_rag.bank.retrieval import ContradictionService, Retriever
from document_rag.bank.runtime import configured_provider, configured_retriever
from document_rag.bank.embeddings import local_encoder
from document_rag.bank.store import Store
from document_rag.config import ROOT


def source(index, text):
    return dict(id=str(index), block_id=str(index), document_id=str(index),
                document_name=f'{index}.txt', document_version_id=f'v{index}',
                section_path='1', page_number='1', text_exact=text,
                text_normalized=text.lower().replace('ё', 'е'), quality_flags=[])


def local_checks():
    rows = []
    def record(name, expected, observed, passed, requirements):
        rows.append(dict(id=name, expected=expected, observed=observed, passed=bool(passed), requirements=requirements))
    cases = [
        ('different_products', 'Лимит кредитного комитета на ипотечную сделку составляет 10 млн руб.',
         'Лимит кредитного комитета на потребительскую сделку составляет 15 млн руб.', 'different_scope_no_conflict'),
        ('same_amount_incompatible_operators', 'Лимит кредитного комитета на сделку — не более 10 млн руб.',
         'Лимит кредитного комитета на сделку — свыше 10 млн руб.', 'nonempty'),
        ('different_durations', 'Срок рассмотрения заявки кредитным комитетом составляет 5 рабочих дней.',
         'Срок рассмотрения заявки кредитным комитетом составляет 10 рабочих дней.', 'nonempty'),
        ('different_periods', 'Лимит кредитного комитета на сделку в январе 2025 года составляет 10 млн руб.',
         'Лимит кредитного комитета на сделку в январе 2026 года составляет 15 млн руб.', 'different_scope_no_conflict'),
    ]
    for name, a, b, expected in cases:
        found = ContradictionService().assess([source(1, a), source(2, b)])
        statuses = [f['status'] for f in found]
        passed = bool(found) if expected == 'nonempty' else statuses == [expected]
        record(name, expected, statuses, passed, ['FR-17', 'FR-18', 'FR-39'])
    a = source(1, 'Лимит кредитного комитета составляет 10 млн руб.')
    malformed = {'claims': [{'status': 'needs_review', 'explanation': 'Нужно проверить область применения.',
                 'evidence': [{'block_id': '1', 'quote': a['text_exact']}]}], 'assessments': [], 'limitations': []}
    try:
        validate_analysis(json.dumps(malformed), [a])
        rejected = False
    except ValueError:
        rejected = True
    record('json_schema_claim_type', 'reject assessment object inside claims', 'rejected' if rejected else 'accepted', rejected, ['GC-07', 'GC-07a'])
    original = source(1, 'Лимит кредитного комитета — не более 10 млн руб.')
    opposite = {'claims': [{'text': 'Лимит кредитного комитета — свыше 10 млн руб.',
                'evidence': [{'block_id': '1', 'quote': original['text_exact']}]}], 'assessments': [], 'limitations': []}
    try:
        validate_analysis(json.dumps(opposite), [original])
        rejected = False
    except ValueError:
        rejected = True
    record('operator_validator', 'deterministic rejection of changed operator', 'rejected' if rejected else 'accepted; relies on LLM entailment', rejected, ['FR-17', 'FR-42'])
    b = source(2, 'Лимит кредитного комитета составляет 15 млн руб.')
    c = source(3, 'С 01.07.2026 изменяется порядок выдачи отпусков сотрудникам.')
    amendment = {'claims': [], 'assessments': [{'status': 'explained_by_amendment',
                 'explanation': 'Различие лимитов объяснено изменением.',
                 'evidence': [{'block_id': s['block_id'], 'quote': s['text_exact']} for s in [a, b, c]]}], 'limitations': []}
    observed = validate_analysis(json.dumps(amendment), [a, b, c], '2026-08-01')['assessments'][0]['status']
    record('unrelated_amendment_validator', 'needs_review or rejection', observed, observed != 'explained_by_amendment', ['FR-11', 'FR-21', 'FR-42'])
    with tempfile.TemporaryDirectory(prefix='bank-compliance-') as temporary:
        store = Store(Path(temporary) / 'audit.sqlite3')
        project = store.create_project('Синтетический аудит', 'auditor')['id']
        old = store.upload(project, 'auditor', 'policy.txt', 'С 01.01.2025 лимит кредитного комитета составляет 10 млн руб.'.encode())
        store.process_one()
        store.upload(project, 'auditor', 'policy.txt', 'С 01.07.2026 лимит кредитного комитета составляет 15 млн руб.'.encode(), old['document_id'])
        store.process_one()
        answer = AnswerService(store).ask(project, 'auditor', 'Какой лимит кредитного комитета?', as_of='2025-03-01')
        versions = {s['document_version_id'] for s in answer['sources']}
        record('historical_version_retrieval', 'prior version present for historical date', {'old_version_present': old['version_id'] in versions}, old['version_id'] in versions, ['FR-19', 'FR-21'])
        client = create_app(store, {'test': 'auditor'}, start_worker=False, demo_mode=True).test_client()
        headers = {'X-RAG-Token': 'test'}
        metadata = client.patch('/api/v1/documents/' + old['document_id'] + '/metadata', headers=headers, json={'effective_from': 'не дата'})
        record('metadata_date_validation', '400 for invalid date', metadata.status_code, metadata.status_code == 400, ['FR-09', 'FR-10'])
        exported = client.get('/api/v1/answers/' + answer['id'] + '/export?format=pdf', headers=headers)
        record('direct_pdf_export', 'application/pdf and PDF signature', {'status': exported.status_code, 'mime': exported.mimetype},
               exported.mimetype == 'application/pdf' and exported.data.startswith(b'%PDF'), ['TZ-2.1'])
    return rows


def live_checks(selected_cases=None, diagnose=False):
    provider = configured_provider()
    if provider is None:
        raise RuntimeError('GigaChat not configured')
    encoder = local_encoder(ROOT / 'models/multilingual-e5-small')
    rows = []
    a = 'Кредитная политика, пункт 4.2. С 01.01.2026 лимит кредитного комитета по одной рублёвой сделке малого бизнеса составляет 10 млн руб.'
    b = 'Положение о кредитном комитете, пункт 3.1. С 01.07.2026 лимит кредитного комитета по одной рублёвой сделке малого бизнеса составляет 15 млн руб.'
    c = 'Решение правления. С 01.07.2026 пункт 4.2 Кредитной политики изменяется: лимит кредитного комитета по одной рублёвой сделке малого бизнеса увеличивается с 10 млн руб. до 15 млн руб. Пункт 3.1 Положения о кредитном комитете подтверждает новый лимит.'
    cases = [
        dict(id='amendment_after', texts=[a, b, c], question='Какой лимит кредитного комитета действует на дату проверки? Объясни изменение.', as_of='2026-08-01', expected='explained_by_amendment'),
        dict(id='amendment_before', texts=[a, b, c], question='Какой лимит кредитного комитета действует на дату проверки? Укажи исторический лимит.', as_of='2026-03-01', expected='historical_10'),
        dict(id='products', texts=['Лимит кредитного комитета на ипотечную сделку составляет 10 млн руб.', 'Лимит кредитного комитета на потребительскую сделку составляет 15 млн руб.'], question='Какие лимиты установлены для кредитного комитета? Есть ли расхождение?', as_of=None, expected='different_scope_no_conflict'),
        dict(id='working_days', texts=['Срок рассмотрения заявки кредитным комитетом составляет 5 рабочих дней.', 'Срок рассмотрения заявки кредитным комитетом составляет 10 рабочих дней.'], question='Какой срок рассмотрения заявки кредитным комитетом?', as_of=None, expected='potential_unresolved_conflict'),
        dict(id='operators', texts=['Лимит кредитного комитета на сделку — не более 10 млн руб.', 'Лимит кредитного комитета на сделку — свыше 10 млн руб.'], question='Какие ограничения суммы сделки установлены для кредитного комитета?', as_of=None, expected='potential_unresolved_conflict'),
    ]
    with tempfile.TemporaryDirectory(prefix='bank-compliance-live-') as temporary:
        store = Store(Path(temporary) / 'audit.sqlite3', encoder)
        retriever = configured_retriever(store, provider)
        service = AnswerService(store, provider, True, retriever)
        for case in cases:
            if selected_cases and case['id'] not in selected_cases:
                continue
            captured=[]
            real_call=provider.call
            def capture(path,body=None):
                response=real_call(path,body)
                if diagnose and path=='/chat/completions':
                    captured.append(response['choices'][0]['message']['content'])
                return response
            provider.call=capture
            project = store.create_project(case['id'], 'tester')['id']
            for index, text in enumerate(case['texts']):
                uploaded = store.upload(project, 'tester', f'{index}.txt', text.encode())
                with store.connect() as conn:
                    conn.execute('UPDATE documents SET external_allowed=1 WHERE id=?', (uploaded['document_id'],))
                store.process_one()
            answer = service.ask(project, 'tester', case['question'], True, as_of=case['as_of'])
            statuses = [d['status'] for d in answer['discrepancies']]
            if case['expected'] == 'historical_10':
                # Human review required for historical semantics; retain exact synthetic answer.
                passed = None
            else:
                passed = answer['mode'] == 'gigachat_answer' and case['expected'] in statuses
                if case['expected'] in ('explained_by_amendment', 'different_scope_no_conflict'):
                    passed = passed and 'potential_unresolved_conflict' not in statuses
            row = dict(id=case['id'], expected=case['expected'], automated_pass=passed,
                       answer=answer['answer_text'], mode=answer['mode'], verification_status=answer['verification_status'],
                       statuses=statuses, discrepancies=answer['discrepancies'], warnings=answer['warnings'],
                       source_count=len(answer['sources']), model_version=answer['model_version'])
            provider.call=real_call
            if diagnose:
                row['synthetic_model_responses']=captured
            with store.connect() as conn:
                row['llm_diagnostics'] = [dict(status=r['status'],stage=r['stage'],metadata=json.loads(r['metadata']))
                    for r in conn.execute('SELECT * FROM llm_calls WHERE question_id=? ORDER BY rowid',(answer['id'],))]
            rows.append(row)
            print(json.dumps({'case':case['id'], 'automated_pass':passed, 'mode':answer['mode'], 'statuses':statuses}, ensure_ascii=False), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--case',action='append')
    parser.add_argument('--diagnose-synthetic',action='store_true')
    parser.add_argument('--report', type=Path, default=ROOT / 'evaluation/results/compliance-probes.json')
    args = parser.parse_args()
    report = dict(synthetic_only=True, persistent_user_data_modified=False, local_checks=local_checks())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if args.live:
        report['live_checks'] = live_checks(args.case,args.diagnose_synthetic)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'local_checks':len(report['local_checks']), 'unmet':sum(not r['passed'] for r in report['local_checks']), 'report':str(args.report)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
