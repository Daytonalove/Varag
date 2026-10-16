"""Model-specific RAG budgets and bounded, provider-counted context packing.

Verified 2026-10-16 against developers.sber.ru/docs/ru/gigachat/models/
gigachat-2-{lite,pro,max}. Unknown model IDs never inherit a larger model's window.
The token endpoint measures text components; framing/boundaries retain a reserve.
"""
import copy
import json
import math
import os
import time

from document_rag.provider import ProviderError

MODEL_WINDOWS = {'GigaChat-2': 128000, 'GigaChat-2-Pro': 128000, 'GigaChat-2-Max': 128000}


def limits(cfg):
    model = cfg.get('model', '')
    base = model.split(':', 1)[0]
    configured = cfg.get('model_context_limits', {})
    if not isinstance(configured, dict):
        raise ValueError('GIGACHAT_MODEL_CONTEXT_LIMITS должен быть JSON-объектом {модель: токены}.')
    declared = configured.get(model, configured.get(base))
    known = MODEL_WINDOWS.get(base)
    window = known or 32768
    source = 'documented' if known else 'conservative_fallback'
    if declared is not None:
        if type(declared) is not int or not 1024 <= declared <= 1000000:
            raise ValueError('Лимит модели должен быть целым числом от 1024 до 1000000.')
        window = min(window, declared) if known else declared
        source = 'configured'
    # An explicit global value is an administrative cap, never a model upgrade.
    if cfg.get('context_window_tokens') is not None:
        window = min(window, int(cfg['context_window_tokens']))
    if window <= 0:
        raise ValueError('Контекстное окно должно быть положительным.')
    output = max(400, min(int(os.environ.get('FAST_RAG_OUTPUT_TOKENS') or (6000 if window >= 128000 else 2500)), 8192))
    max_sources = max(1, min(int(os.environ.get('FAST_RAG_MAX_SOURCES') or 64), 128))
    max_chars = max(2000, min(int(os.environ.get('FAST_RAG_CONTEXT_CHARS') or min(window * 3, 240000)), 500000))
    per_source = max(300, min(int(os.environ.get('FAST_RAG_SOURCE_CHARS') or 6000), 12000))
    return dict(model=model, context_tokens=window, limit_source=source,
                output_tokens=output, max_sources=max_sources, context_characters=max_chars,
                source_characters=per_source)


def dumps(value):
    return json.dumps(value, ensure_ascii=False)


def pack_context(provider, body, payload, chosen, schema, policy, *, before_call, history, record_call):
    """One batched token-count request at most; no completion/retry in this function.

    Count the fixed envelope and individual source components once. Greedily
    fit complete snapshots, skipping an oversized source instead of losing all
    subsequent short documents. On counter failure use the byte bound.
    """
    snapshots = list(chosen)
    candidates = list(payload['sources'])
    base_body = copy.deepcopy(body)
    base_payload = dict(payload, sources=[])
    base_body['messages'][1]['content'] = dumps(base_payload)
    components = [dumps(base_body)] + [dumps(s) for s in candidates]
    costs = [len(s.encode('utf-8')) for s in components]
    method, count_failure = 'utf8_upper_bound', None
    if provider.cfg.get('count_tokens', False):
        before_call(snapshots, history or [])
        call_id = record_call('started', 'tokens_count', snapshots, {'model': provider.cfg['model']})
        started = time.monotonic()
        try:
            with provider.call_lock:
                counted = provider.call('/tokens/count', {'model': provider.cfg['model'], 'input': components},
                                        retry_unauthorized=False, timeout_seconds=min(15,provider.cfg.get('timeout_seconds',90)))
            if (not isinstance(counted, list) or len(counted) != len(components)
                    or any(not isinstance(row, dict) or type(row.get('tokens')) is not int
                           or row['tokens'] <= 0 or row['tokens'] > len(text.encode('utf-8')) + 256
                           for row, text in zip(counted, components))):
                raise ValueError('Некорректный результат подсчёта токенов.')
            costs = [row['tokens'] for row in counted]
            method = 'provider_components'
            record_call('succeeded', 'tokens_count', snapshots, {'call_id': call_id,
                'model': provider.cfg['model'], 'component_tokens': sum(costs),
                'elapsed_ms': round((time.monotonic()-started)*1000)})
        except (ProviderError, ValueError, TypeError, KeyError) as exc:
            count_failure = exc.code if isinstance(exc, ProviderError) else 'invalid_response'
            record_call('failed', 'tokens_count', snapshots, {'call_id': call_id,
                'reason': count_failure, 'fallback': 'utf8_upper_bound',
                'elapsed_ms': round((time.monotonic()-started)*1000)})
            if isinstance(exc, ProviderError) and exc.status in (401, 403):
                raise
    reserve = 1024
    ceiling = policy['context_tokens'] - policy['output_tokens']
    chosen.clear(); payload['sources'] = []
    total = costs[0]
    for source, item, cost in zip(snapshots, candidates, costs[1:]):
        # Allow for separator tokenization and chat/schema framing differences.
        estimate = math.ceil((total + cost + 32) * 1.02) + reserve
        if estimate > ceiling:
            continue
        chosen.append(source)
        payload['sources'].append(dict(item, source_id='Q'+str(len(chosen))))
        total += cost + 32
    if not chosen:
        raise ProviderError('Нужный контекст не помещается в окно выбранной модели. Уточните вопрос или выберите модель с большим контекстом.', 422, 'context_budget')
    body['response_format']['schema'] = schema([s['source_id'] for s in payload['sources']])
    body['messages'][1]['content'] = dumps(payload)
    estimate = math.ceil(total * 1.02) + reserve
    if method == 'utf8_upper_bound':
        # Check the actual assembled JSON, including nested escaping, on fallback.
        estimate = len(dumps(body).encode('utf-8')) + reserve
        while estimate > ceiling and len(chosen) > 1:
            chosen.pop(); payload['sources'].pop()
            body['response_format']['schema'] = schema([s['source_id'] for s in payload['sources']])
            body['messages'][1]['content'] = dumps(payload)
            estimate = len(dumps(body).encode('utf-8')) + reserve
        if estimate > ceiling:
            raise ProviderError('Контекст превышает бюджет выбранной модели.', 422, 'context_budget')
    selected_ids = {s['block_id'] for s in chosen}
    return dict(**policy, token_count_method=method, input_tokens_estimate=estimate,
                token_count_failure=count_failure, selected_sources=len(chosen),
                selected_document_ids=list(dict.fromkeys(s.get('document_id') for s in chosen if s.get('document_id'))),
                omitted_source_ids=[s['block_id'] for s in snapshots if s['block_id'] not in selected_ids])
