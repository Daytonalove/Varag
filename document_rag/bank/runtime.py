"""Server-side configuration. Credentials never enter the public settings response."""
import json
import os
from pathlib import Path

from document_rag.config import ROOT
from .generation import GigaChatRAG


def configured_retriever(store, provider=None):
    from .retrieval import Retriever
    from .reranking import LocalCrossEncoder
    mode = os.environ.get('BANK_RERANKER_MODE', 'required')
    if mode not in ('required', 'disabled'):
        raise RuntimeError('BANK_RERANKER_MODE: required или disabled.')
    directory = os.environ.get('RERANKER_MODEL_DIRECTORY') or str(ROOT / 'models/mmarco-mMiniLMv2-L12-H384-v1')
    reranker = LocalCrossEncoder(directory) if mode == 'required' else None
    return Retriever(store, context_characters=56000 if hasattr(provider, 'analyze') else 24000, reranker=reranker)


def configured_provider():
    legacy = {}
    config = ROOT / 'config.json'
    if config.exists():
        legacy = json.loads(config.read_text()).get('gateway', {})
    scope = os.environ.get('GIGACHAT_SCOPE') or legacy.get('scope')
    model = os.environ.get('GIGACHAT_MODEL') or legacy.get('model')
    key_file = os.environ.get('GIGACHAT_KEY_FILE') or str(ROOT / 'secrets/gigachat-bank.key')
    if scope not in ('GIGACHAT_API_PERS', 'GIGACHAT_API_B2B', 'GIGACHAT_API_CORP') or not model:
        return None
    if not os.environ.get('GIGACHAT_AUTH_KEY') and not Path(key_file).is_file():
        return None
    ca = os.environ.get('GIGACHAT_CA_BUNDLE') or legacy.get('ca_file') or ''
    if ca and not Path(ca).is_absolute():
        ca = str(ROOT / ca)
    return GigaChatRAG(dict(scope=scope, model=model, key_file=key_file, ca_file=ca,
        oauth_url=os.environ.get('GIGACHAT_OAUTH_URL') or 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth',
        base_url='https://api.giga.chat/v1', timeout_seconds=30, max_output_tokens=3000,
        context_window_tokens=int(os.environ.get('GIGACHAT_CONTEXT_TOKENS','32768')), count_tokens=True,reference_citations=True,
        question_token_budget=int(os.environ.get('GIGACHAT_QUESTION_TOKEN_BUDGET','120000')),
        requests_per_minute=int(os.environ.get('GIGACHAT_REQUESTS_PER_MINUTE','30')),
        allowed_model_versions=[v.strip() for v in os.environ.get('GIGACHAT_ALLOWED_MODEL_VERSIONS','').split(',') if v.strip()]))
