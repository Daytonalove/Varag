import importlib.metadata
import json
import os
from pathlib import Path
import re
import secrets

ROOT = Path(__file__).resolve().parents[1]


def audit():
    blocked = []
    for dist in importlib.metadata.distributions():
        name = re.sub(r'[-_.]+', '-', dist.metadata.get('Name', '').lower())
        if 'openai' in name or name == 'tiktoken' or name.startswith('nvidia-'):
            blocked.append(name)
    if blocked:
        raise RuntimeError('Запрещённые пакеты в окружении приложения: ' + ', '.join(sorted(blocked)))


def load(path=None):
    path = Path(path or os.environ.get('RAG_CONFIG') or ROOT / 'config.json').resolve()
    if not path.exists():
        raise ValueError('Нет config.json. Выполните python scripts/bootstrap.py.')
    cfg = json.loads(path.read_text(encoding='utf-8'))
    for group, field in [('storage', 'database'), ('retrieval', 'model_directory'),
                         ('gateway', 'key_file'), ('gateway', 'ca_file')]:
        value = cfg[group].get(field)
        if value:
            cfg[group][field] = str((path.parent / Path(value).expanduser()).resolve())
    r = cfg['retrieval']
    if r['mode'] not in ('hybrid', 'lexical') or not 200 <= r['chunk_characters'] <= 1500:
        raise ValueError('Некорректный режим поиска или размер фрагмента (200–1500).')
    if not 1 <= r['top_k'] <= 20 or not 3000 <= r['context_characters'] <= 30000:
        raise ValueError('Некорректный лимит поиска или контекста.')
    return cfg


def local_token(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'server.token'
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        token = path.read_text().strip()
    else:
        token = secrets.token_urlsafe(32)
        with os.fdopen(fd, 'w') as file:
            file.write(token)
    if len(token) < 24:
        raise ValueError('Некорректный внутренний токен в data/server.token.')
    return token
