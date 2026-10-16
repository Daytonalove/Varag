"""GigaChat OAuth and REST via urllib. Credentials never appear in errors or logs."""
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


class ProviderError(RuntimeError):
    def __init__(self, message, status=None, code=None):
        super().__init__(message)
        self.status = status
        self.code = code or ('http_'+str(status) if status else 'provider_error')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class GigaChat:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.RLock()
        self.cached = None

    def credential(self):
        key = os.environ.get('GIGACHAT_CREDENTIALS', '').strip()
        if not key:
            try:
                key = Path(self.cfg['key_file']).read_text().strip()
            except (OSError, KeyError):
                raise ProviderError('Ключ не настроен. Выполните python manage.py configure.') from None
        if not key or len(key) > 16384 or any(c.isspace() for c in key):
            raise ProviderError('Нужен Authorization key одной строкой, без префикса Basic.')
        return key

    def request(self, url, headers, data=None, *, timeout_seconds=None):
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != 'https' or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
            raise ProviderError('API URL должен быть HTTPS без учётных данных и параметров.')
        stage = 'OAuth' if parts.path.endswith('/oauth') else parts.path
        try:
            tls = ssl.create_default_context()
            if self.cfg.get('ca_file'):
                tls.load_verify_locations(cafile=self.cfg['ca_file'])
            opener = urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(context=tls))
            req = urllib.request.Request(url, headers=headers, data=data)
            timeout = self.cfg.get('timeout_seconds', 90) if timeout_seconds is None else timeout_seconds
            with opener.open(req, timeout=timeout) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ProviderError('Слишком большой ответ API.')
            return json.loads(raw)
        except urllib.error.HTTPError as exc:
            reasons = {400: 'проверьте параметры запроса и scope', 401: 'отказ авторизации',
                       402: 'проверьте остаток токенов модели', 403: 'нет доступа к ресурсу',
                       404: 'проверьте адрес API и ID модели', 422: 'модель отклонила параметры или размер запроса',
                       429: 'ограничение частоты запросов или квоты'}
            # Restrict provider error details to known public messages; never print arbitrary bodies.
            detail = ''
            try:
                message = json.loads(exc.read(16000)).get('message', '')
                if message in ('Unauthorized', 'Payment Required', 'Permission denied', 'Invalid model', 'Model not found'):
                    detail = ' (' + message + ')'
            except (ValueError, AttributeError):
                pass
            raise ProviderError(f'GigaChat {stage}: HTTP {exc.code}{detail}; ' + reasons.get(exc.code, 'ошибка сервиса'), exc.code) from None
        except urllib.error.URLError as exc:
            cause = exc.reason
            if isinstance(cause,TimeoutError):
                raise ProviderError(f'GigaChat {stage}: время ожидания ответа истекло.',code='timeout') from None
            if isinstance(cause, ssl.SSLCertVerificationError):
                raise ProviderError(f'GigaChat {stage}: сертификат TLS не подтверждён. Настройте gateway.ca_file.') from None
            raise ProviderError(f'GigaChat {stage}: ошибка сети ({type(cause).__name__}).') from None
        except (ssl.SSLError, OSError, TimeoutError) as exc:
            if isinstance(exc,TimeoutError):
                raise ProviderError(f'GigaChat {stage}: время ожидания ответа истекло.',code='timeout') from None
            raise ProviderError(f'GigaChat {stage}: ошибка сети/CA ({type(exc).__name__}).') from None
        except (ValueError, TypeError):
            raise ProviderError(f'GigaChat {stage}: некорректный JSON ответа.') from None

    def token(self, refresh=False):
        key = self.credential()
        fingerprint = hashlib.sha256(key.encode()).hexdigest()
        with self.lock:
            if not refresh and self.cached and self.cached[0] == fingerprint and self.cached[2] > time.time() + 60:
                return self.cached[1]
            data = self.request(self.cfg['oauth_url'], {
                'Authorization': 'Basic ' + key, 'RqUID': str(uuid.uuid4()),
                'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json',
            }, urllib.parse.urlencode({'scope': self.cfg['scope']}).encode())
            try:
                token, expires = data['access_token'], float(data['expires_at'])
                if expires > 100_000_000_000:
                    expires /= 1000
                if not isinstance(token, str) or not token or any(c.isspace() for c in token) or not time.time() < expires < time.time() + 86400:
                    raise ValueError()
            except (KeyError, ValueError, TypeError):
                raise ProviderError('OAuth вернул некорректный токен или срок действия.') from None
            self.cached = (fingerprint, token, expires)
            return token

    def call(self, path, body=None, *, retry_unauthorized=None, timeout_seconds=None):
        retry = self.cfg.get('retry_unauthorized', True) if retry_unauthorized is None else retry_unauthorized
        for attempt in range(2 if retry else 1):
            token = self.token(refresh=bool(attempt))
            try:
                return self.request(self.cfg['base_url'].rstrip('/') + path, {
                    'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json',
                    'Accept': 'application/json', 'User-Agent': 'DocumentRAG-Local/1.0',
                }, json.dumps(body, ensure_ascii=False).encode() if body is not None else None,
                    **({'timeout_seconds':timeout_seconds} if timeout_seconds is not None else {}))
            except ProviderError as exc:
                if exc.status != 401 or attempt or not retry:
                    raise

    def models(self):
        try:
            return [row['id'] for row in self.call('/models')['data']]
        except (KeyError, TypeError):
            raise ProviderError('Некорректный список моделей GigaChat.') from None

    def generate(self, system, question):
        result = self.call('/chat/completions', {
            'model': self.cfg['model'], 'stream': False, 'temperature': 0.1,
            'max_tokens': self.cfg.get('max_output_tokens', 1800),
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': question}],
        })
        try:
            choice = result['choices'][0]
            if choice.get('finish_reason') not in (None, 'stop'):
                raise ProviderError('Ответ модели обрезан или заблокирован. Уменьшите задачу или измените лимит ответа.')
            answer = choice['message']['content']
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError()
            return answer
        except (KeyError, IndexError, TypeError, ValueError):
            raise ProviderError('GigaChat не вернул ожидаемый текст.') from None
