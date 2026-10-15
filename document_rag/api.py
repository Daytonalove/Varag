from concurrent.futures import ThreadPoolExecutor
import hmac
import io
import logging
from pathlib import Path
import tempfile
import threading
import uuid

from .config import audit, load, local_token
audit()
from flask import Flask, jsonify, request, send_file
from werkzeug.exceptions import HTTPException
from .engine import Engine, StateConflict
from .extraction import SUPPORTED, MAX_BYTES
from .provider import ProviderError


def create_app(engine=None, token=None):
    engine = engine or Engine(load())
    token = token or local_token(engine.db.parent)
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = MAX_BYTES + 1024 * 1024
    app.json.ensure_ascii = False
    executor, slots, lock, jobs = ThreadPoolExecutor(max_workers=1), threading.BoundedSemaphore(4), threading.Lock(), {}
    app.extensions.update(engine=engine, executor=executor)

    @app.before_request
    def auth():
        provided = request.headers.get('X-RAG-Token', '')
        if not hmac.compare_digest(provided.encode(), token.encode()):
            return jsonify(error='Unauthorized'), 401

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return jsonify(error=exc.name), exc.code

    @app.errorhandler(StateConflict)
    def state_conflict(exc):
        return jsonify(error=str(exc)), 409

    @app.errorhandler(ValueError)
    def value_error(exc):
        return jsonify(error=str(exc)), 400

    @app.errorhandler(ProviderError)
    def provider_error(exc):
        return jsonify(error=str(exc)), 502

    @app.errorhandler(Exception)
    def other_error(exc):
        return jsonify(error='Ошибка обработки: ' + type(exc).__name__ + '. Подробности документа не записываются в лог.'), 500

    @app.get('/health')
    def health():
        return jsonify(status='ok', retrieval=engine.cfg['retrieval']['mode'], model=engine.cfg['gateway']['model'],
                       ocr=engine.cfg['ocr']['enabled'])

    @app.get('/provider/models')
    def models():
        available = engine.provider.models()
        selected = engine.cfg['gateway']['model']
        return jsonify(models=available, selected=selected, selected_available=selected in available)

    @app.get('/documents')
    def documents():
        return jsonify(engine.documents())

    @app.get('/documents/<doc_id>/original')
    def original(doc_id):
        name, content = engine.original(doc_id)
        return send_file(io.BytesIO(content), download_name=name, as_attachment=True, mimetype='application/octet-stream')

    @app.delete('/documents/<doc_id>')
    def delete_document(doc_id):
        engine.delete_document(doc_id)
        return jsonify(deleted=doc_id)

    def process(job_id, path, name, directory):
        with lock:
            jobs[job_id]['status'] = 'processing'
        try:
            result = engine.ingest(path, name)
            with lock:
                jobs[job_id].update(status='done', result=result)
        except Exception as exc:
            with lock:
                jobs[job_id].update(status='failed', error=str(exc) if isinstance(exc, ValueError) else 'Ошибка файла: ' + type(exc).__name__)
        finally:
            directory.cleanup()
            slots.release()

    @app.post('/documents')
    def upload():
        file = request.files.get('file')
        if file is None or not file.filename:
            raise ValueError('Выберите файл.')
        name = Path(file.filename.replace('\\', '/')).name
        if len(name) > 180 or Path(name).suffix.lower() not in SUPPORTED:
            raise ValueError('Недопустимое имя или формат файла.')
        if not slots.acquire(blocking=False):
            return jsonify(error='Очередь занята. Дождитесь обработки файлов.'), 429
        directory = tempfile.TemporaryDirectory(prefix='document-rag-')
        path = Path(directory.name) / ('upload' + Path(name).suffix.lower())
        try:
            file.save(path)
            if path.stat().st_size > MAX_BYTES:
                raise ValueError('Максимум 25 MiB на файл.')
            job_id = uuid.uuid4().hex
            with lock:
                for old in list(jobs):
                    if len(jobs) < 100:
                        break
                    if jobs[old]['status'] in ('done', 'failed'):
                        del jobs[old]
                jobs[job_id] = {'id': job_id, 'name': name, 'status': 'queued'}
            executor.submit(process, job_id, path, name, directory)
        except Exception:
            directory.cleanup()
            slots.release()
            raise
        return jsonify(job_id=job_id), 202

    @app.get('/jobs/<job_id>')
    def job(job_id):
        with lock:
            return (jsonify(dict(jobs[job_id])), 200) if job_id in jobs else (jsonify(error='Задание не найдено.'), 404)

    def query():
        body = request.get_json()
        if not isinstance(body, dict):
            raise ValueError('Ожидается JSON-объект.')
        question, ids = body.get('question'), body.get('document_ids')
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
            raise ValueError('Вопрос: 1–2000 символов.')
        if not isinstance(ids, list) or not ids or len(ids) > 100 or any(not isinstance(i, str) for i in ids):
            raise ValueError('Выберите документы перед отправкой вопроса.')
        if not isinstance(body.get('compare', False), bool):
            raise ValueError('Некорректный compare.')
        if body.get('chat_id') is not None and not isinstance(body['chat_id'], str):
            raise ValueError('Некорректный chat_id.')
        return body

    @app.post('/search')
    def search():
        body = query()
        return jsonify(engine.search(body['question'], body['document_ids']))

    @app.post('/ask')
    def ask():
        body = query()
        return jsonify(engine.ask(body['question'], body['document_ids'], body.get('compare', False), body.get('chat_id')))

    @app.get('/chats')
    def chats():
        return jsonify(engine.chats())

    @app.get('/chats/<chat_id>')
    def chat(chat_id):
        return jsonify(engine.chat(chat_id))

    @app.delete('/chats/<chat_id>')
    def delete_chat(chat_id):
        engine.delete_chat(chat_id)
        return jsonify(deleted=chat_id)

    return app


def serve():
    import os
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    create_app().run(host='127.0.0.1', port=int(os.environ.get('RAG_API_PORT', '8765')), threaded=True,
                     debug=False, use_reloader=False)


if __name__ == '__main__':
    serve()
