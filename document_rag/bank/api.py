"""Loopback-only development API; refuses a production/demo-auth combination."""
import hmac
import io
import logging
import mimetypes
import os
from pathlib import Path
import threading

from flask import Flask, g, jsonify, request, send_file
from werkzeug.exceptions import HTTPException

from document_rag.config import ROOT, local_token
from .answering import AnswerService, GigaChatEvidence
from .store import NotFound, Store, encoded, uid
from .jobs import QuestionQueue


class Worker:
    def __init__(self, store, questions=None):
        import fcntl
        self.lock_file = open(str(store.database) + '.worker.lock', 'a')
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock_file.close()
            raise RuntimeError('Для этой базы уже работает worker.') from None
        self.store, self.stop_event = store, threading.Event()
        store.recover()
        if questions:
            questions.recover()
        self.errors = 0
        self.threads = [threading.Thread(target=self.run, args=(task,), daemon=True, name=name)
                        for name, task in [('bank-ingest', store.process_one), ('bank-index', store.process_maintenance)] +
                        ([('bank-questions', questions.process_one)] if questions else [])]
        for thread in self.threads:
            thread.start()

    def run(self, task):
        while not self.stop_event.is_set():
            try:
                if task():
                    continue
            except Exception:
                self.errors += 1
            if not self.stop_event.is_set():
                self.stop_event.wait(.3)

    def close(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join()
        self.lock_file.close()


def create_app(store=None, tokens=None, start_worker=True, service=None, demo_mode=None):
    if demo_mode is None:
        demo_mode = os.environ.get('LOCAL_DEMO_MODE', '').lower() == 'true'
    if not demo_mode or os.environ.get('APP_ENV', 'development') != 'development':
        raise RuntimeError('Этот контур пока поддерживает только LOCAL_DEMO_MODE=true и APP_ENV=development. Production требует SSO и отдельного развёртывания.')
    store = store or Store(os.environ.get('BANK_DATABASE', str(ROOT / 'data/bank/audit.sqlite3')))
    tokens = tokens or {local_token(store.database.parent): 'local-auditor'}
    if path:=os.environ.get('BANK_LOCAL_USERS_FILE'):
        import json
        file=Path(path)
        if file.stat().st_mode & 0o077:
            raise RuntimeError('Файл локальных токенов должен иметь права 0600.')
        configured=json.loads(file.read_text())
        if not isinstance(configured,dict) or any(not isinstance(k,str) or len(k)<24 or not isinstance(v,str) or not 1<=len(v)<=160 for k,v in configured.items()):
            raise RuntimeError('Файл пользователей: объект {длинный_случайный_токен: user_id}.')
        tokens=dict(tokens,**configured)
    service = service or AnswerService(store)
    questions = QuestionQueue(store, service)
    app = Flask(__name__)
    app.json.ensure_ascii = False
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
    app.extensions.update(bank_store=store, bank_service=service, bank_questions=questions)
    if start_worker:
        app.extensions['bank_worker'] = Worker(store, questions)

    @app.before_request
    def authenticate():
        token = request.headers.get('X-RAG-Token', '')
        g.user = next((user for secret, user in tokens.items() if hmac.compare_digest(secret.encode(), token.encode())), None)
        if not g.user:
            return jsonify(error='Unauthorized'), 401

    @app.after_request
    def secure_headers(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.errorhandler(NotFound)
    def missing(exc):
        return jsonify(error='Ресурс не найден.'), 404

    @app.errorhandler(ValueError)
    def invalid(exc):
        return jsonify(error=str(exc)), 400

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return jsonify(error=exc.name), exc.code

    @app.errorhandler(Exception)
    def internal(exc):
        return jsonify(error='Ошибка обработки: ' + type(exc).__name__), 500

    def body():
        value = request.get_json()
        if not isinstance(value, dict):
            raise ValueError('Ожидается JSON-объект.')
        return value

    @app.get('/api/v1/health')
    def health():
        return jsonify(status='ok', mode='local_development', authentication='local_token',
                       external_llm_approved=service.external_approved, production_ready=False,
                       gigachat_configured=service.provider is not None,
                       model=getattr(service.provider, 'cfg', {}).get('model'),
                       automatic_external=service.automatic_external,
                       retrieval_mode='hybrid' if store.encoder else 'lexical',
                       reranker={'enabled': service.retriever.reranker is not None,
                                 'model_id': getattr(service.retriever.reranker, 'model_id', None)},
                       worker_running=bool(app.extensions.get('bank_worker')) and all(t.is_alive() for t in app.extensions['bank_worker'].threads),
                       worker_errors=getattr(app.extensions.get('bank_worker'), 'errors', 0),
                       can_configure=g.user == 'local-auditor')

    @app.patch('/api/v1/settings')
    def settings():
        if g.user != 'local-auditor':
            raise NotFound()
        payload = body()
        if set(payload) != {'external_llm_approved'} or type(payload['external_llm_approved']) is not bool:
            raise ValueError('Ожидается boolean external_llm_approved.')
        approved = payload['external_llm_approved']
        if service.automatic_external and not approved:
            raise ValueError('В режиме Варяг ответы через GigaChat включены постоянно.')
        if approved and not service.provider:
            raise ValueError('Сначала настройте серверный ключ, scope и модель GigaChat.')
        service.external_approved = approved
        with store.connect() as conn:
            conn.execute('INSERT OR REPLACE INTO settings VALUES(?,?)', ('external_llm_approved', encoded(approved)))
            store.event(conn, 'local-system', g.user, 'llm.policy_changed', 'external_llm_approved', payload)
        return jsonify(external_llm_approved=approved)

    @app.post('/api/v1/provider/check')
    def provider_check():
        if g.user != 'local-auditor':
            raise NotFound()
        if not service.provider:
            raise ValueError('GigaChat не настроен.')
        from document_rag.provider import ProviderError
        try:
            available = service.provider.models()
        except ProviderError as exc:
            return jsonify(error=str(exc)), 502
        model = service.provider.cfg['model']
        return jsonify(model=model, available=model in available, tls_verified=True)

    @app.get('/api/v1/provider/models')
    def provider_models():
        from document_rag.provider import ProviderError
        try:
            from .context import limits
            models=service.available_models()
            return jsonify(models=models, default=service.provider.cfg['model'],
                           limits={m:limits(dict(service.provider.cfg,model=m)) for m in models})
        except ProviderError as exc:
            return jsonify(error=str(exc)), 502

    @app.get('/api/v1/chats')
    def chats():
        result=[]
        for project in store.projects(g.user):
            with store.connect() as conn:
                rows=conn.execute('''SELECT c.*, (SELECT model_id FROM question_jobs j
                    WHERE j.conversation_id=c.id ORDER BY j.rowid DESC LIMIT 1) AS model_id
                    FROM conversations c WHERE c.project_id=? AND c.user_id=? ORDER BY c.rowid DESC''',
                    (project['id'],g.user)).fetchall()
                for row in rows:
                    result.append(dict(project_id=project['id'],conversation_id=row['id'],
                                       title=row['title'],model_id=row['model_id'],created=row['created']))
                if not rows:
                    legacy=conn.execute('SELECT 1 FROM answers WHERE project_id=? AND user_id=? LIMIT 1',
                                        (project['id'],g.user)).fetchone()
                    result.append(dict(project_id=project['id'],conversation_id='legacy' if legacy else None,
                                       title=project['name'],model_id=None,created=project['created']))
        return jsonify(sorted(result,key=lambda row:row['created'],reverse=True))

    @app.post('/api/v1/chats')
    def new_chat():
        project=store.create_project('Новый чат',g.user)
        return jsonify(project_id=project['id'],conversation_id=None,title='Новый чат'),201

    @app.get('/api/v1/projects')
    def projects():
        return jsonify(store.projects(g.user))

    @app.post('/api/v1/projects')
    def create_project():
        return jsonify(store.create_project(body().get('name'), g.user)), 201

    @app.get('/api/v1/projects/<project>/documents')
    def documents(project):
        docs=store.documents(project, g.user)
        with store.connect() as conn:
            for doc in docs:
                row=conn.execute('SELECT completed,total FROM processing_progress WHERE version_id=?',(doc['active_version_id'],)).fetchone()
                doc['progress']=dict(row) if row else None
        return jsonify(docs)

    @app.get('/api/v1/projects/<project>/members')
    def members(project):
        from .operations import owner
        with store.connect() as conn:
            owner(store,conn,project,g.user)
            return jsonify([dict(r) for r in conn.execute('SELECT user_id,role FROM members WHERE project_id=?',(project,))])

    @app.put('/api/v1/projects/<project>/members/<member>')
    def member_update(project,member):
        from .operations import set_member
        payload=body()
        if set(payload)!={'role'}:
            raise ValueError('Ожидается поле role.')
        set_member(store,project,g.user,member,payload['role'])
        return jsonify(user_id=member,role=payload['role'])

    @app.put('/api/v1/documents/<document>/acl')
    def acl_update(document):
        from .operations import set_acl
        payload=body()
        set_acl(store,document,g.user,payload)
        return jsonify(payload)

    @app.get('/api/v1/projects/<project>/metrics')
    def project_metrics(project):
        from .operations import metrics
        return jsonify(metrics(store,project,g.user))

    @app.post('/api/v1/projects/<project>/retention')
    def project_retention(project):
        from .operations import retention
        payload=body()
        if set(payload)-{'days','execute'}:
            raise ValueError('Ожидаются days и execute.')
        return jsonify(retention(store,project,g.user,payload.get('days',365),payload.get('execute',False)))

    @app.post('/api/v1/projects/<project>/documents')
    def upload(project):
        files = request.files.getlist('files') or request.files.getlist('file')
        if not 1 <= len(files) <= 100:
            raise ValueError('Выберите от 1 до 100 файлов (до 100 MiB на пакет).')
        # A batch can partially succeed: each item gets its own result, never hide accepted uploads.
        results = []
        external = 'true' if service.automatic_external else request.form.get('external_allowed', 'false')
        if external not in ('true', 'false'):
            raise ValueError('external_allowed должен быть true или false.')
        if external == 'true' and not service.automatic_external:
            with store.connect() as conn:
                if store.project_access(conn, project, g.user) != 'owner':
                    raise NotFound()
        for file in files:
            try:
                result = store.upload(project, g.user, file.filename or '', file.read(25 * 1024 * 1024 + 1), request.form.get('document_id'))
                if external == 'true' and not result['duplicate']:
                    with store.connect() as conn:
                        store.document_access(conn, result['document_id'], g.user, True)
                        conn.execute('UPDATE documents SET external_allowed=1 WHERE id=?', (result['document_id'],))
                        conn.execute('UPDATE projects SET index_version=index_version+1 WHERE id=?', (project,))
                        store.event(conn, project, g.user, 'document.external_allowed', result['document_id'])
                results.append(result)
            except ValueError as exc:
                results.append({'error': str(exc)})
        return jsonify(results), 202

    @app.get('/api/v1/documents/<document>/original')
    def original(document):
        name, content = store.original(document, g.user, request.args.get('version'))
        with store.connect() as conn:
            doc=store.document_access(conn,document,g.user)
            store.event(conn,doc['project_id'],g.user,'document.original_read',document,{'version':request.args.get('version')})
        return send_file(io.BytesIO(content), download_name=name, as_attachment=False,
                         mimetype=mimetypes.guess_type(name)[0] or 'application/octet-stream')

    @app.get('/api/v1/documents/<document>/pages/<int:number>')
    def page(document, number):
        return jsonify(store.page(document, number, g.user, request.args.get('version')))

    @app.get('/api/v1/documents/<document>/pages/<int:number>/image')
    def page_image(document,number):
        from .preview import render_page
        version=request.args.get('version')
        store.page(document,number,g.user,version)
        name,content=store.original(document,g.user,version)
        boxes=None
        block_id=request.args.get('block')
        if block_id:
            import json
            with store.connect() as conn:
                doc=store.document_access(conn,document,g.user)
                row=conn.execute('SELECT bbox FROM blocks WHERE id=? AND version_id=? AND page_index=?',(block_id,version or doc['active_version_id'],number-1)).fetchone()
                if row and row[0]:
                    boxes=json.loads(row[0])
        return send_file(io.BytesIO(render_page(name,content,number,boxes)),mimetype='image/png')

    @app.post('/api/v1/documents/<document>/pages/<int:number>/correction')
    def correction(document, number):
        payload = body()
        return jsonify(store.correct_page(document, g.user, number, payload.get('text'), payload.get('base_version'))), 201

    @app.delete('/api/v1/documents/<document>')
    def delete(document):
        store.delete(document, g.user)
        return jsonify(deleted=document)

    @app.post('/api/v1/documents/<document>/reprocess')
    def reprocess(document):
        return jsonify(store.reprocess(document, g.user)), 202

    @app.post('/api/v1/documents/<document>/reindex')
    def reindex(document):
        return jsonify(store.queue_reindex(document, g.user)), 202

    @app.get('/api/v1/documents/<document>/versions')
    def versions(document):
        with store.connect() as conn:
            store.document_access(conn, document, g.user)
            return jsonify([dict(r) for r in conn.execute('SELECT id,status,page_count,parser_version,created FROM versions WHERE document_id=? ORDER BY rowid DESC', (document,))])

    @app.patch('/api/v1/documents/<document>/metadata')
    def metadata(document):
        payload = body()
        from .registry import validate_metadata
        validate_metadata({k:v for k,v in payload.items() if k!='external_allowed'})
        if 'external_allowed' in payload and type(payload['external_allowed']) is not bool:
            raise ValueError('external_allowed должен быть boolean.')
        with store.connect() as conn:
            import json
            doc = store.document_access(conn, document, g.user, True)
            if 'external_allowed' in payload and store.project_access(conn, doc['project_id'], g.user) != 'owner':
                raise NotFound()
            previous = json.loads(doc['metadata'])
            updated = dict(previous, **{k: v for k, v in payload.items() if k != 'external_allowed'})
            validate_metadata(updated)
            external = payload.get('external_allowed', bool(doc['external_allowed']))
            conn.execute('UPDATE documents SET metadata=?,external_allowed=? WHERE id=?', (encoded(updated), int(external), document))
            conn.execute('UPDATE projects SET index_version=index_version+1 WHERE id=?', (doc['project_id'],))
            store.event(conn, doc['project_id'], g.user, 'document.metadata_changed', document,
                        {'before': previous, 'after': updated, 'external_before': bool(doc['external_allowed']), 'external_after': external})
        return jsonify(metadata=updated, external_allowed=external)

    @app.get('/api/v1/documents/<document>/registry')
    def registry(document):
        with store.connect() as conn:
            doc = store.document_access(conn,document,g.user)
            import json
            proposal=conn.execute('SELECT data FROM metadata_proposals WHERE version_id=?',(doc['active_version_id'],)).fetchone()
            blocks=[dict(b) for b in conn.execute('SELECT id,section_path,page_number,text_exact FROM blocks WHERE version_id=? ORDER BY ordinal',(doc['active_version_id'],))]
            return jsonify(metadata=json.loads(doc['metadata']),proposal=json.loads(proposal[0]) if proposal else {},blocks=blocks)

    @app.post('/api/v1/relations')
    def relation_create():
        from .registry import create_relation
        return jsonify(create_relation(store,g.user,body())),201

    @app.get('/api/v1/projects/<project>/relations')
    def relations(project):
        with store.connect() as conn:
            visible={d['id'] for d in store.documents(project,g.user,conn)}
            rows=conn.execute('SELECT r.* FROM document_relations r JOIN documents d ON d.id=r.source_document_id WHERE d.project_id=? AND r.status=?',(project,'active'))
            return jsonify([dict(r) for r in rows if r['source_document_id'] in visible and r['target_document_id'] in visible])

    @app.delete('/api/v1/relations/<relation>')
    def relation_revoke(relation):
        with store.connect() as conn:
            row=conn.execute('SELECT * FROM document_relations WHERE id=?',(relation,)).fetchone()
            if not row:
                raise NotFound()
            doc=store.document_access(conn,row['source_document_id'],g.user,True)
            store.document_access(conn,row['target_document_id'],g.user,True)
            conn.execute("UPDATE document_relations SET status='revoked' WHERE id=?",(relation,))
            conn.execute('UPDATE projects SET index_version=index_version+1 WHERE id=?',(doc['project_id'],))
            store.event(conn,doc['project_id'],g.user,'relation.revoked',relation,dict(row))
            return jsonify(id=relation,status='revoked')

    @app.get('/api/v1/jobs/<job_id>')
    def job(job_id):
        with store.connect() as conn:
            row = conn.execute('SELECT j.*,v.document_id,v.status processing_status FROM jobs j JOIN versions v ON v.id=j.version_id WHERE j.id=?', (job_id,)).fetchone()
            if not row:
                raise NotFound()
            store.document_access(conn, row['document_id'], g.user)
            return jsonify(dict(row))

    @app.post('/api/v1/projects/<project>/search')
    def search(project):
        payload = body()
        if set(payload) - {'query', 'top_k', 'document_ids','as_of','filters'}:
            raise ValueError('Допустимые поля: query, top_k, document_ids, as_of, filters.')
        if type(payload.get('top_k', 10)) is not int:
            raise ValueError('top_k: целое число от 1 до 50.')
        result = service.retriever.search(project, g.user, payload.get('query'),
                                         top_k=payload.get('top_k', 10), document_ids=payload.get('document_ids'),as_of=payload.get('as_of'),filters=payload.get('filters'))
        with store.connect() as conn:
            store.event(conn, project, g.user, 'search.completed', project,
                        {'index_version': result['index_version'], 'reranker': result['coverage_manifest']['reranker'],
                         'source_block_ids': [s['block_id'] for s in result['sources']]})
        return jsonify(result)

    @app.post('/api/v1/projects/<project>/questions')
    def ask(project):
        payload = body()
        use_llm = True if service.automatic_external else payload.get('use_llm', False)
        if type(use_llm) is not bool:
            raise ValueError('use_llm должен быть boolean.')
        if payload.get('async', True) is not False:
            return jsonify(questions.submit(project, g.user, payload.get('question'), use_llm,
                conversation_id=payload.get('conversation_id'), as_of=payload.get('as_of'),
                idempotency_key=request.headers.get('Idempotency-Key'),model_id=payload.get('model_id'))), 202
        return jsonify(service.ask(project, g.user, payload.get('question'), use_llm, as_of=payload.get('as_of'),
                                   model_id=payload.get('model_id'))), 201

    @app.get('/api/v1/projects/<project>/questions')
    def history(project):
        return jsonify(store.history(project, g.user))

    @app.get('/api/v1/questions/<answer_id>')
    def answer(answer_id):
        with store.connect() as conn:
            exists = conn.execute('SELECT 1 FROM question_jobs WHERE id=?', (answer_id,)).fetchone()
        if exists:
            return jsonify(questions.get(answer_id, g.user))
        return jsonify(store.answer(answer_id, g.user))

    @app.get('/api/v1/projects/<project>/question-jobs')
    def question_jobs(project):
        return jsonify(questions.list(project, g.user))

    @app.post('/api/v1/answers/<answer_id>/review')
    def review(answer_id):
        store.answer(answer_id, g.user)
        payload = body()
        verdict, comment = payload.get('verdict'), payload.get('comment', '')
        if verdict not in ('confirmed', 'rejected', 'needs_review', 'needs_documents') or not isinstance(comment, str) or len(comment) > 2000:
            raise ValueError('Некорректное заключение аудитора.')
        review_id = uid()
        with store.connect() as conn:
            project = conn.execute('SELECT project_id FROM answers WHERE id=?', (answer_id,)).fetchone()[0]
            store.project_access(conn, project, g.user, True)
            conn.execute('INSERT INTO reviews(id,answer_id,user_id,verdict,comment) VALUES(?,?,?,?,?)',
                         (review_id, answer_id, g.user, verdict, comment))
            store.event(conn, project, g.user, 'answer.reviewed', answer_id, {'review_id': review_id, 'verdict': verdict})
        return jsonify(id=review_id), 201

    @app.get('/api/v1/answers/<answer_id>/export')
    def export(answer_id):
        result = store.answer(answer_id, g.user)
        format = request.args.get('format', 'json')
        if format not in ('json', 'html', 'pdf'):
            raise ValueError('Формат экспорта: json, html или pdf.')
        with store.connect() as conn:
            result['reviews'] = [dict(r) for r in conn.execute('SELECT * FROM reviews WHERE answer_id=? ORDER BY created,id', (answer_id,))]
            project=conn.execute('SELECT project_id FROM answers WHERE id=?',(answer_id,)).fetchone()[0]
            store.event(conn,project,g.user,'answer.exported',answer_id,{'format':format})
        if format == 'pdf':
            from .reports import pdf_report
            return send_file(io.BytesIO(pdf_report(result)), mimetype='application/pdf',
                             as_attachment=True, download_name='audit-answer-' + answer_id + '.pdf')
        if request.args.get('format') == 'html':
            from .reports import html_report
            return send_file(io.BytesIO(html_report(result).encode()), mimetype='text/html',
                             as_attachment=True, download_name='audit-answer-' + answer_id + '.html')
        return send_file(io.BytesIO(encoded(result).encode()), mimetype='application/json',
                         as_attachment=True, download_name='audit-answer-' + answer_id + '.json')

    @app.get('/api/v1/openapi.json')
    def openapi():
        from .contract import specification
        return jsonify(specification(app))

    return app


def serve():
    from .embeddings import local_encoder
    from .runtime import configured_provider, configured_retriever
    mode = os.environ.get('BANK_RETRIEVAL_MODE', 'auto')
    if mode not in ('lexical', 'hybrid', 'auto'):
        raise RuntimeError('BANK_RETRIEVAL_MODE должен быть lexical, hybrid или auto.')
    directory = os.environ.get('EMBEDDING_MODEL_DIRECTORY') or str(ROOT / 'models/multilingual-e5-small')
    encoder = local_encoder(directory) if mode == 'hybrid' or (mode == 'auto' and Path(directory).is_dir()) else None
    store = Store(os.environ.get('BANK_DATABASE', str(ROOT / 'data/bank/audit.sqlite3')), encoder)
    with store.connect() as conn:
        persisted = conn.execute("SELECT value FROM settings WHERE key='external_llm_approved'").fetchone()
    approved = os.environ.get('EXTERNAL_LLM_APPROVED', persisted[0] if persisted else 'false').lower() == 'true'
    provider = configured_provider()
    app = create_app(store, service=AnswerService(store, provider, approved, configured_retriever(store, provider), automatic_external=True))
    if encoder:
        # Upgrade existing lexical-only documents without reparsing or changing canonical IDs.
        with store.connect() as conn:
            pending_indexes = conn.execute('''SELECT d.id,j.user_id FROM documents d
                JOIN versions v ON v.id=d.active_version_id JOIN jobs j ON j.version_id=v.id
                LEFT JOIN embedding_versions e ON e.version_id=v.id AND e.model_id=?
                WHERE d.deleted=0 AND v.status IN ('ready','ready_with_warnings')
                AND (e.status IS NULL OR e.status!='ready')''', (store.embedding_model,)).fetchall()
        for pending in pending_indexes:
            try:
                store.queue_reindex(pending['id'], pending['user_id'])
            except (NotFound, ValueError):
                pass
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    try:
        app.run(host='127.0.0.1', port=int(os.environ.get('BANK_API_PORT', '8766')), debug=False, use_reloader=False)
    finally:
        app.extensions['bank_worker'].close()


if __name__ == '__main__':
    serve()
