"""SQLite development repository, with ACL enforced below the HTTP layer."""
from contextlib import contextmanager, closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import uuid
import os

from .ingest import PARSER_VERSION, SUPPORTED, extract_pages, extract_isolated, structure
from document_rag.extraction import chunks
from .embeddings import MODEL_ID
from .registry import rebuild_derived


def uid():
    return uuid.uuid4().hex


def encoded(value):
    return json.dumps(value, ensure_ascii=False)


class NotFound(Exception):
    """Identical response for missing and inaccessible resources."""


class Store:
    def __init__(self, database, encoder=None, embedding_model=MODEL_ID):
        self.database = Path(database)
        self.encoder, self.embedding_model = encoder, embedding_model
        self.embedding_lock = threading.Lock()
        self.claims_enabled = os.environ.get('ENABLE_CLAIMS_INDEX','false').lower() == 'true'
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(Path(__file__).with_name('schema.sql').read_text())
            conn.execute('BEGIN IMMEDIATE')
            if 'model_id' not in {row['name'] for row in conn.execute('PRAGMA table_info(question_jobs)')}:
                conn.execute('ALTER TABLE question_jobs ADD COLUMN model_id TEXT')
            conn.execute('INSERT OR IGNORE INTO schema_versions VALUES(6)')
            conn.execute('INSERT OR IGNORE INTO schema_versions VALUES(7)')
            from .norms import VERSION
            from .structural import indexed_version
            VERSION = indexed_version(VERSION)
            changed=set()
            for row in conn.execute('''SELECT DISTINCT b.version_id,d.project_id FROM blocks b
                JOIN versions v ON v.id=b.version_id JOIN documents d ON d.id=v.document_id
                LEFT JOIN derived_versions p ON p.version_id=b.version_id
                WHERE p.version_id IS NULL OR p.normalizer_version!=? OR p.claims_enabled!=?''',(VERSION,int(self.claims_enabled))).fetchall():
                rebuild_derived(conn,row[0],claims=self.claims_enabled)
                changed.add(row[1])
            for project in changed:
                conn.execute('UPDATE projects SET index_version=index_version+1 WHERE id=?',(project,))
                self.event(conn,project,'system','index.migrated',project,{'normalizer':VERSION,'claims_enabled':self.claims_enabled})

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.database, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON')
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def project_access(self, conn, project, user, write=False):
        member = conn.execute('SELECT role FROM members WHERE project_id=? AND user_id=?', (project, user)).fetchone()
        if not member or (write and member['role'] not in ('owner', 'auditor')):
            raise NotFound()
        return member['role']

    def document_access(self, conn, document, user, write=False):
        row = conn.execute('SELECT * FROM documents WHERE id=? AND deleted=0', (document,)).fetchone()
        if not row:
            raise NotFound()
        self.project_access(conn, row['project_id'], user, write)
        if row['restricted'] and not conn.execute('SELECT 1 FROM document_acl WHERE document_id=? AND user_id=?',
                                                 (document, user)).fetchone():
            raise NotFound()
        return dict(row)

    def event(self, conn, project, user, action, target, detail=None):
        conn.execute('INSERT INTO audit_events(project_id,user_id,action,target_id,detail) VALUES(?,?,?,?,?)',
                     (project, user, action, target, encoded(detail or {})))

    def create_project(self, name, user):
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 160:
            raise ValueError('Название проекта: 1–160 символов.')
        project = uid()
        with self.connect() as conn:
            conn.execute('INSERT INTO projects(id,name) VALUES(?,?)', (project, name.strip()))
            conn.execute('INSERT INTO members VALUES(?,?,?)', (project, user, 'owner'))
            self.event(conn, project, user, 'project.created', project)
        return {'id': project, 'name': name.strip()}

    def projects(self, user):
        with self.connect() as conn:
            return [dict(r) for r in conn.execute('SELECT p.* FROM projects p JOIN members m ON m.project_id=p.id WHERE m.user_id=?', (user,))]

    def documents(self, project, user, conn=None):
        if conn is None:
            with self.connect() as connection:
                return self.documents(project, user, connection)
        self.project_access(conn, project, user)
        rows = conn.execute('''SELECT d.*,v.status,v.page_count,v.warnings,v.digest,
            (SELECT error FROM jobs WHERE version_id=v.id) AS processing_error,
            (SELECT status FROM maintenance_jobs WHERE version_id=v.id ORDER BY rowid DESC LIMIT 1) AS reindex_status,
            (SELECT status FROM embedding_versions WHERE version_id=v.id AND model_id=?) AS semantic_status FROM documents d
            JOIN versions v ON v.id=d.active_version_id WHERE d.project_id=? AND d.deleted=0 AND
            (d.restricted=0 OR EXISTS(SELECT 1 FROM document_acl a WHERE a.document_id=d.id AND a.user_id=?))
            ORDER BY d.name,d.id''', (self.embedding_model, project, user))
        return [dict(r, warnings=json.loads(r['warnings']), metadata=json.loads(r['metadata'])) for r in rows]

    def upload(self, project, user, name, content, document_id=None):
        name = Path(name.replace('\\', '/')).name
        suffix = Path(name).suffix.lower()
        if not name or len(name) > 180 or suffix not in SUPPORTED:
            raise ValueError('Поддерживаются PDF, PNG, JPEG, TIFF, TXT и MD; имя до 180 символов.')
        if not content or len(content) > 25 * 1024 * 1024:
            raise ValueError('Размер файла: от 1 байта до 25 MiB.')
        if suffix == '.pdf' and not content.startswith(b'%PDF-'):
            raise ValueError('Содержимое не является PDF.')
        with self.connect() as conn:
            self.project_access(conn,project,user,True)
        from .operations import scan_upload
        av_status=scan_upload(content)
        digest, version, job = hashlib.sha256(content).hexdigest(), uid(), uid()
        with self.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self.project_access(conn, project, user, True)
            if document_id:
                doc = self.document_access(conn, document_id, user, True)
                if doc['project_id'] != project or Path(doc['name']).suffix.lower() != suffix:
                    raise ValueError('Версия должна принадлежать тому же проекту и формату.')
            else:
                # Dedup only within this user's accessible corpus, never reveal a hidden document.
                duplicate = next((d for d in self.documents(project, user, conn) if d['digest'] == digest), None)
                if duplicate:
                    return {'document_id': duplicate['id'], 'version_id': duplicate['active_version_id'], 'duplicate': True}
                document_id = uid()
                conn.execute('INSERT INTO documents(id,project_id,name) VALUES(?,?,?)', (document_id, project, name))
            prior = conn.execute('SELECT id FROM versions WHERE document_id=? AND digest=? AND parser_version=?',
                                 (document_id, digest, PARSER_VERSION)).fetchone()
            if prior:
                return {'document_id': document_id, 'version_id': prior['id'], 'duplicate': True}
            conn.execute('INSERT OR IGNORE INTO blobs VALUES(?,?)', (digest, content))
            conn.execute('INSERT INTO versions(id,document_id,digest,status,parser_version) VALUES(?,?,?,?,?)',
                         (version, document_id, digest, 'uploaded', PARSER_VERSION))
            conn.execute('UPDATE documents SET active_version_id=? WHERE id=?', (version, document_id))
            conn.execute('INSERT INTO jobs(id,version_id,user_id) VALUES(?,?,?)', (job, version, user))
            conn.execute('UPDATE projects SET index_version=index_version+1 WHERE id=?', (project,))
            self.event(conn, project, user, 'document.uploaded', document_id, {'version_id': version, 'sha256': digest, 'antivirus': av_status})
        return {'document_id': document_id, 'version_id': version, 'job_id': job, 'duplicate': False}

    def recover(self):
        """Call only after obtaining the exclusive worker lock."""
        with self.connect() as conn:
            conn.execute("UPDATE jobs SET status='failed',error='Превышено число восстановлений задания.' WHERE status='running' AND attempts>=3")
            conn.execute("UPDATE versions SET status='failed' WHERE id IN (SELECT version_id FROM jobs WHERE status='failed')")
            conn.execute("UPDATE jobs SET status='queued' WHERE status='running'")
            conn.execute("UPDATE maintenance_jobs SET status='queued' WHERE status='running'")

    def process_one(self, ocr=None):
        with self.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            job = conn.execute("SELECT j.*,d.id document_id,d.project_id,d.name FROM jobs j JOIN versions v ON v.id=j.version_id JOIN documents d ON d.id=v.document_id WHERE j.status='queued' ORDER BY j.rowid LIMIT 1").fetchone()
            if not job:
                return False
            job = dict(job)
            conn.execute("UPDATE jobs SET status='running',attempts=attempts+1,updated=CURRENT_TIMESTAMP WHERE id=?", (job['id'],))
        def stage(status):
            with self.connect() as conn:
                if status.startswith('progress:'):
                    completed,total = map(int,status.split(':',1)[1].split('/'))
                    conn.execute('INSERT OR REPLACE INTO processing_progress VALUES(?,?,?)',(job['version_id'],completed,total))
                    return
                conn.execute('UPDATE versions SET status=? WHERE id=?', (status, job['version_id']))
        try:
            with self.connect() as conn:
                self.document_access(conn, job['document_id'], job['user_id'], True)
                original = conn.execute('SELECT b.content FROM blobs b JOIN versions v ON b.digest=v.digest WHERE v.id=?', (job['version_id'],)).fetchone()[0]
            stage('validating')
            with tempfile.TemporaryDirectory(prefix='bank-ingest-') as temp:
                path = Path(temp) / ('original' + Path(job['name']).suffix.lower())
                path.write_bytes(original)
                pages = extract_pages(path, ocr, stage) if ocr is not None else extract_isolated(path, stage)
            stage('structuring')
            blocks = structure(pages, job['version_id'])
            warnings = [{'page_index': p['page_index'], 'page_number': p['page_number'], 'flags': p['quality_flags']}
                        for p in pages if p['quality_flags']]
            readable = any(b['text_exact'].strip() for b in blocks)
            status = ('ready_with_warnings' if warnings else 'ready') if readable else 'failed'
            stage('indexing')
            vector_rows, embedding_status = [], 'disabled'
            if self.encoder and readable:
                try:
                    segments = [(b['id'], i, text) for b in blocks for i, text in enumerate(chunks(b['text_exact'], 700)) if text.strip()]
                    with self.embedding_lock:
                        values = self.encoder.encode([s[2] for s in segments])
                    if len(values) != len(segments):
                        raise ValueError('Embedding count mismatch')
                    vector_rows = [(bid, self.embedding_model, index, encoded(value)) for (bid, index, _), value in zip(segments, values)]
                    embedding_status = 'ready'
                except Exception:
                    embedding_status = 'failed'
                    warnings.append({'flags': ['embedding_failed_lexical_available']})
                    status = 'ready_with_warnings'
            with self.connect() as conn:
                conn.execute('BEGIN IMMEDIATE')
                self.document_access(conn, job['document_id'], job['user_id'], True)
                # Atomic publish makes retries safe; no partial canonical state is committed.
                for page in pages:
                    if page.get('layout_data'):
                        conn.execute('INSERT OR REPLACE INTO page_layouts VALUES(?,?,?)',(job['version_id'],page['page_index'],encoded(page['layout_data'])))
                    conn.execute('INSERT OR IGNORE INTO pages VALUES(?,?,?,?,?,?)',
                                 (job['version_id'], page['page_index'], page['page_number'], page['text_exact'],
                                  page['extraction_method'], encoded(page['quality_flags'])))
                for block in blocks:
                    record = {k: encoded(v) if isinstance(v, (list,dict)) else v for k, v in block.items()}
                    conn.execute('INSERT OR IGNORE INTO blocks VALUES(' + ','.join('?' for _ in record) + ')', tuple(record.values()))
                conn.execute('DELETE FROM block_fts WHERE version_id=?', (job['version_id'],))
                conn.executemany('INSERT INTO block_fts VALUES(?,?,?)', [(b['id'], b['version_id'], b['text_normalized']) for b in blocks])
                rebuild_derived(conn,job['version_id'],claims=self.claims_enabled)
                conn.execute('INSERT OR REPLACE INTO embedding_versions VALUES(?,?,?)', (job['version_id'], self.embedding_model, embedding_status))
                conn.executemany('INSERT OR REPLACE INTO block_vectors VALUES(?,?,?,?)', vector_rows)
                conn.execute('UPDATE versions SET status=?,page_count=?,warnings=? WHERE id=?',
                             (status, len(pages), encoded(warnings), job['version_id']))
                conn.execute('UPDATE jobs SET status=?,error=?,updated=CURRENT_TIMESTAMP WHERE id=?',
                             ('done' if readable else 'failed', None if readable else 'Текст не извлечён.', job['id']))
                conn.execute('UPDATE projects SET index_version=index_version+1 WHERE id=?', (job['project_id'],))
                self.event(conn, job['project_id'], job['user_id'], 'document.processed', job['document_id'],
                           {'version_id': job['version_id'], 'status': status, 'pages': len(pages)})
        except Exception as exc:
            with self.connect() as conn:
                conn.execute("UPDATE versions SET status='failed',warnings=? WHERE id=?",
                             (encoded([{'flags': ['processing_failed']}]), job['version_id']))
                conn.execute("UPDATE jobs SET status='failed',error=?,updated=CURRENT_TIMESTAMP WHERE id=?",
                             ('Обработка не выполнена: ' + type(exc).__name__, job['id']))
                conn.execute('UPDATE projects SET index_version=index_version+1 WHERE id=?', (job['project_id'],))
        return True

    def reprocess(self, document, user):
        """Always a new immutable extraction version, even for identical original bytes."""
        with self.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            doc = self.document_access(conn, document, user, True)
            previous = conn.execute('SELECT * FROM versions WHERE id=?', (doc['active_version_id'],)).fetchone()
            if previous['status'] not in ('ready', 'ready_with_warnings', 'failed'):
                raise ValueError('Документ уже обрабатывается.')
            version, job = uid(), uid()
            conn.execute('INSERT INTO versions(id,document_id,digest,status,parser_version) VALUES(?,?,?,?,?)',
                         (version, document, previous['digest'], 'uploaded', PARSER_VERSION + ':retry:' + version))
            conn.execute('UPDATE documents SET active_version_id=? WHERE id=?', (version, document))
            conn.execute('INSERT INTO jobs(id,version_id,user_id) VALUES(?,?,?)', (job, version, user))
            conn.execute('UPDATE projects SET index_version=index_version+1 WHERE id=?', (doc['project_id'],))
            self.event(conn, doc['project_id'], user, 'document.reprocessed', document, {'previous': previous['id'], 'version_id': version})
            return {'job_id': job, 'version_id': version, 'document_id': document}

    def queue_reindex(self, document, user):
        with self.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            doc = self.document_access(conn, document, user, True)
            version = conn.execute('SELECT status FROM versions WHERE id=?', (doc['active_version_id'],)).fetchone()
            if version['status'] not in ('ready', 'ready_with_warnings'):
                raise ValueError('Сначала дождитесь извлечения текста.')
            prior = conn.execute("SELECT id FROM maintenance_jobs WHERE version_id=? AND model_id=? AND status IN ('queued','running')", (doc['active_version_id'], self.embedding_model)).fetchone()
            job = prior['id'] if prior else uid()
            if not prior:
                conn.execute('INSERT INTO maintenance_jobs(id,version_id,user_id,model_id) VALUES(?,?,?,?)', (job, doc['active_version_id'], user, self.embedding_model))
                self.event(conn, doc['project_id'], user, 'index.queued', document, {'job_id': job})
            return {'id': job, 'status': 'queued'}

    def process_maintenance(self):
        with self.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute("SELECT j.*,v.document_id,d.project_id FROM maintenance_jobs j JOIN versions v ON v.id=j.version_id JOIN documents d ON d.id=v.document_id WHERE j.status='queued' ORDER BY j.rowid LIMIT 1").fetchone()
            if not row:
                return False
            job = dict(row)
            conn.execute("UPDATE maintenance_jobs SET status='running',updated=CURRENT_TIMESTAMP WHERE id=?", (job['id'],))
        try:
            if job['model_id'] != self.embedding_model:
                raise ValueError('Модель индекса недоступна или изменилась.')
            with self.connect() as conn:
                self.document_access(conn, job['document_id'], job['user_id'], True)
                blocks = conn.execute('SELECT id,text_exact FROM blocks WHERE version_id=? ORDER BY ordinal', (job['version_id'],)).fetchall()
            segments = [(b['id'], i, t) for b in blocks for i, t in enumerate(chunks(b['text_exact'], 700)) if t.strip()]
            with self.embedding_lock:
                vectors = self.encoder.encode([s[2] for s in segments]) if self.encoder else []
            if self.encoder and len(vectors) != len(segments):
                raise ValueError('Неполный результат embedding.')
            with self.connect() as conn:
                conn.execute('BEGIN IMMEDIATE')
                self.document_access(conn, job['document_id'], job['user_id'], True)
                conn.execute('DELETE FROM block_vectors WHERE model_id=? AND block_id IN (SELECT id FROM blocks WHERE version_id=?)', (self.embedding_model, job['version_id']))
                conn.executemany('INSERT INTO block_vectors VALUES(?,?,?,?)', [(bid, self.embedding_model, i, encoded(v)) for (bid, i, _), v in zip(segments, vectors)])
                conn.execute('INSERT OR REPLACE INTO embedding_versions VALUES(?,?,?)', (job['version_id'], self.embedding_model, 'ready' if self.encoder else 'disabled'))
                rebuild_derived(conn,job['version_id'],claims=self.claims_enabled)
                conn.execute("UPDATE maintenance_jobs SET status='done',error=NULL,updated=CURRENT_TIMESTAMP WHERE id=?", (job['id'],))
                conn.execute('UPDATE projects SET index_version=index_version+1 WHERE id=?', (job['project_id'],))
                self.event(conn, job['project_id'], job['user_id'], 'index.rebuilt', job['document_id'], {'version_id': job['version_id']})
        except Exception as exc:
            with self.connect() as conn:
                conn.execute("UPDATE maintenance_jobs SET status='failed',error=?,updated=CURRENT_TIMESTAMP WHERE id=?", ('Ошибка индекса: ' + type(exc).__name__, job['id']))
        return True

    def backup(self, target):
        target = Path(target)
        if target.exists() or target.resolve() == self.database.resolve():
            raise ValueError('Укажите новый файл резервной копии.')
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as source, closing(sqlite3.connect(target)) as destination:
            source.backup(destination)

    def correct_page(self, document, user, page_number, text, base_version):
        if type(page_number) is not int or page_number < 1 or not isinstance(text, str) or not 1 <= len(text.strip()) <= 100000:
            raise ValueError('Нужны номер страницы и исправленный текст (до 100000 символов).')
        with self.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            doc = self.document_access(conn, document, user, True)
            if doc['active_version_id'] != base_version:
                raise ValueError('Редакция изменилась. Откройте актуальную страницу заново.')
            old = conn.execute('SELECT * FROM versions WHERE id=?', (base_version,)).fetchone()
            if old['status'] not in ('ready', 'ready_with_warnings'):
                raise ValueError('Исправление доступно после обработки документа.')
            pages = [dict(r) for r in conn.execute('SELECT * FROM pages WHERE version_id=? ORDER BY page_index', (base_version,))]
            if page_number > len(pages):
                raise NotFound()
            for page in pages:
                page['quality_flags'] = json.loads(page['quality_flags'])
            pages[page_number - 1].update(text_exact=text, extraction_method='manual_correction', quality_flags=[])
            version = uid()
            warnings = [{'page_index': p['page_index'], 'page_number': p['page_number'], 'flags': p['quality_flags']} for p in pages if p['quality_flags']]
            conn.execute('INSERT INTO versions(id,document_id,digest,status,page_count,warnings,parser_version) VALUES(?,?,?,?,?,?,?)',
                (version, document, old['digest'], 'ready_with_warnings' if warnings else 'ready', len(pages), encoded(warnings), PARSER_VERSION + ':correction:' + version))
            for page in pages:
                conn.execute('INSERT INTO pages VALUES(?,?,?,?,?,?)', (version, page['page_index'], page['page_number'], page['text_exact'], page['extraction_method'], encoded(page['quality_flags'])))
            blocks = structure(pages, version)
            for block in blocks:
                record = {k: encoded(v) if isinstance(v, (list,dict)) else v for k, v in block.items()}
                conn.execute('INSERT INTO blocks VALUES(' + ','.join('?' for _ in record) + ')', tuple(record.values()))
            conn.executemany('INSERT INTO block_fts VALUES(?,?,?)', [(b['id'], version, b['text_normalized']) for b in blocks])
            rebuild_derived(conn,version,claims=self.claims_enabled)
            conn.execute('UPDATE documents SET active_version_id=? WHERE id=?', (version, document))
            conn.execute('UPDATE projects SET index_version=index_version+1 WHERE id=?', (doc['project_id'],))
            self.event(conn, doc['project_id'], user, 'page.corrected', document, {'previous_version': base_version, 'version_id': version, 'page_number': page_number})
            if self.encoder:
                conn.execute('INSERT INTO maintenance_jobs(id,version_id,user_id,model_id) VALUES(?,?,?,?)', (uid(), version, user, self.embedding_model))
            return {'document_id': document, 'version_id': version, 'page_number': page_number}

    def original(self, document, user, version=None):
        with self.connect() as conn:
            doc = self.document_access(conn, document, user)
            row = conn.execute('SELECT b.content FROM versions v JOIN blobs b ON b.digest=v.digest WHERE v.id=? AND v.document_id=?',
                               (version or doc['active_version_id'], document)).fetchone()
            if not row:
                raise NotFound()
            self.event(conn, doc['project_id'], user, 'document.viewed', document)
            return doc['name'], row[0]

    def page(self, document, number, user, version=None):
        with self.connect() as conn:
            doc = self.document_access(conn, document, user)
            version = version or doc['active_version_id']
            if not conn.execute('SELECT 1 FROM versions WHERE id=? AND document_id=?', (version, document)).fetchone():
                raise NotFound()
            page = conn.execute('SELECT * FROM pages WHERE version_id=? AND page_index=?', (version, number - 1)).fetchone()
            if not page:
                raise NotFound()
            blocks = [dict(r) for r in conn.execute('SELECT * FROM blocks WHERE version_id=? AND page_index=? ORDER BY ordinal', (version, number - 1))]
            return dict(page, blocks=blocks, quality_flags=json.loads(page['quality_flags']))

    def delete(self, document, user):
        with self.connect() as conn:
            doc = self.document_access(conn, document, user, True)
            conn.execute('UPDATE documents SET deleted=1 WHERE id=?', (document,))
            conn.execute('UPDATE projects SET index_version=index_version+1 WHERE id=?', (doc['project_id'],))
            self.event(conn, doc['project_id'], user, 'document.deleted', document)

    def answer(self, answer_id, user):
        with self.connect() as conn:
            row = conn.execute('SELECT * FROM answers WHERE id=?', (answer_id,)).fetchone()
            if not row:
                raise NotFound()
            self.project_access(conn, row['project_id'], user)
            result = json.loads(row['result'])
            # Historical coverage itself contains IDs; deny entire snapshot if any access changed.
            for document in result['coverage_manifest']['accessible_document_ids']:
                self.document_access(conn, document, user)
            return dict(result, question=row['question'])

    def history(self, project, user):
        with self.connect() as conn:
            self.project_access(conn, project, user)
            ids = [r[0] for r in conn.execute('SELECT id FROM answers WHERE project_id=? ORDER BY created,rowid', (project,))]
        results = []
        for answer_id in ids:
            try:
                results.append(self.answer(answer_id, user))
            except NotFound:
                pass
        return results
