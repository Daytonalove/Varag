"""Persistent question orchestration on SQLite. One worker process owns execution."""
import json
from datetime import date
import sqlite3

from .store import NotFound, uid


class QuestionQueue:
    def __init__(self, store, service):
        self.store, self.service = store, service

    def submit(self, project, user, question, use_llm=True, conversation_id=None, as_of=None, idempotency_key=None):
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
            raise ValueError('Вопрос: 1–2000 символов.')
        if type(use_llm) is not bool:
            raise ValueError('use_llm должен быть boolean.')
        if as_of is not None:
            try:
                date.fromisoformat(as_of)
            except (ValueError, TypeError):
                raise ValueError('Дата проверки должна иметь формат YYYY-MM-DD.') from None
        key = idempotency_key or uid()
        if not isinstance(key, str) or not 1 <= len(key) <= 100:
            raise ValueError('Некорректный Idempotency-Key.')
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self.store.project_access(conn, project, user)
            existing = conn.execute('SELECT * FROM question_jobs WHERE project_id=? AND user_id=? AND idempotency_key=?', (project, user, key)).fetchone()
            if existing:
                if existing['question'] != question.strip() or existing['use_llm'] != int(use_llm) or existing['as_of'] != as_of or (conversation_id and existing['conversation_id'] != conversation_id):
                    raise ValueError('Idempotency-Key уже использован для другого запроса.')
                return dict(existing)
            if conversation_id:
                if not conn.execute('SELECT 1 FROM conversations WHERE id=? AND project_id=? AND user_id=?', (conversation_id, project, user)).fetchone():
                    raise NotFound()
            else:
                conversation_id = uid()
                conn.execute('INSERT INTO conversations(id,project_id,user_id,title) VALUES(?,?,?,?)', (conversation_id, project, user, question.strip()[:100]))
            if conn.execute("SELECT count(*) FROM question_jobs WHERE project_id=? AND user_id=? AND status IN ('queued','running')", (project, user)).fetchone()[0] >= 5:
                raise ValueError('Уже ожидают обработки пять вопросов. Дождитесь результата.')
            job_id = uid()
            conn.execute('INSERT INTO question_jobs(id,project_id,user_id,conversation_id,question,use_llm,as_of,idempotency_key) VALUES(?,?,?,?,?,?,?,?)',
                         (job_id, project, user, conversation_id, question.strip(), int(use_llm), as_of, key))
            return dict(id=job_id, conversation_id=conversation_id, status='queued')

    def get(self, job_id, user):
        with self.store.connect() as conn:
            row = conn.execute('SELECT * FROM question_jobs WHERE id=? AND user_id=?', (job_id, user)).fetchone()
            if not row:
                raise NotFound()
            self.store.project_access(conn, row['project_id'], user)
            result = dict(row)
        if result['status'] == 'done':
            result['answer'] = self.store.answer(job_id, user)
        return result

    def list(self, project, user):
        with self.store.connect() as conn:
            self.store.project_access(conn, project, user)
            rows = conn.execute('SELECT id FROM question_jobs WHERE project_id=? AND user_id=? ORDER BY rowid DESC LIMIT 100', (project, user)).fetchall()
        results = []
        for row in rows:
            try:
                results.append(self.get(row['id'], user))
            except NotFound:
                pass
        return results

    def recover(self):
        with self.store.connect() as conn:
            conn.execute("UPDATE question_jobs SET status='done',error=NULL WHERE id IN (SELECT id FROM answers)")
            conn.execute("UPDATE question_jobs SET status='failed',error='Выполнение прервано повторными перезапусками. Отправьте вопрос заново.' WHERE status='running' AND attempts>=2")
            conn.execute("UPDATE question_jobs SET status='queued' WHERE status='running'")

    def process_one(self):
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute("SELECT * FROM question_jobs WHERE status='queued' ORDER BY rowid LIMIT 1").fetchone()
            if not row:
                return False
            job = dict(row)
            conn.execute("UPDATE question_jobs SET status='running',attempts=attempts+1,updated=CURRENT_TIMESTAMP WHERE id=?", (job['id'],))
        try:
            self.service.ask(job['project_id'], job['user_id'], job['question'], bool(job['use_llm']),
                             conversation_id=job['conversation_id'], as_of=job['as_of'], answer_id=job['id'])
            with self.store.connect() as conn:
                conn.execute("UPDATE question_jobs SET status='done',error=NULL,updated=CURRENT_TIMESTAMP WHERE id=?", (job['id'],))
        except Exception as exc:
            # Only our public errors are retained; never serialize arbitrary provider/parser objects.
            message = str(exc) if type(exc) is ValueError else 'Не удалось обработать вопрос: ' + type(exc).__name__
            with self.store.connect() as conn:
                conn.execute("UPDATE question_jobs SET status='failed',error=?,updated=CURRENT_TIMESTAMP WHERE id=?", (message[:500], job['id']))
        return True
