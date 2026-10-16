"""Local administration, explicit retention and operational accounting."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import socket
import struct


def scan_upload(content):
    """ClamAV INSTREAM. Configured scanner failure never means clean."""
    host = os.environ.get('BANK_CLAMAV_HOST')
    if not host:
        if os.environ.get('BANK_REQUIRE_AV','false').lower()=='true':
            raise ValueError('Антивирус обязателен, но BANK_CLAMAV_HOST не настроен.')
        return 'not_configured'
    try:
        with socket.create_connection((host,int(os.environ.get('BANK_CLAMAV_PORT','3310'))), timeout=15) as connection:
            connection.sendall(b'zINSTREAM\0')
            for start in range(0,len(content),65536):
                part=content[start:start+65536]
                connection.sendall(struct.pack('!I',len(part))+part)
            connection.sendall(struct.pack('!I',0))
            response=b''
            while b'\0' not in response and len(response)<4096:
                part=connection.recv(1024)
                if not part:
                    break
                response+=part
        if response.rstrip(b'\0\n')!=b'stream: OK':
            raise ValueError('Файл отклонён антивирусной проверкой.')
    except (OSError,ValueError) as exc:
        raise ValueError('Антивирус не подтвердил безопасность файла; загрузка остановлена.') from exc
    return 'clean'


def owner(store,conn,project,user):
    from .store import NotFound
    if store.project_access(conn,project,user)!='owner':
        raise NotFound()


def set_member(store,project,user,target,role):
    if not isinstance(target,str) or not 1<=len(target)<=160 or role not in ('owner','auditor','viewer',None):
        raise ValueError('Нужны user_id и роль owner/auditor/viewer; null удаляет участника.')
    with store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        owner(store,conn,project,user)
        prior=conn.execute('SELECT role FROM members WHERE project_id=? AND user_id=?',(project,target)).fetchone()
        count=conn.execute("SELECT count(*) FROM members WHERE project_id=? AND role='owner'",(project,)).fetchone()[0]
        if prior and prior[0]=='owner' and role!='owner' and count<=1:
            raise ValueError('В проекте должен остаться хотя бы один владелец.')
        if role:
            conn.execute('INSERT OR REPLACE INTO members VALUES(?,?,?)',(project,target,role))
        else:
            conn.execute('DELETE FROM members WHERE project_id=? AND user_id=?',(project,target))
        conn.execute('UPDATE projects SET index_version=index_version+1 WHERE id=?',(project,))
        store.event(conn,project,user,'member.changed',target,{'role':role})


def set_acl(store,document,user,payload):
    if set(payload)!={'restricted','user_ids'} or type(payload['restricted']) is not bool or not isinstance(payload['user_ids'],list) or any(not isinstance(v,str) for v in payload['user_ids']):
        raise ValueError('Ожидаются restricted:boolean и user_ids:list[str].')
    with store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        doc=store.document_access(conn,document,user,True)
        owner(store,conn,doc['project_id'],user)
        members={r[0] for r in conn.execute('SELECT user_id FROM members WHERE project_id=?',(doc['project_id'],))}
        allowed=set(payload['user_ids'])
        if not allowed<=members or (payload['restricted'] and user not in allowed):
            raise ValueError('Разрешены участники проекта; текущий владелец должен сохранить доступ.')
        conn.execute('DELETE FROM document_acl WHERE document_id=?',(document,))
        conn.executemany('INSERT INTO document_acl VALUES(?,?)',[(document,v) for v in allowed])
        conn.execute('UPDATE documents SET restricted=? WHERE id=?',(int(payload['restricted']),document))
        conn.execute('UPDATE projects SET index_version=index_version+1 WHERE id=?',(doc['project_id'],))
        store.event(conn,doc['project_id'],user,'document.acl_changed',document,payload)


def metrics(store,project,user):
    with store.connect() as conn:
        owner(store,conn,project,user)
        visible={d['id'] for d in store.documents(project,user,conn)}
        blocks={r[0] for r in conn.execute('SELECT b.id,v.document_id FROM blocks b JOIN versions v ON v.id=b.version_id') if r[1] in visible}
        calls=[dict(r) for r in conn.execute('SELECT * FROM llm_calls WHERE project_id=?',(project,)) if set(json.loads(r['source_block_ids']))<=blocks]
        failures=sum(r['status'] not in ('started','succeeded') for r in calls)
        tokens=0
        elapsed=[]
        for row in calls:
            meta=json.loads(row['metadata'])
            tokens+=meta.get('usage',{}).get('total_tokens',0)
            if 'elapsed_ms' in meta:
                elapsed.append(meta['elapsed_ms'])
        price=os.environ.get('GIGACHAT_PRICE_PER_MILLION_RUB')
        return dict(calls=len(calls),failed_calls=failures,tokens=tokens,
                    estimated_cost_rub=round(tokens*float(price)/1e6,4) if price else None,
                    price_source='operator_configuration' if price else 'not_configured',
                    average_latency_ms=round(sum(elapsed)/len(elapsed)) if elapsed else None,
                    documents=len(visible),antivirus='configured' if os.environ.get('BANK_CLAMAV_HOST') else 'not_configured',
                    storage_encryption='external_volume_required',sso='deferred_by_user')


def encrypted_backup(store,target,key_file):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    key=Path(key_file).read_bytes()
    if len(key)!=32:
        raise ValueError('Ключ резервной копии должен содержать 32 случайных байта.')
    nonce=os.urandom(12)
    with store.connect() as conn:
        content=conn.serialize()
    # serialize() includes committed WAL pages, but retains the WAL-mode header.
    # Make the standalone image readable without a companion WAL file.
    content=content[:18]+b'\x01\x01'+content[20:]
    payload=b'BANKBACKUP1'+nonce+AESGCM(key).encrypt(nonce,content,b'BANKBACKUP1')
    with open(target,'xb') as out:
        os.chmod(target,0o600)
        out.write(payload)


def restore_encrypted_backup(source,target,key_file):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    value=Path(source).read_bytes()
    if not value.startswith(b'BANKBACKUP1'):
        raise ValueError('Неверный формат резервной копии.')
    content=AESGCM(Path(key_file).read_bytes()).decrypt(value[11:23],value[23:],b'BANKBACKUP1')
    import sqlite3
    conn=sqlite3.connect(':memory:')
    try:
        conn.deserialize(content)
        if conn.execute('PRAGMA integrity_check').fetchone()[0]!='ok':
            raise ValueError('Повреждённая резервная копия.')
    finally:
        conn.close()
    with open(target,'xb') as out:
        os.chmod(target,0o600)
        out.write(content)


def retention(store,project,user,days=365,execute=False):
    """Purge only soft-deleted documents older than policy, including derived copies.

    Append-only audit metadata remains intentionally, originals/answers do not.
    No automatic purge runs on startup or migration.
    """
    if type(days) is not int or not 1<=days<=36500 or type(execute) is not bool:
        raise ValueError('days: 1–36500; execute:boolean.')
    cutoff=(datetime.now(timezone.utc)-timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
    with store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        owner(store,conn,project,user)
        docs=[r[0] for r in conn.execute("SELECT d.id FROM documents d WHERE project_id=? AND deleted=1 AND EXISTS(SELECT 1 FROM audit_events a WHERE a.target_id=d.id AND a.action='document.deleted' AND a.created<?)",(project,cutoff))]
        if not execute:
            return dict(document_ids=docs,cutoff=cutoff,executed=False)
        for document in docs:
            for answer in conn.execute('SELECT id,result FROM answers WHERE project_id=?',(project,)).fetchall():
                if document in json.loads(answer['result'])['coverage_manifest']['accessible_document_ids']:
                    conn.execute('DELETE FROM reviews WHERE answer_id=?',(answer['id'],))
                    conn.execute('DELETE FROM answers WHERE id=?',(answer['id'],))
                    conn.execute('DELETE FROM llm_calls WHERE question_id=?',(answer['id'],))
                    conn.execute('DELETE FROM question_jobs WHERE id=?',(answer['id'],))
            conn.execute('DELETE FROM document_relations WHERE source_document_id=? OR target_document_id=?',(document,document))
            versions=[r[0] for r in conn.execute('SELECT id FROM versions WHERE document_id=?',(document,))]
            for version in versions:
                conn.execute('DELETE FROM block_vectors WHERE block_id IN(SELECT id FROM blocks WHERE version_id=?)',(version,))
                for table in ('block_fts','claims','exact_terms','block_links','metadata_proposals','structural_keys','derived_versions','processing_progress','page_layouts','pages','blocks','jobs','maintenance_jobs','embedding_versions'):
                    conn.execute(f'DELETE FROM {table} WHERE version_id=?',(version,))
            conn.execute('DELETE FROM versions WHERE document_id=?',(document,))
            conn.execute('DELETE FROM document_acl WHERE document_id=?',(document,))
            conn.execute('DELETE FROM documents WHERE id=?',(document,))
            store.event(conn,project,user,'document.purged',document,{'retention_days':days})
        conn.execute('DELETE FROM blobs WHERE digest NOT IN(SELECT digest FROM versions)')
        if docs:
            conn.execute('UPDATE projects SET index_version=index_version+1 WHERE id=?',(project,))
        return dict(document_ids=docs,cutoff=cutoff,executed=True,physical_erasure='requires_encrypted_volume_and_backup_retention')
