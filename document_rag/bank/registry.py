"""Typed, audited metadata and document relationships, independent of the LLM."""
from datetime import date
import json
import re
from .norms import periods, facts, DATE, iso_date, VERSION

TEXT_FIELDS = {'title','document_type','document_number','revision','approving_body','department','language','upload_source','product','status'}
DATE_FIELDS = {'approved_at','effective_from','effective_until'}
FIELDS = TEXT_FIELDS | DATE_FIELDS | {'tags','verification_status'}
RELATIONS = {'amends','repeals','replaces','supplements','references','exception'}


def validate_metadata(payload):
    if not isinstance(payload, dict) or set(payload) - FIELDS:
        raise ValueError('Неизвестные поля метаданных.')
    for key, value in payload.items():
        if key in TEXT_FIELDS and (not isinstance(value, str) or len(value) > 500):
            raise ValueError('Текстовое поле метаданных: максимум 500 символов.')
        if key in DATE_FIELDS and value is not None:
            try:
                date.fromisoformat(value)
            except (TypeError, ValueError):
                raise ValueError('Дата метаданных должна иметь формат YYYY-MM-DD.') from None
        if key == 'tags' and (not isinstance(value,list) or len(value)>30 or any(not isinstance(t,str) or len(t)>80 for t in value)):
            raise ValueError('Метки: список до 30 строк длиной до 80 символов.')
        if key == 'verification_status' and value not in ('proposed','confirmed','rejected'):
            raise ValueError('Статус подтверждения: proposed, confirmed или rejected.')
    if payload.get('effective_from') and payload.get('effective_until') and payload['effective_from'] > payload['effective_until']:
        raise ValueError('Конец действия не может предшествовать началу.')
    return payload


def filter_documents(docs, filters):
    validate_metadata(filters)
    def matches(doc):
        metadata = doc['metadata']
        for key, value in filters.items():
            if key == 'tags':
                if not set(value) <= set(metadata.get('tags', [])):
                    return False
            elif metadata.get(key) != value:
                return False
        return True
    return [d for d in docs if matches(d)]


def rebuild_derived(conn, version, *, claims=False):
    """Rebuild ALL derived text/normalized/claim/link indices from canonical text."""
    blocks = [dict(b) for b in conn.execute('SELECT * FROM blocks WHERE version_id=? ORDER BY ordinal', (version,))]
    conn.execute('DELETE FROM block_fts WHERE version_id=?', (version,))
    conn.executemany('INSERT INTO block_fts VALUES(?,?,?)', [(b['id'],version,b['text_normalized']) for b in blocks])
    for table in ('exact_terms','claims','block_links'):
        conn.execute(f'DELETE FROM {table} WHERE version_id=?', (version,))
    for i,b in enumerate(blocks):
        extracted = facts(b['text_exact'])
        terms = [(f['kind'],f['value'],f['unit']) for f in extracted]
        for raw in DATE.findall(b['text_exact']):
            try:
                terms.append(('date',iso_date(raw),''))
            except ValueError:
                pass
        if b['section_path']:
            terms.append(('clause',b['section_path'],''))
        conn.executemany('INSERT OR IGNORE INTO exact_terms VALUES(?,?,?,?,?)', [(version,b['id'],*t) for t in terms])
        if claims:
            conn.executemany('INSERT INTO claims(version_id,block_id,extractor_version,data) VALUES(?,?,?,?)',
                             [(version,b['id'],VERSION,json.dumps(f,ensure_ascii=False)) for f in extracted])
        related = [(b['parent_block_id'],'parent')] if b['parent_block_id'] else []
        related += [(blocks[n]['id'],'neighbor') for n in (i-1,i+1) if 0<=n<len(blocks)]
        refs = re.findall(r'(?:пункт\w*|п\.)\s*(\d+(?:\.\d+)*)', b['text_exact'], re.I)
        related += [(t['id'],'reference') for t in blocks if t['section_path'] in refs and t['id']!=b['id']]
        conn.executemany('INSERT OR IGNORE INTO block_links VALUES(?,?,?,?)', [(version,b['id'],target,kind) for target,kind in related])
    text = '\n'.join(b['text_exact'] for b in blocks)
    proposal = {'title':text.split('\n')[0][:160], 'language':'ru' if re.search('[а-яА-Я]',text) else 'und', 'verification_status':'proposed'}
    period = periods(text[:12000])
    proposal.update({k:v for k,v in [('effective_from',period['from']),('effective_until',period['until'])] if v})
    number = re.search(r'№\s*([\w/-]+)',text[:3000])
    if number:
        proposal['document_number']=number[1]
    conn.execute('INSERT OR REPLACE INTO metadata_proposals VALUES(?,?,?)', (version,json.dumps(proposal,ensure_ascii=False),VERSION))
    conn.execute('INSERT OR REPLACE INTO derived_versions VALUES(?,?,?)',(version,VERSION,int(claims)))


def create_relation(store, user, payload):
    from .store import uid, encoded
    required = {'source_document_id','target_document_id','kind','source_block_id','effective_from','confirmed'}
    if not isinstance(payload,dict) or set(payload)!=required or payload['kind'] not in RELATIONS or type(payload['confirmed']) is not bool:
        raise ValueError('Некорректная связь документов.')
    if payload['source_document_id']==payload['target_document_id']:
        raise ValueError('Нужны два разных документа.')
    validate_metadata({'effective_from':payload['effective_from']})
    with store.connect() as conn:
        source = store.document_access(conn,payload['source_document_id'],user,True)
        target = store.document_access(conn,payload['target_document_id'],user,True)
        if source['project_id']!=target['project_id']:
            raise ValueError('Документы должны принадлежать одному проекту.')
        block=conn.execute('SELECT b.id FROM blocks b JOIN versions v ON v.id=b.version_id WHERE b.id=? AND v.document_id=?',
                           (payload['source_block_id'],source['id'])).fetchone()
        if not block:
            raise ValueError('Пункт-основание должен существовать в исходном документе.')
        relation=uid()
        conn.execute('INSERT INTO document_relations VALUES(?,?,?,?,?,?,?,?,?)',
                     (relation,source['id'],target['id'],payload['kind'],payload['source_block_id'],payload['effective_from'],int(payload['confirmed']),user,'active'))
        conn.execute('UPDATE projects SET index_version=index_version+1 WHERE id=?',(source['project_id'],))
        store.event(conn,source['project_id'],user,'relation.created',relation,payload)
        return dict(id=relation,**payload)
