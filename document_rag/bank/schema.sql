PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS schema_versions(version INTEGER PRIMARY KEY);
INSERT OR IGNORE INTO schema_versions VALUES(1);
INSERT OR IGNORE INTO schema_versions VALUES(2);
INSERT OR IGNORE INTO schema_versions VALUES(3);
CREATE TABLE IF NOT EXISTS projects(
 id TEXT PRIMARY KEY, name TEXT NOT NULL, index_version INTEGER NOT NULL DEFAULT 0,
 created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS members(
 project_id TEXT REFERENCES projects(id), user_id TEXT, role TEXT NOT NULL,
 PRIMARY KEY(project_id,user_id));
CREATE TABLE IF NOT EXISTS documents(
 id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id), name TEXT NOT NULL,
 active_version_id TEXT, deleted INTEGER NOT NULL DEFAULT 0,
 external_allowed INTEGER NOT NULL DEFAULT 0, restricted INTEGER NOT NULL DEFAULT 0,
 metadata TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS document_acl(
 document_id TEXT REFERENCES documents(id), user_id TEXT, PRIMARY KEY(document_id,user_id));
CREATE TABLE IF NOT EXISTS blobs(digest TEXT PRIMARY KEY, content BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS versions(
 id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id),
 digest TEXT NOT NULL REFERENCES blobs(digest), status TEXT NOT NULL,
 page_count INTEGER NOT NULL DEFAULT 0, warnings TEXT NOT NULL DEFAULT '[]',
 parser_version TEXT NOT NULL, created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(document_id,digest,parser_version));
CREATE TABLE IF NOT EXISTS jobs(
 id TEXT PRIMARY KEY, version_id TEXT NOT NULL UNIQUE REFERENCES versions(id),
 user_id TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued', attempts INTEGER NOT NULL DEFAULT 0,
 error TEXT, updated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS pages(
 version_id TEXT REFERENCES versions(id), page_index INTEGER, page_number TEXT NOT NULL,
 text_exact TEXT NOT NULL, extraction_method TEXT NOT NULL, quality_flags TEXT NOT NULL,
 PRIMARY KEY(version_id,page_index));
CREATE TABLE IF NOT EXISTS blocks(
 id TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES versions(id), ordinal INTEGER NOT NULL,
 page_index INTEGER NOT NULL, page_number TEXT NOT NULL, section_path TEXT NOT NULL,
 parent_block_id TEXT, block_type TEXT NOT NULL, text_exact TEXT NOT NULL,
 text_normalized TEXT NOT NULL, bbox TEXT, extraction_method TEXT NOT NULL,
 quality_flags TEXT NOT NULL, content_hash TEXT NOT NULL, parser_version TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS blocks_version ON blocks(version_id,ordinal);
CREATE VIRTUAL TABLE IF NOT EXISTS block_fts USING fts5(block_id UNINDEXED,version_id UNINDEXED,text);
CREATE TABLE IF NOT EXISTS embedding_versions(
 version_id TEXT REFERENCES versions(id), model_id TEXT, status TEXT NOT NULL,
 PRIMARY KEY(version_id,model_id));
CREATE TABLE IF NOT EXISTS block_vectors(
 block_id TEXT REFERENCES blocks(id), model_id TEXT, chunk_index INTEGER, value TEXT NOT NULL,
 PRIMARY KEY(block_id,model_id,chunk_index));
CREATE TABLE IF NOT EXISTS answers(
 id TEXT PRIMARY KEY, project_id TEXT REFERENCES projects(id), user_id TEXT NOT NULL,
 question TEXT NOT NULL, result TEXT NOT NULL, created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS reviews(
 id TEXT PRIMARY KEY, answer_id TEXT REFERENCES answers(id), user_id TEXT NOT NULL,
 verdict TEXT NOT NULL, comment TEXT NOT NULL, created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS audit_events(
 id INTEGER PRIMARY KEY, project_id TEXT NOT NULL, user_id TEXT NOT NULL, action TEXT NOT NULL,
 target_id TEXT NOT NULL, detail TEXT NOT NULL, created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_events
 BEGIN SELECT RAISE(ABORT,'append-only audit'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_events
 BEGIN SELECT RAISE(ABORT,'append-only audit'); END;
CREATE TRIGGER IF NOT EXISTS answers_no_update BEFORE UPDATE ON answers
 BEGIN SELECT RAISE(ABORT,'immutable answer'); END;
CREATE TRIGGER IF NOT EXISTS blocks_no_update BEFORE UPDATE ON blocks
 BEGIN SELECT RAISE(ABORT,'immutable block'); END;
CREATE TABLE IF NOT EXISTS conversations(
 id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id), user_id TEXT NOT NULL,
 title TEXT NOT NULL, created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS question_jobs(
 id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id), user_id TEXT NOT NULL,
 conversation_id TEXT NOT NULL REFERENCES conversations(id), question TEXT NOT NULL,
 use_llm INTEGER NOT NULL, as_of TEXT, status TEXT NOT NULL DEFAULT 'queued',
 attempts INTEGER NOT NULL DEFAULT 0, error TEXT, idempotency_key TEXT NOT NULL,
 created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, updated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(project_id,user_id,idempotency_key));
CREATE TABLE IF NOT EXISTS maintenance_jobs(
 id TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES versions(id), user_id TEXT NOT NULL,
 model_id TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued', error TEXT,
 updated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_maintenance ON maintenance_jobs(version_id,model_id)
 WHERE status IN ('queued','running');
CREATE TABLE IF NOT EXISTS llm_calls(
 id TEXT PRIMARY KEY, project_id TEXT NOT NULL, user_id TEXT NOT NULL, question_id TEXT,
 stage TEXT NOT NULL, source_block_ids TEXT NOT NULL, model TEXT NOT NULL,
 status TEXT NOT NULL, metadata TEXT NOT NULL, created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
INSERT OR IGNORE INTO schema_versions VALUES(4);
CREATE TABLE IF NOT EXISTS metadata_proposals(version_id TEXT PRIMARY KEY REFERENCES versions(id), data TEXT NOT NULL, extractor_version TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS document_relations(id TEXT PRIMARY KEY, source_document_id TEXT REFERENCES documents(id),
 target_document_id TEXT REFERENCES documents(id), kind TEXT NOT NULL, source_block_id TEXT REFERENCES blocks(id),
 effective_from TEXT, confirmed INTEGER NOT NULL DEFAULT 0, user_id TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active');
CREATE TABLE IF NOT EXISTS exact_terms(version_id TEXT,block_id TEXT,kind TEXT,value TEXT,unit TEXT,
 PRIMARY KEY(version_id,block_id,kind,value,unit));
CREATE INDEX IF NOT EXISTS exact_lookup ON exact_terms(kind,value,unit,version_id);
CREATE TABLE IF NOT EXISTS claims(id INTEGER PRIMARY KEY, version_id TEXT,block_id TEXT,extractor_version TEXT,data TEXT);
CREATE TABLE IF NOT EXISTS block_links(version_id TEXT,block_id TEXT,target_id TEXT,kind TEXT,PRIMARY KEY(version_id,block_id,target_id,kind));
CREATE TABLE IF NOT EXISTS processing_progress(version_id TEXT PRIMARY KEY,completed INTEGER,total INTEGER);
INSERT OR IGNORE INTO schema_versions VALUES(5);
CREATE TABLE IF NOT EXISTS derived_versions(version_id TEXT PRIMARY KEY,normalizer_version TEXT,claims_enabled INTEGER);
