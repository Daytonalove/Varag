"""Local retrieval with separate passes, provenance and explicit budget accounting."""
from collections import defaultdict
from decimal import Decimal
import json
import math
import re
import time

STOP = set('какой какая какие как что это для или по из все всех у в на и с к о а ли'.split())
RULE = re.compile(r'лимит|срок|полномоч|правил|обязан|запре|согласован|процедур|разреш|исключен|комитет', re.I)
SPECIAL = r'исключен|кроме|за исключением|приложен|специальн|сноск'
AMEND = r'измен|отмен|замен|утратив|вступает|редакци'
NORMATIVE_QUERY = re.compile(RULE.pattern + r'|вправе|можно ли|может.*(?:одобр|реш|выда)|кто.*(?:принима|утверж|реш|отвеч)|максим|миним|порог|ограничен|превыш|предел|сколько.*(?:дн|день|месяц|год)|хранени|порядок', re.I)
from .norms import amounts, compare, facts, DATE, iso_date


def terms(question):
    # Prefix lexical retrieval approximates inflection; not a Russian semantic model.
    return list(dict.fromkeys(w[:7] if len(w) > 7 else w for w in re.findall(r'\w+', question.lower().replace('ё', 'е')) if w not in STOP and len(w) > 2))[:32]


class Retriever:
    def __init__(self, store, context_characters=16000, candidate_limit=80, *, reranker=None, rerank_limit=160):
        self.store, self.budget, self.limit = store, context_characters, candidate_limit
        self.reranker, self.rerank_limit = reranker, rerank_limit

    def search(self, project, user, question, *, document_ids=None, top_k=None, as_of=None, filters=None):
        started = time.monotonic()
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
            raise ValueError('Вопрос: 1–2000 символов.')
        if top_k is not None and (type(top_k) is not int or not 1 <= top_k <= 50):
            raise ValueError('top_k: целое число от 1 до 50.')
        if document_ids is not None and (not isinstance(document_ids, list) or not document_ids or
                len(document_ids) > 200 or any(not isinstance(d, str) for d in document_ids)):
            raise ValueError('document_ids: непустой список идентификаторов, не более 200.')
        if as_of:
            from datetime import date
            try:
                date.fromisoformat(as_of)
            except (TypeError, ValueError):
                raise ValueError('Дата проверки должна иметь формат YYYY-MM-DD.') from None
        result = self._search(project, user, question.strip(), document_ids, top_k, as_of, filters)
        # Inference may be slow. Recheck ACL and corpus after closing the snapshot.
        with self.store.connect() as conn:
            self.store.project_access(conn, project, user)
            for document in result['coverage_manifest']['accessible_document_ids']:
                self.store.document_access(conn, document, user)
            current = conn.execute('SELECT index_version FROM projects WHERE id=?', (project,)).fetchone()[0]
            if current != result['index_version']:
                raise ValueError('Корпус изменился во время поиска. Повторите запрос.')
        result['coverage_manifest']['elapsed_ms'] = round((time.monotonic() - started) * 1000)
        return result

    def _search(self, project, user, question, document_ids, top_k, as_of, filters):
        from .fast_generation import is_summary
        summary = is_summary(question)
        with self.store.connect() as conn:
            conn.execute('BEGIN')  # Consistent corpus + ACL snapshot for the whole search.
            docs = self.store.documents(project, user, conn)
            if filters is not None:
                from .registry import filter_documents
                docs = filter_documents(docs, filters)
            if document_ids is not None:
                from .store import NotFound
                if not set(document_ids).issubset({d['id'] for d in docs}):
                    raise NotFound()
                docs = [d for d in docs if d['id'] in document_ids]
            if as_of:
                # Search all immutable versions, not only the latest upload.
                # Later filtering/comparison uses explicit dates, never file recency.
                historical = []
                for doc in docs:
                    for version in conn.execute('SELECT * FROM versions WHERE document_id=? ORDER BY rowid', (doc['id'],)):
                        historical.append(dict(doc, active_version_id=version['id'], status=version['status'],
                                               warnings=json.loads(version['warnings'])))
                docs = historical
            index_version = conn.execute('SELECT index_version FROM projects WHERE id=?', (project,)).fetchone()[0]
            ready = [d for d in docs if d['status'] in ('ready', 'ready_with_warnings')]
            skipped = [{'document_id': d['id'], 'reason': d['status']} for d in docs if d not in ready]
            manifest = dict(accessible_document_ids=list(dict.fromkeys(d['id'] for d in docs)), indexed_document_ids=list(dict.fromkeys(d['id'] for d in ready)),
                            searched_document_ids=[], skipped_documents=skipped, retrieval_passes=['lexical', 'numeric', 'parent_neighbor'],
                            retrieved_candidate_count=0, examined_candidate_count=0, unexamined_candidate_count=0,
                            scope='selected_documents' if document_ids else 'project',
                            coverage_status='insufficient', limitations=['cross_reference_resolution_partial'])
            if not self.store.claims_enabled:
                manifest['limitations'].append('claims_index_disabled')
            visible_ids = {d['id'] for d in ready}
            document_relations = [dict(r) for r in conn.execute("SELECT * FROM document_relations WHERE status='active'")
                                  if r['source_document_id'] in visible_ids and r['target_document_id'] in visible_ids]
            manifest['document_relation_ids'] = [r['id'] for r in document_relations]
            manifest['as_of'] = as_of
            manifest['searched_version_ids'] = []
            manifest['parameters'] = dict(context_characters=self.budget, candidate_limit=top_k or self.limit, rerank_limit=self.rerank_limit,
                                          embedding_model=self.store.embedding_model, filters=filters or {}, rrf_k=60)
            query_vector = None
            if self.store.encoder:
                try:
                    with self.store.embedding_lock:
                        query_vector = self.store.encoder.encode([question], query=True)[0]
                    manifest['retrieval_passes'].append('semantic')
                except Exception:
                    manifest['limitations'].append('semantic_query_failed')
            else:
                manifest['limitations'].append('semantic_disabled')
            normative = bool(NORMATIVE_QUERY.search(question))
            if normative:
                manifest['retrieval_passes'] += ['alternatives', 'exceptions', 'amendments']
            query_terms = terms(question)
            query_amounts = {(a['value'], a['currency']) for a in amounts(question)}
            exact = {(f['kind'], f['value'], f['unit']) for f in facts(question)}
            for value in DATE.findall(question):
                try:
                    exact.add(('date', iso_date(value), ''))
                except ValueError:
                    pass
            exact.update(('clause', v, '') for v in re.findall(r'(?:пункт\w*|п\.)\s*(\d+(?:\.\d+)*)', question, re.I))
            manifest['retrieval_passes'] += ['exact_index', 'document_relations']
            if self.store.claims_enabled:
                manifest['retrieval_passes'].append('claims_index')
            from .structural import codes, query_pages, standards
            requested_codes, requested_pages = codes(question), query_pages(question)
            requested_standards = standards(question) if re.search('гост|стандарт',question,re.I) else []
            manifest['structural_query'] = dict(codes=requested_codes,pages=requested_pages,standards=requested_standards)
            manifest['retrieval_passes'].append('structural_keys')
            all_hits = []
            for doc in ready:
                if doc['id'] not in manifest['searched_document_ids']:
                    manifest['searched_document_ids'].append(doc['id'])
                manifest['searched_version_ids'].append(doc['active_version_id'])
                rows = [dict(r) for r in conn.execute('SELECT * FROM blocks WHERE version_id=? ORDER BY ordinal', (doc['active_version_id'],))]
                identity = doc['name']+' '+json.dumps(doc['metadata'],ensure_ascii=False)+' '+''.join(b['text_exact'] for b in rows if b['page_index']==0)[:1500]
                available_standards=standards(identity)
                if requested_standards and not any(a==s or a.startswith(s+'-') for s in requested_standards for a in available_standards):
                    continue
                if requested_pages:
                    rows = [b for b in rows if b['page_index']+1 in requested_pages]
                by_id = {b['id']: b for b in rows}
                page_layouts={r['page_index']:json.loads(r['data']) for r in conn.execute('SELECT * FROM page_layouts WHERE version_id=?',(doc['active_version_id'],))}
                reasons = defaultdict(set)
                code_spans = defaultdict(list)
                if summary:
                    for block in rows:
                        reasons[block['id']].add('summary_coverage')
                for code in requested_codes:
                    for hit in conn.execute('SELECT * FROM structural_keys WHERE version_id=? AND code=? ORDER BY definition DESC,start', (doc['active_version_id'],code)):
                        if hit['block_id'] in by_id:
                            code_spans[hit['block_id']].append(dict(hit))
                            reasons[hit['block_id']].add('exact_code')
                if requested_pages:
                    for b in rows:
                        reasons[b['id']].add('exact_page')
                for kind, value, unit in exact:
                    for row in conn.execute('SELECT block_id FROM exact_terms WHERE version_id=? AND kind=? AND value=? AND unit=?',
                                            (doc['active_version_id'], kind, value, unit)):
                        if row[0] in by_id:
                            reasons[row[0]].add('exact_index')
                for relation in document_relations:
                    if relation['source_document_id'] == doc['id'] and relation['source_block_id'] in by_id:
                        reasons[relation['source_block_id']].add('document_relation')
                    if relation['target_document_id'] == doc['id']:
                        origin=conn.execute('SELECT text_exact FROM blocks WHERE id=?',(relation['source_block_id'],)).fetchone()
                        refs=re.findall(r'(?:пункт\w*|п\.)\s*(\d+(?:\.\d+)*)',origin[0],re.I) if origin else []
                        for block in rows:
                            if block['section_path'] in refs:
                                reasons[block['id']].add('document_relation')
                if self.store.claims_enabled and normative:
                    for row in conn.execute('SELECT block_id FROM claims WHERE version_id=?', (doc['active_version_id'],)):
                        if row[0] in by_id and any(t in by_id[row[0]]['text_normalized'] for t in query_terms):
                            reasons[row[0]].add('claims_index')
                semantic, lexical_scores = {}, {}
                if query_vector is not None:
                    state = conn.execute('SELECT status FROM embedding_versions WHERE version_id=? AND model_id=?',
                                         (doc['active_version_id'], self.store.embedding_model)).fetchone()
                    if not state or state[0] != 'ready':
                        manifest['limitations'].append('semantic_index_unavailable:' + doc['id'])
                    else:
                        semantic = {}
                        vectors = conn.execute('SELECT v.block_id,v.value FROM block_vectors v JOIN blocks b ON b.id=v.block_id WHERE b.version_id=? AND v.model_id=?',
                                               (doc['active_version_id'], self.store.embedding_model))
                        for vector in vectors:
                            value = json.loads(vector['value'])
                            if len(value) != len(query_vector):
                                raise ValueError('Размерность embedding не совпадает; требуется переиндексация.')
                            cosine = math.fsum(a * b for a, b in zip(query_vector, value))
                            semantic[vector['block_id']] = max(cosine, semantic.get(vector['block_id'], -1))
                        ranked = sorted(semantic, key=semantic.get, reverse=True)
                        if len(ranked) > 8:
                            manifest['limitations'].append('semantic_per_document_limit:' + doc['id'])
                        for block_id in ranked[:8]:
                            if block_id in by_id:
                                reasons[block_id].add('semantic')
                if query_terms:
                    expression = ' OR '.join('"' + t + '"*' for t in query_terms)
                    lexical = conn.execute('SELECT block_id,bm25(block_fts) rank FROM block_fts WHERE block_fts MATCH ? AND version_id=? ORDER BY rank',
                                           (expression, doc['active_version_id']))
                    for row in lexical:
                        if row['block_id'] in by_id:
                            reasons[row['block_id']].add('lexical')
                            lexical_scores[row['block_id']] = row['rank']
                for block in rows:
                    text = block['text_normalized']
                    values = {(a['value'], a['currency']) for a in amounts(text)}
                    if values & query_amounts:
                        reasons[block['id']].add('numeric')
                    if normative:
                        for pattern, label in [(RULE.pattern, 'alternatives'), (SPECIAL, 'exceptions'), (AMEND, 'amendments')]:
                            if re.search(pattern, text, re.I):
                                reasons[block['id']].add(label)
                relations = defaultdict(set)
                for block_id in list(reasons):
                    b = by_id[block_id]
                    related = [n for n in rows if abs(n['ordinal'] - b['ordinal']) == 1]
                    related += [n for n in rows if n['id']!=b['id'] and (
                        (b['section_path'] and n['section_path']==b['section_path']) or
                        (n['page_index']==b['page_index'] and n['block_type'] in ('footnote','table')) or
                        (re.search(r'означает|понимается|определяется как',n['text_exact'],re.I) and any(t in n['text_normalized'] for t in query_terms)))]
                    references = re.findall(r'(?:пункт\w*|п\.)\s*(\d+(?:\.\d+)+)', b['text_exact'], re.I)
                    for target in rows:
                        if target['section_path'] in references:
                            related.append(target)
                            reasons[target['id']].add('cross_reference')
                    seen = set()
                    while b.get('parent_block_id') in by_id and b['parent_block_id'] not in seen:
                        b = by_id[b['parent_block_id']]
                        seen.add(b['id'])
                        related.append(b)
                    for neighbor in related:
                        reasons[neighbor['id']].add('parent_neighbor')
                        relations[block_id].add(neighbor['id'])
                hits = []
                for block_id, paths in reasons.items():
                    b = by_id[block_id]
                    if not b['text_exact'].strip():
                        continue
                    hits.append(dict(b, block_id=block_id, document_id=doc['id'], document_name=doc['name'],
                                     document_version_id=doc['active_version_id'], reasons=sorted(paths), score=0,
                                     structural_matches=code_spans[block_id],
                                     table_rows=[r for r in page_layouts.get(b['page_index'],{}).get('tables',[]) if r['code'] in codes(b['text_exact'])],
                                     document_metadata=doc['metadata'],
                                     document_relations=[r for r in document_relations if r['source_document_id']==doc['id'] or r['target_document_id']==doc['id']],
                                     related_block_ids=sorted(relations[block_id]),
                                     semantic_score=semantic.get(block_id), lexical_bm25=lexical_scores.get(block_id),
                                     quality_flags=json.loads(b['quality_flags']), external_allowed=bool(doc['external_allowed'])))
                all_hits.append(hits)
            # Reciprocal rank fusion across the entire accessible candidate corpus.
            # Per-document rank 1 is not globally comparable to another rank 1.
            ordered = [block for group in all_hits for block in group]
            for relation in document_relations:
                origin=next((b for b in ordered if b['block_id']==relation['source_block_id']),None)
                if origin:
                    targets=[b for b in ordered if b['document_id']==relation['target_document_id'] and 'document_relation' in b['reasons']]
                    origin['related_block_ids'].extend(b['block_id'] for b in targets)
                    for target in targets:
                        target['related_block_ids'].append(origin['block_id'])
            fused = defaultdict(float)
            for key, reverse in [('semantic_score', True), ('lexical_bm25', False)]:
                ranked = sorted([b for b in ordered if b[key] is not None], key=lambda b: b[key], reverse=reverse)
                for rank, block in enumerate(ranked, 1):
                    fused[block['block_id']] += 1 / (60 + rank)
            for block in ordered:
                block['score'] = fused[block['block_id']] + (1 / 60 if set(block['reasons']) & {'numeric','exact_index'} else 0)
            ordered.sort(key=lambda b: (-b['score'], b['document_name'], b['ordinal']))
            manifest['retrieval_passes'].append('rrf_fusion')
            total_candidates = len(ordered)
            manifest['reranker'] = dict(status='disabled', model_id=None, candidates=0)
            if self.reranker and ordered and not summary:
                # Each independent route receives candidates before the shared cap.
                lanes = [[b for b in ordered if reason in b['reasons']] for reason in
                         ('exact_code', 'exact_page', 'semantic', 'lexical', 'numeric', 'exact_index', 'document_relation', 'claims_index', 'alternatives', 'exceptions', 'amendments', 'cross_reference')]
                # Each requested key gets a protected lane, even if reranking
                # would otherwise prefer a semantically similar but wrong row.
                lanes = [[b for b in ordered if any(m['code']==code for m in b['structural_matches'])] for code in requested_codes] + lanes
                doc_best = {}
                for block in ordered:
                    doc_best.setdefault(block['document_id'], block)
                lanes.append(list(doc_best.values()))
                lanes.append(ordered)
                pool, seen = [], set()
                for i in range(max(map(len, lanes), default=0)):
                    for lane in lanes:
                        if i < len(lane) and lane[i]['block_id'] not in seen:
                            seen.add(lane[i]['block_id'])
                            pool.append(lane[i])
                            if len(pool) == self.rerank_limit:
                                break
                    if len(pool) == self.rerank_limit:
                        break
                if len(pool) < total_candidates:
                    manifest['limitations'].append('reranker_candidate_budget')
                rank_started = time.monotonic()
                try:
                    scores = self.reranker.score(question, [b['text_exact'] for b in pool])
                    if len(scores) != len(pool) or any(not math.isfinite(s.score) or not 0 <= s.start <= s.end <= len(b['text_exact']) for b, s in zip(pool, scores)):
                        raise ValueError('Невалидные оценки реранкера.')
                    ranked = [dict(b, retrieval_score=b['score'], rerank_score=s.score,
                                   rerank_span=[s.start, s.end], rerank_windows=s.windows,
                                   rerank_total_windows=s.total_windows) for b, s in zip(pool, scores)]
                    ordered = sorted(ranked, key=lambda b: (-b['rerank_score'], -b['score'], b['block_id']))
                    manifest['retrieval_passes'].append('cross_encoder_rerank')
                    manifest['reranker'] = dict(status='applied', model_id=self.reranker.model_id, candidates=len(pool),
                        windows=sum(s.windows for s in scores), elapsed_ms=round((time.monotonic() - rank_started) * 1000),
                        score_type='relevance_logit')
                    if any(s.windows < s.total_windows for s in scores):
                        manifest['limitations'].append('reranker_window_budget')
                    if any(s.query_truncated for s in scores):
                        manifest['limitations'].append('reranker_query_truncated')
                except Exception as exc:
                    # Keep usable hybrid results, but never claim reranking succeeded.
                    manifest['reranker'] = dict(status='failed', model_id=self.reranker.model_id,
                                               candidates=len(pool), error_type=type(exc).__name__)
                    manifest['limitations'].append('reranker_failed')
            elif not self.reranker:
                manifest['limitations'].append('reranker_disabled')
            else:
                manifest['reranker'] = dict(status='no_candidates', model_id=self.reranker.model_id, candidates=0)

            # Keep the best result first; reserve independent evidence for audit questions.
            priority, added = [], set()
            def reserve(block, reason):
                if block['block_id'] not in added:
                    priority.append(dict(block, selection_reason=reason))
                    added.add(block['block_id'])
            if summary:
                # Sample across the whole active version, not only passages matching 'summary'.
                from itertools import zip_longest
                versions = list(dict.fromkeys(b['document_version_id'] for b in ordered))
                count = max(1, min(max(24, len(versions)), top_k or self.limit) // max(1, len(versions)))
                lanes = []
                for version in versions:
                    blocks = sorted([b for b in ordered if b['document_version_id'] == version], key=lambda b: b['ordinal'])
                    indexes = sorted({round(i * (len(blocks)-1) / max(1, count-1)) for i in range(min(count, len(blocks)))}) if len(blocks) > count else range(len(blocks))
                    lanes.append([blocks[i] for i in indexes])
                for group in zip_longest(*lanes):
                    for block in group:
                        if block is not None:
                            reserve(block, 'summary_coverage')
                manifest['retrieval_passes'].append('summary_coverage')
                manifest['limitations'].append('summary_sampled_context')
            for code in requested_codes:
                exact_matches=[b for b in ordered if any(m['code']==code for m in b['structural_matches'])]
                exact_matches.sort(key=lambda b:not any(m['code']==code and m['definition'] for m in b['structural_matches']))
                # Preserve document/version ambiguity instead of silently choosing one edition.
                seen_versions=set()
                for block in exact_matches:
                    if block['document_version_id'] not in seen_versions:
                        reserve(block,'exact_code')
                        seen_versions.add(block['document_version_id'])
            for block in ordered[:3]:
                reserve(block, 'relevance')
            if getattr(self,'diversify_documents',False):
                represented=set()
                for block in ordered:
                    if block['document_version_id'] not in represented:
                        reserve(block,'document_coverage')
                        represented.add(block['document_version_id'])
            if normative:
                for reason in ('exceptions', 'amendments', 'numeric', 'document_relation'):
                    for block in [b for b in ordered if reason in b['reasons']][:2]:
                        reserve(block, reason)
                represented = set()
                for block in ordered:
                    if block['document_id'] not in represented and len(represented) < 24:
                        reserve(block, 'document_alternative')
                        represented.add(block['document_id'])
            by_id = {b['block_id']: b for b in ordered}
            for anchor in list(priority):
                for related_id in anchor.get('related_block_ids', []):
                    if related_id in by_id:
                        reserve(by_id[related_id], 'related_context')
            for block in ordered:
                reserve(block, 'relevance')
            selected, used = [], 0
            for block in priority:
                if len(block['text_exact']) > 6000:
                    lowered = block['text_exact'].lower().replace('ё', 'е')
                    position = min((m['start'] for m in block['structural_matches']),default=block.get('rerank_span', [min((lowered.find(t) for t in query_terms if t in lowered), default=0)])[0])
                    start = max(0, position - 1000)
                    end = min(len(block['text_exact']), start + 6000)
                    excerpt = block['text_exact'][start:end]
                    block = dict(block, text_exact=excerpt, text_normalized=excerpt.lower().replace('ё', 'е'),
                                 excerpt_range=[start, end], full_block_truncated=True, full_block_length=len(block['text_exact']))
                cost = len(block['text_exact']) + 300
                if len(selected) < (top_k or self.limit) and used + cost <= self.budget:
                    selected.append(block)
                    used += cost
            # Display selected evidence by relevance, retaining reservation reasons.
            if manifest['reranker']['status'] == 'applied':
                selected.sort(key=lambda b: (b.get('selection_reason')!='exact_code', -b['rerank_score'], -b['score'], b['block_id']))
            manifest['missing_exact_codes']=[code for code in requested_codes if not any(code in codes(b['text_exact']) for b in selected)]
            if manifest['missing_exact_codes']:
                manifest['limitations'].append('exact_code_not_found')
            manifest.update(retrieved_candidate_count=total_candidates, examined_candidate_count=len(selected),
                            unexamined_candidate_count=total_candidates - len(selected), context_characters=used,
                            selected_document_ids=list(dict.fromkeys(b['document_id'] for b in selected)))
            if len(selected) < total_candidates:
                manifest['limitations'].append('candidate_or_context_budget')
            if any(b.get('full_block_truncated') for b in selected):
                manifest['limitations'].append('long_block_excerpt')
            selected_ids = {b['block_id'] for b in selected}
            if any(set(b.get('related_block_ids', [])) - selected_ids for b in selected):
                manifest['limitations'].append('related_context_budget')
            if any(d['warnings'] for d in ready):
                manifest['limitations'].append('page_quality_warnings')
            # Full cross-reference/context verification still requires further implementation.
            manifest['coverage_status'] = 'partial' if selected or docs else 'insufficient'
            return dict(sources=selected, coverage_manifest=manifest, index_version=index_version, normative=normative)


class ContradictionService:
    """Source-bound comparison; money, durations, operators, scope and dates."""
    def assess(self, sources, as_of=None):
        return compare(sources, as_of)
