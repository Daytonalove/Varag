# API SQLite MVP v0.2

Base: http://127.0.0.1:8766/api/v1. Все запросы требуют X-RAG-Token.
Неверная аутентификация → 401; недоступный и отсутствующий ресурс → одинаковый 404.
Cache-Control: no-store. Клиент — Streamlit backend, секреты в браузер не попадают.

Машинный контракт: GET /openapi.json (OpenAPI 3.1, инвентарь фактических маршрутов,
типизированные основные QuestionRequest/QuestionJob и multipart upload).

| Маршрут | Назначение |
|---|---|
| GET /health | Режим поиска, доступность конфигурации GigaChat, worker, model |
| PATCH /settings | `{external_llm_approved:boolean}`, только локальный пользователь настройки |
| POST /provider/check | OAuth/TLS/models без передачи документов |
| GET/POST /projects | Список / создать `{name}` |
| GET/POST /projects/{id}/documents | Список / multipart file или files, document_id для версии, external_allowed=true/false |
| GET /jobs/{id} | Статус ingestion |
| POST /documents/{id}/reprocess | Новая версия извлечения тех же оригинальных байтов, 202 |
| POST /documents/{id}/reindex | Перестроить vectors по canonical blocks, 202 |
| GET /documents/{id}/versions | Исторические редакции |
| GET /documents/{id}/original?version=… | Оригинал конкретной версии |
| GET /documents/{id}/pages/{n}?version=… | Страница и блоки, n с единицы |
| POST /documents/{id}/pages/{n}/correction | `{text,base_version}`, новая исправленная редакция, 201 |
| PATCH /documents/{id}/metadata | Метаданные и политика внешней передачи |
| DELETE /documents/{id} | Логическое исключение из текущего корпуса |
| POST /projects/{id}/questions | По умолчанию 202, сохраняемое задание |
| GET /questions/{id} | question job + answer после done; для старых ID только historical answer |
| GET /projects/{id}/question-jobs | Последние 100 заданий текущего пользователя |
| GET /projects/{id}/questions | Доступная история ответов с повторным ACL |
| POST /answers/{id}/review | verdict/comment, аудиторское заключение |
| GET /answers/{id}/export | JSON либо `?format=html` для печати |

## Вопрос

```json
{
  "question": "Какой лимит у кредитного комитета?",
  "use_llm": true,
  "conversation_id": null,
  "as_of": "2026-10-15",
  "async": true
}
```

Передавайте Idempotency-Key для повторов одного HTTP-запроса. С тем же ключом
нельзя отправить другой вопрос/дату/режим/диалог. Ответ 202 содержит id,
conversation_id и status. GET /questions/{id} возвращает queued/running/done/failed,
error для ошибки, answer для завершённого. Повторный запуск после сохранения ответа
не вызывает GigaChat снова. `async:false` оставлен для простых локальных интеграций
и тестов (201 с ответом). Пользовательский интерфейс использует очередь.

AnswerResult: answer_text, claims, evidence, discrepancies, sources,
verification_status, coverage_status, coverage_manifest, warnings, model_version,
index_version, llm_call, conversation_id, as_of. citations_checked означает
пройденные проверки источников/чисел и модельную проверку поддержки, а не юридическую
истинность. Источники содержат block_id, версию, страницу, пункт, точную цитату и
reasons. Большой блок может иметь full_block_truncated/excerpt_range;
полный текст доступен через pages. Оригинальный block ID сохраняется.

Настройки external_allowed доступны владельцу проекта. На каждый LLM-вызов
проверяются текущие ACL/policy/index_version, включая контекст предыдущих вопросов.
Если corpus изменился в ходе ответа, задача завершается ошибкой с предложением
повторить вопрос. Исторический ответ с отозванным доступом не выдаётся целиком.

## Локальный интеллектуальный поиск

`POST /api/v1/projects/{project}/search`

```json
{"query":"Какой лимит у кредитного комитета?","top_k":10,"document_ids":null}
```

`query`: 1–2000 символов, `top_k`: целое 1–50 (по умолчанию 10),
`document_ids`: null для проекта либо непустой список до 200 доступных ID.
Чужой/недоступный ID → 404; неизвестные поля/некорректный тип → 400.
Нет внешнего вызова и сохранённого вопроса/диалога. Событие search.completed
содержит IDs источников, версию корпуса и модель; текст запроса не логируется.

Ответ: sources, coverage_manifest, index_version, normative. Источник содержит
canonical block_id/document_version_id, страницу/пункт, text_exact, score (RRF),
rerank_score (logit релевантности), rerank_span, selection_reason и поисковые пути.
При отключении/ошибке модели rerank_score отсутствует. Манифест содержит
reranker.status=applied/disabled/failed/no_candidates, model_id, candidates,
windows, elapsed_ms и ограничения. Значение applied не означает полный охват.

`GET /api/v1/health`: reranker.enabled и reranker.model_id отражают загрузку;
фактическое выполнение конкретного запроса определяется его манифестом.

## Реестр, администрирование и эксплуатация

- SearchRequest принимает `as_of: YYYY-MM-DD|null` и `filters` по полям реестра
  (например `{"product":"МСБ","tags":["проверен"]}`). При as_of просматриваются
  исторические версии; дата сама по себе не устанавливает правовой приоритет.
- `GET /documents/{id}/registry`: метаданные, отдельное автоматическое предложение,
  блоки для выбора основания. `PATCH /documents/{id}/metadata` валидирует типы,
  даты, tags, verification_status и ведёт журнал до/после.
- `POST /relations`: source_document_id, target_document_id, kind,
  source_block_id, effective_from, confirmed. Обе стороны доступны на запись и
  принадлежат одному проекту. Виды: amends/repeals/replaces/supplements/references/exception.
  `GET /projects/{id}/relations`; `DELETE /relations/{id}` отзывает связь с аудитом.
- `GET /projects/{id}/members`, `PUT /projects/{id}/members/{user}` с `role`
  owner/auditor/viewer/null. Только владелец; последнего владельца удалить нельзя.
- `PUT /documents/{id}/acl`: restricted:boolean, user_ids:list[str]. Только
  владелец, список должен состоять из участников проекта.
- `GET /projects/{id}/metrics`: доступен владельцу, считает только доступные ему
  источники; usage, ошибки, latency, оценка стоимости при заданном тарифе.
- `POST /projects/{id}/retention`: days:1–36500, execute:false по умолчанию.
  execute:true очищает только ранее исключённые документы старше срока, индексы
  и связанные ответы. Append-only аудит остаётся. Это не secure erase диска.
- `GET /documents/{id}/pages/{number}/image?version=...&block=...`: PNG физической
  страницы/кадра; необязательный block включает подсветку его layout-областей.
- `GET /answers/{id}/export?format=json|html|pdf`. Другие форматы: 400.
  Заключение аудитора дополнительно поддерживает verdict=needs_documents.

Все эти пути входят в `/api/v1`, требуют локального токена и повторной проверки
прав. SSO в этом этапе отложен пользователем.
