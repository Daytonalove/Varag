"""Bank audit UI. Only the server process can access GigaChat credentials."""
import base64
import json
import os
import uuid
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components

from document_rag.config import ROOT, local_token

st.set_page_config(page_title='Банковский аудит · Документы', page_icon='📑', layout='wide')
API = os.environ.get('BANK_API_URL', 'http://127.0.0.1:8766')
database = Path(os.environ.get('BANK_DATABASE', str(ROOT / 'data/bank/audit.sqlite3')))


def call(method, path, **kwargs):
    headers = {'X-RAG-Token': local_token(database.parent)}
    headers.update(kwargs.pop('headers', {}))
    response = requests.request(method, API + '/api/v1' + path,
                                headers=headers, timeout=120, **kwargs)
    if not response.ok:
        try:
            error = response.json().get('error', 'Ошибка API')
        except ValueError:
            error = 'API недоступен'
        raise ValueError(error)
    return response


def show_source(source, key):
    label = f"{source['document_name']} · стр. {source['page_number']} · п. {source['section_path'] or '—'}"
    with st.expander(label):
        st.text(source['text_exact'])
        if source.get('extraction_method') == 'manual_correction':
            st.caption('Текст этой страницы исправлен пользователем; исходный файл сохранён отдельно.')
        if source.get('full_block_truncated'):
            st.caption('Показана найденная часть длинного блока. Полный текст доступен в оригинале страницы.')
        if source['quality_flags']:
            st.warning('Замечания: ' + ', '.join(source['quality_flags']))
        st.caption('Найдено: ' + ', '.join(source['reasons']))
        if 'rerank_score' in source:
            st.caption(f"Релевантность: {source['rerank_score']:.3f} · Окон проверено: {source.get('rerank_windows', 1)}. Оценка не является вероятностью достоверности.")
        if source.get('selection_reason') in ('exceptions', 'amendments', 'document_alternative', 'related_context'):
            st.caption('Включён для проверки исключений, изменений или связанного контекста.')
        if st.button('Открыть оригинал', key=key):
            st.session_state['source_view'] = source
            st.rerun()


def show_answer(answer):
    st.markdown(answer['answer_text'])
    st.caption(f"Охват: {answer['coverage_status']} · Версия корпуса: {answer['index_version']} · Режим: {answer['mode']}")
    for discrepancy in answer['discrepancies']:
        text = discrepancy['status'] + ': ' + discrepancy['explanation']
        if discrepancy['status'] == 'potential_unresolved_conflict':
            st.error(text)
        else:
            st.warning(text)
    with st.expander('Ограничения и подробности проверки'):
        for warning in answer['warnings']:
            st.warning(warning)
    selected_quotes = {e['block_id']: e['quote'] for e in answer['evidence']}
    if answer['mode'] == 'gigachat_extractive':
        st.write('Цитаты, выбранные GigaChat:')
        for source in answer['sources']:
            if source['block_id'] in selected_quotes:
                st.text(selected_quotes[source['block_id']])
    st.write('Источники и контекст')
    for index, source in enumerate(answer['sources'], 1):
        st.caption(f'[{index}]')
        show_source(source, answer['id'] + source['block_id'])
    with st.expander('Манифест охвата'):
        st.json(answer['coverage_manifest'])
    with st.expander('Заключение аудитора'):
        verdict = st.selectbox('Результат проверки', ['needs_review', 'confirmed', 'rejected','needs_documents'], key='verdict_' + answer['id'])
        comment = st.text_input('Комментарий', key='comment_' + answer['id'])
        if st.button('Сохранить заключение', key='review_' + answer['id']):
            call('POST', '/answers/' + answer['id'] + '/review', json={'verdict': verdict, 'comment': comment})
            st.success('Заключение записано в журнал.')
    if st.button('Подготовить JSON-экспорт', key='export_' + answer['id']):
        content = call('GET', '/answers/' + answer['id'] + '/export').content
        st.session_state['export_data_' + answer['id']] = content
    if content := st.session_state.get('export_data_' + answer['id']):
        st.download_button('Скачать результат', content, 'audit-answer.json', 'application/json', key='download_' + answer['id'])
    if st.button('Подготовить отчёт для печати', key='print_' + answer['id']):
        st.session_state['print_data_' + answer['id']] = call('GET', '/answers/' + answer['id'] + '/export?format=html').content
    if content := st.session_state.get('print_data_' + answer['id']):
        st.download_button('Скачать HTML-отчёт (печать в PDF)', content, 'audit-answer.html', 'text/html', key='print_download_' + answer['id'])
    if st.button('Подготовить PDF-отчёт', key='pdf_'+answer['id']):
        st.session_state['pdf_'+answer['id']]=call('GET','/answers/'+answer['id']+'/export?format=pdf').content
    if content:=st.session_state.get('pdf_'+answer['id']):
        st.download_button('Скачать PDF',content,'audit-answer.pdf','application/pdf',key='pdf_download_'+answer['id'])


st.title('Документы внутреннего аудита')
st.caption('Локальный MVP · SQLite · Источники и ограничения сохраняются вместе с каждым ответом.')


@st.fragment(run_every=2)
def chat_history(project, conversation):
    if conversation is None:
        st.caption('Задайте первый вопрос в новом диалоге.')
        return
    jobs = call('GET', f'/projects/{project}/question-jobs').json()
    for job in reversed(jobs):
        if job['conversation_id'] != conversation:
            continue
        if job['status'] in ('queued', 'running'):
            st.info(('В очереди: ' if job['status'] == 'queued' else 'Готовится ответ: ') + job['question'])
        elif job['status'] == 'failed':
            st.error(job['question'] + ': ' + job['error'])
    answers = call('GET', f'/projects/{project}/questions').json()
    for answer in answers:
        if (answer.get('conversation_id') or 'legacy') != conversation:
            continue
        with st.chat_message('user'):
            st.write(answer['question'])
        with st.chat_message('assistant'):
            show_answer(answer)


@st.fragment(run_every=2)
def document_statuses(project):
    docs = call('GET', f'/projects/{project}/documents').json()
    if docs:
        st.dataframe([{'Документ': d['name'], 'Статус': d['status'], 'Страниц': d['page_count'],
                       'Обработано': (str(d['progress']['completed'])+'/'+str(d['progress']['total'])) if d.get('progress') else '—',
                       'Смысловой индекс': d.get('reindex_status') or d.get('semantic_status') or 'не построен',
                       'Замечаний': len(d['warnings']), 'Внешний анализ': 'разрешён' if d['external_allowed'] else 'запрещён'} for d in docs], hide_index=True, width='stretch')


try:
    health = call('GET', '/health').json()
    projects = call('GET', '/projects').json()
    with st.sidebar:
        st.header('Аудиторские проекты')
        if health.get('can_configure'):
            with st.expander('Подключение GigaChat', expanded=not health['external_llm_approved']):
                st.caption('Модель: ' + (health.get('model') or 'не настроена'))
                permission = st.checkbox('Разрешить отправку вопросов и разрешённых фрагментов в GigaChat', value=health['external_llm_approved'], key='global_llm_permission')
                if st.button('Применить настройку GigaChat'):
                    call('PATCH', '/settings', json={'external_llm_approved': permission})
                    st.rerun()
                if st.button('Проверить подключение'):
                    checked = call('POST', '/provider/check').json()
                    if checked['available']:
                        st.success('Авторизация, TLS и доступ к модели работают.')
                    else:
                        st.error('Выбранная модель недоступна аккаунту.')
        with st.form('create_project'):
            name = st.text_input('Название нового проекта')
            if st.form_submit_button('Создать проект'):
                created = call('POST', '/projects', json={'name': name}).json()
                st.session_state['selected_project'] = created['id']
                st.rerun()
        if not projects:
            st.caption('Создайте первый проект.')
            st.stop()
        project_map = {p['id']: p['name'] for p in projects}
        project = st.selectbox('Проект', list(project_map), format_func=project_map.get, key='selected_project')
        st.caption('Внешний анализ: ' + ('разрешён настройкой сервера' if health['external_llm_approved'] else 'выключен'))
        st.caption('Поиск: ' + health.get('retrieval_mode', 'lexical'))
        ranking = health.get('reranker', {})
        if ranking.get('enabled'):
            st.caption('Реранкер: локальный cross-encoder включён')
        else:
            st.warning('Реранкер отключён. Результаты ранжируются только поиском.')
    if st.session_state.get('last_project') != project:
        st.session_state.pop('source_view', None)
        st.session_state.pop('conversation_pick', None)
        st.session_state.pop('search_results', None)
        st.session_state['last_project'] = project
    documents_tab, search_tab, chat_tab, admin_tab = st.tabs(['Документы и обработка', 'Интеллектуальный поиск', 'Вопросы и источники','Реестр и управление'])
    with documents_tab:
        st.subheader(project_map[project])
        files = st.file_uploader('Добавить PDF или сканы', type=['pdf', 'png', 'jpg', 'jpeg', 'tif', 'tiff', 'txt', 'md'], accept_multiple_files=True)
        allow_upload = st.checkbox('Разрешаю анализ этих файлов через GigaChat', value=False)
        if st.button('Загрузить документы', disabled=not files):
            # Individual HTTP uploads keep large batches below the request memory limit.
            progress = st.progress(0)
            for index, file in enumerate(files):
                results = call('POST', f'/projects/{project}/documents', data={'external_allowed': str(allow_upload).lower()}, files={'file': (file.name, file.getvalue())}).json()
                for result in results:
                    if result.get('error'):
                        st.error(file.name + ': ' + result['error'])
                progress.progress((index + 1) / len(files))
            st.success('Файлы приняты. Обработка идёт в фоне.')
        st.button('Обновить статусы')
        docs = call('GET', f'/projects/{project}/documents').json()
        document_statuses(project)
        for doc in docs:
            with st.expander('Управление: ' + doc['name']):
                if doc.get('processing_error'):
                    st.error(doc['processing_error'])
                if doc['warnings']:
                    st.json(doc['warnings'])
                if doc['page_count']:
                    number = st.number_input('Страница', min_value=1, max_value=doc['page_count'], step=1, key='page_number_' + doc['id'])
                    if st.button('Открыть страницу и текст', key='view_page_' + doc['id']):
                        st.session_state['source_view'] = dict(document_name=doc['name'], document_id=doc['id'], document_version_id=doc['active_version_id'], page_index=number - 1, page_number=str(number))
                external = st.checkbox('Разрешить передачу фрагментов этого документа в GigaChat', value=bool(doc['external_allowed']), key='external_' + doc['id'])
                if st.button('Сохранить политику передачи', key='policy_' + doc['id']):
                    call('PATCH', '/documents/' + doc['id'] + '/metadata', json={'external_allowed': external})
                    st.rerun()
                if st.button('Повторить распознавание', key='reprocess_' + doc['id']):
                    call('POST', '/documents/' + doc['id'] + '/reprocess')
                    st.rerun()
                if st.button('Перестроить смысловой индекс', key='reindex_' + doc['id']):
                    call('POST', '/documents/' + doc['id'] + '/reindex')
                    st.success('Переиндексация поставлена в очередь.')
                replacement = st.file_uploader('Новая версия документа', type=['pdf', 'png', 'jpg', 'jpeg', 'tif', 'tiff', 'txt', 'md'], key='version_' + doc['id'])
                if st.button('Добавить версию', disabled=replacement is None, key='replace_' + doc['id']):
                    response = call('POST', f'/projects/{project}/documents', data={'document_id': doc['id']}, files={'file': (replacement.name, replacement.getvalue())}).json()
                    if response[0].get('error'):
                        st.error(response[0]['error'])
                    else:
                        st.rerun()
                if st.button('Исключить из текущего корпуса', key='delete_' + doc['id']):
                    call('DELETE', '/documents/' + doc['id'])
                    st.session_state.pop('source_view', None)
                    st.rerun()
    with search_tab:
        st.subheader('Поиск по документам')
        st.caption('Поиск и реранкинг выполняются локально. Разрешение GigaChat для этого не требуется.')
        document_map = {d['id']: d['name'] for d in docs}
        with st.form('document_search'):
            query = st.text_input('Поисковый запрос', placeholder='Например: как согласовать исключение из лимита?', max_chars=2000)
            scope = st.multiselect('Документы (пусто — весь проект)', list(document_map), format_func=document_map.get)
            top_k = st.slider('Количество результатов', 1, 50, 10)
            search_date=st.text_input('Историческая дата поиска (YYYY-MM-DD)',max_chars=10)
            filter_product=st.text_input('Фильтр по продукту',max_chars=100)
            filter_status=st.text_input('Фильтр по статусу документа',max_chars=100)
            filter_tags=st.text_input('Метки через запятую',max_chars=1000)
            submitted = st.form_submit_button('Найти фрагменты')
        if submitted:
            with st.spinner('Ищу и оцениваю релевантность фрагментов…'):
                st.session_state['search_results'] = call('POST', f'/projects/{project}/search',
                    json={'query': query, 'document_ids': scope or None, 'top_k': top_k,'as_of':search_date or None,
                          'filters':dict(**({'product':filter_product} if filter_product else {}),
                                         **({'status':filter_status} if filter_status else {}),
                                         **({'tags':[t.strip() for t in filter_tags.split(',') if t.strip()]} if filter_tags else {}))}).json()
        cached = st.session_state.get('search_results')
        cached_ids = set(cached['coverage_manifest']['accessible_document_ids']) if cached else set()
        changed_scope = cached and (not cached_ids.issubset(document_map) or
                                    (cached['coverage_manifest'].get('scope') == 'project' and not cached['coverage_manifest'].get('parameters',{}).get('filters') and cached_ids != set(document_map)))
        if cached and (changed_scope or any(s['document_id'] not in document_map or
                          (not cached['coverage_manifest'].get('as_of') and s['document_version_id'] != next(d['active_version_id'] for d in docs if d['id'] == s['document_id']))
                          for s in cached['sources'])):
            st.session_state.pop('search_results', None)
            st.info('Состав или версии документов изменились. Выполните поиск заново.')
        if result := st.session_state.get('search_results'):
            manifest = result['coverage_manifest']
            rank = manifest.get('reranker', {})
            st.caption(f"Найдено: {len(result['sources'])} · Проверено индексов документов: {len(manifest['searched_document_ids'])} · Время: {manifest.get('elapsed_ms', 0) / 1000:.2f} с · Версия корпуса: {result['index_version']}")
            if rank.get('status') == 'applied':
                st.caption(f"Реранкер обработал {rank['candidates']} кандидатов.")
            elif rank.get('status') != 'no_candidates':
                st.warning('Реранкинг не выполнен; показаны результаты базового поиска.')
            if manifest['skipped_documents']:
                st.warning('Часть документов ещё не проиндексирована или содержит ошибки. Поиск неполный.')
            if not result['sources']:
                st.info('Фрагменты не найдены. Уточните запрос или дождитесь обработки документов.')
            for index, source in enumerate(result['sources'], 1):
                st.caption(f'Результат {index}')
                show_source(source, 'search_' + source['block_id'])
            with st.expander('Как выполнен поиск'):
                st.json(manifest)
    with chat_tab:
        jobs = call('GET', f'/projects/{project}/question-jobs').json()
        conversations = {'new': 'Новый диалог', 'legacy': 'Ранее сохранённые ответы (до диалогов)'}
        for job in reversed(jobs):
            conversations.setdefault(job['conversation_id'], job['question'][:80])
        if target := st.session_state.pop('switch_conversation', None):
            st.session_state['conversation_pick'] = target
        if st.session_state.get('conversation_pick') not in conversations:
            st.session_state['conversation_pick'] = 'new'
        selected = st.selectbox('Диалог', list(conversations), format_func=conversations.get, key='conversation_pick')
        conversation = None if selected == 'new' else selected
        use_llm = st.checkbox('Ответить через GigaChat', value=health['external_llm_approved'], disabled=not health['external_llm_approved'])
        as_of = st.text_input('Дата проверки (необязательно, YYYY-MM-DD)', placeholder='2026-10-15')
        st.caption('Уточняющие вопросы используют контекст выбранного диалога. Источники каждый раз ищутся заново.')
        question = st.chat_input('Например: какой лимит у кредитного комитета?')
        if question:
            job = call('POST', f'/projects/{project}/questions', headers={'Idempotency-Key': uuid.uuid4().hex},
                       json={'question': question, 'use_llm': use_llm, 'conversation_id': None if conversation == 'legacy' else conversation, 'as_of': as_of.strip() or None}).json()
            st.session_state['switch_conversation'] = job['conversation_id']
            st.rerun()
        chat_history(project, conversation)
    with admin_tab:
        from document_rag.bank.admin_ui import render
        render(call,project,docs)
    if source := st.session_state.get('source_view'):
        st.divider()
        st.subheader(f"Оригинал: {source['document_name']} · стр. {source['page_number']}")
        path = '/documents/' + source['document_id']
        version = '?version=' + source['document_version_id']
        page = call('GET', path + '/pages/' + str(source['page_index'] + 1) + version).json()
        original = call('GET', path + '/original' + version).content
        left, right = st.columns(2)
        with left:
            st.text(page['text_exact'])
            st.caption('Подсветка показывает область извлечённого текста; точность распознавания сверяйте с оригиналом.' if source.get('bbox') else 'Для этого фрагмента координаты не сохранены.')
            with st.expander('Исправить распознанный текст'):
                corrected = st.text_area('Текст страницы', value=page['text_exact'], max_chars=100000, height=280,
                                         key='correction_' + source['document_version_id'] + str(source['page_index']))
                if st.button('Сохранить отдельной редакцией'):
                    updated = call('POST', path + '/pages/' + str(source['page_index'] + 1) + '/correction',
                                   json={'text': corrected, 'base_version': source['document_version_id']}).json()
                    st.session_state['source_view'] = dict(source, document_version_id=updated['version_id'])
                    st.rerun()
        with right:
            st.download_button('Скачать оригинал', original, source['document_name'])
            if source['document_name'].lower().endswith(('.pdf','.png','.jpg','.jpeg','.tif','.tiff')):
                params={'version':source['document_version_id']}
                if source.get('block_id'):
                    params['block']=source['block_id']
                st.image(call('GET',path+'/pages/'+str(source['page_index']+1)+'/image',params=params).content)
            if source['document_name'].lower().endswith('.pdf'):
                data = base64.b64encode(original).decode('ascii')
                components.html(f'<iframe src="data:application/pdf;base64,{data}#page={source["page_index"] + 1}" width="100%" height="650"></iframe>', height=660)
            elif source['document_name'].lower().endswith(('.png', '.jpg', '.jpeg')):
                st.image(original)
except (requests.RequestException, ValueError) as exc:
    st.error(str(exc) if isinstance(exc, ValueError) else 'Backend недоступен. Запустите python -m document_rag.bank.api.')
