"""Varyag: document conversations. Provider credentials stay on the server."""
import os
import time
import uuid
from pathlib import Path

import requests
import streamlit as st

from document_rag.config import ROOT, local_token

st.set_page_config(page_title='Варяг', page_icon='✦', layout='wide')
API = os.environ.get('BANK_API_URL', 'http://127.0.0.1:8766')
database = Path(os.environ.get('BANK_DATABASE', str(ROOT / 'data/bank/audit.sqlite3')))
st.markdown('''<style>
.block-container {max-width: 1080px; padding-top: 2.5rem;}
h1 {font-size: 3.5rem !important; letter-spacing: -.05em;}
[data-testid="stSidebar"] {background: #f5f6f8;}
[data-testid="stChatMessage"] {border-radius: 16px;}
[data-testid="stSidebar"] button {text-align: left;}
</style>''', unsafe_allow_html=True)


def call(method, path, **kwargs):
    headers = {'X-RAG-Token': local_token(database.parent)}
    headers.update(kwargs.pop('headers', {}))
    response = requests.request(method, API + '/api/v1' + path,
                                headers=headers, timeout=120, **kwargs)
    if not response.ok:
        try:
            error = response.json().get('error', 'Не удалось выполнить запрос.')
        except ValueError:
            error = 'Сервер недоступен.'
        raise ValueError(error)
    return response


def select_chat(chat):
    st.session_state['active_chat'] = (chat['project_id'], chat.get('conversation_id'))
    st.session_state.pop('source_view', None)


def show_answer(answer):
    if answer.get('mode') == 'gigachat_failed':
        st.error((answer.get('failure') or {}).get('message') or answer['answer_text'])
    else:
        st.markdown(answer['answer_text'])
        if answer.get('failure'):
            st.warning(answer['failure']['message'])
    for limitation in answer.get('answer_limitations',[]) if answer.get('claims') else []:
        st.warning(limitation)
    if 'llm_analysis_incomplete' in answer.get('coverage_manifest',{}).get('limitations',[]):
        st.warning('Часть найденных фрагментов не удалось проверить. Ответ может быть неполным.')
    display_sources = answer.get('sources') or answer.get('retrieved_sources', [])
    if display_sources:
        label = 'Найденные фрагменты — ответ модели не получен' if answer.get('mode') == 'gigachat_failed' else 'Источники'
        with st.expander(label):
            for index, source in enumerate(display_sources, 1):
                st.markdown(f"**[{index}] {source['document_name']} · стр. {source['page_number']}**")
                st.text(source['text_exact'])
                if st.button('Открыть документ', key='source_'+answer['id']+source['block_id']):
                    st.session_state['source_view'] = source
                    st.rerun()


def send_question(project, conversation, question, model_id):
    job = call('POST', f'/projects/{project}/questions',
               headers={'Idempotency-Key': uuid.uuid4().hex},
               json={'question': question, 'use_llm': True,
                     'conversation_id': None if conversation == 'legacy' else conversation,
                     'model_id': model_id}).json()
    st.session_state['active_chat'] = (project, job['conversation_id'])
    return job


@st.fragment(run_every=2)
def messages(project, conversation):
    try:
        if conversation == 'legacy':
            for answer in call('GET', f'/projects/{project}/questions').json():
                if answer.get('conversation_id'):
                    continue
                with st.chat_message('user'):
                    st.write(answer['question'])
                with st.chat_message('assistant'):
                    show_answer(answer)
            return
        if conversation is None:
            return
        jobs = call('GET', f'/projects/{project}/question-jobs').json()
        for job in reversed(jobs):
            if job['conversation_id'] != conversation:
                continue
            with st.chat_message('user'):
                st.write(job['question'])
            with st.chat_message('assistant'):
                if job['status'] in ('queued', 'running'):
                    st.info('Готовлю ответ…')
                elif job['status'] == 'failed':
                    st.error(job['error'])
                    if st.button('Повторить', key='retry_'+job['id']):
                        send_question(project, conversation, job['question'], job.get('model_id'))
                        st.rerun()
                elif job.get('answer'):
                    show_answer(job['answer'])
    except (requests.RequestException, ValueError) as exc:
        st.error(str(exc) if isinstance(exc, ValueError) else 'Не удалось обновить сообщения.')


STATUS = {'queued': 'В очереди', 'processing': 'Обрабатывается', 'running': 'Обрабатывается',
          'validating': 'Проверяется файл', 'parsing': 'Извлекается текст', 'ocr': 'Распознаётся текст',
          'structuring': 'Обрабатывается текст', 'indexing': 'Подготавливается поиск',
          'ready': 'Готов', 'ready_with_warnings': 'Готов', 'failed': 'Ошибка обработки'}


@st.fragment(run_every=2)
def processing_status(project, initial):
    try:
        docs = call('GET', f'/projects/{project}/documents').json()
        signature = [(d['id'], d['status']) for d in docs]
        if signature != initial:
            st.rerun()
        for doc in docs:
            progress = doc.get('progress')
            suffix = f" · {progress['completed']}/{progress['total']} стр." if progress and doc['status'] not in ('ready', 'ready_with_warnings') else ''
            st.write(doc['name'] + ' — ' + STATUS.get(doc['status'], doc['status']) + suffix)
            if doc.get('processing_error'):
                st.error(doc['processing_error'])
                if st.button('Повторить обработку', key='reprocess_'+doc['id']):
                    call('POST', '/documents/'+doc['id']+'/reprocess')
                    st.rerun()
    except (requests.RequestException, ValueError) as exc:
        st.error(str(exc) if isinstance(exc, ValueError) else 'Не удалось обновить документы.')


try:
    health = call('GET', '/health').json()
    chats = call('GET', '/chats').json()
    with st.sidebar:
        if st.button('＋ Новый чат', type='primary', use_container_width=True):
            select_chat(call('POST', '/chats', json={}).json())
            st.rerun()
        st.divider()
        active = st.session_state.get('active_chat')
        if active is None and chats:
            select_chat(chats[0])
            active = st.session_state['active_chat']
        for chat in chats:
            selected = active == (chat['project_id'], chat.get('conversation_id'))
            chat_key = chat['project_id']+'_'+str(chat.get('conversation_id'))
            name_column, delete_column = st.columns([5, 1])
            with name_column:
                if st.button(chat['title'], key='chat_'+chat_key,
                             type='primary' if selected else 'secondary', use_container_width=True):
                    select_chat(chat)
                    st.rerun()
            with delete_column:
                if st.button('🗑', key='delete_chat_'+chat_key, help='Удалить чат из списка', use_container_width=True):
                    conversation = chat.get('conversation_id') or 'empty'
                    call('DELETE', f"/chats/{chat['project_id']}/{conversation}")
                    if selected:
                        st.session_state.pop('active_chat', None)
                        st.session_state.pop('source_view', None)
                    st.rerun()
    title, picker = st.columns([3, 2], vertical_alignment='center')
    with title:
        st.title('Варяг')
    with picker:
        cache = st.session_state.get('available_models')
        if cache is None or cache['expires'] < time.monotonic():
            try:
                payload = call('GET', '/provider/models').json()
                cache = {'models': payload['models'], 'limits': payload.get('limits',{}), 'expires': time.monotonic()+60}
                st.session_state['available_models'] = cache
            except (requests.RequestException, ValueError):
                cache = {'models': [health['model']] if health.get('model') else []}
                st.warning('Не удалось получить список моделей.')
        options = cache['models']
        active = st.session_state.get('active_chat')
        model_key = 'model_'+str(active)
        previous = next((c.get('model_id') for c in chats if (c['project_id'], c.get('conversation_id')) == active), None)
        default = previous or health.get('model')
        current_model=st.session_state.get(model_key,default)
        profile=cache.get('limits',{}).get(current_model,{})
        model_help=(f"Бюджет контекста: {profile['context_tokens']:,} токенов, включая ответ. "
                    + ('Для этой модели используется консервативный лимит.' if profile.get('limit_source')=='conservative_fallback'
                       else 'Объём фрагментов подбирается автоматически.')) if profile else None
        model = st.selectbox('Модель', options, index=options.index(default) if default in options else 0,
                             key=model_key, disabled=not options, help=model_help) if options else None
    if not health.get('gigachat_configured'):
        st.error('Подключение GigaChat не настроено на сервере.')
    if active:
        project, conversation = active
        docs = call('GET', f'/projects/{project}/documents').json()
    else:
        project, conversation, docs = None, None, []
    with st.expander('Загрузить документы', expanded=not docs):
        upload_key = 'files_'+str(active)+'_'+str(st.session_state.get('upload_epoch', 0))
        files = st.file_uploader('Выберите файлы', type=['pdf', 'png', 'jpg', 'jpeg', 'tif', 'tiff', 'txt', 'md'],
                                 accept_multiple_files=True, key=upload_key)
        if st.button('Обработать', type='primary', disabled=not files):
            if project is None:
                created = call('POST', '/chats', json={}).json()
                select_chat(created)
                project = created['project_id']
            errors = []
            with st.spinner('Загружаю документы…'):
                for file in files:
                    results = call('POST', f'/projects/{project}/documents',
                                   files={'file': (file.name, file.getvalue())}).json()
                    errors.extend(file.name+': '+r['error'] for r in results if r.get('error'))
            st.session_state['upload_errors'] = errors
            st.session_state['upload_epoch'] = st.session_state.get('upload_epoch', 0)+1
            st.rerun()
    for error in st.session_state.pop('upload_errors', []):
        st.error(error)
    if docs:
        with st.expander(f'Документы · {len(docs)}', expanded=any(d['status'] not in ('ready', 'ready_with_warnings') for d in docs)):
            processing_status(project, [(d['id'], d['status']) for d in docs])
    if project:
        messages(project, conversation)
    ready = any(d['status'] in ('ready', 'ready_with_warnings') for d in docs)
    busy = any(d['status'] not in ('ready', 'ready_with_warnings', 'failed') for d in docs)
    question = st.chat_input('Задайте вопрос или попросите краткое содержание',
                             disabled=busy or not model or not health.get('gigachat_configured'), max_chars=2000)
    if question:
        send_question(project, conversation, question, model)
        st.rerun()
    if source := st.session_state.get('source_view'):
        with st.expander(source['document_name']+' · стр. '+str(source['page_number']), expanded=True):
            path = '/documents/'+source['document_id']
            params = {'version': source['document_version_id']}
            original = call('GET', path+'/original', params=params).content
            st.download_button('Скачать документ', original, source['document_name'])
            if source['document_name'].lower().endswith(('.pdf','.png','.jpg','.jpeg','.tif','.tiff')):
                st.image(call('GET',path+'/pages/'+str(source['page_index']+1)+'/image',
                              params=dict(params,block=source['block_id'])).content)
            else:
                st.text(source['text_exact'])
            if st.button('Закрыть документ'):
                st.session_state.pop('source_view',None)
                st.rerun()
except (requests.RequestException, ValueError) as exc:
    st.error(str(exc) if isinstance(exc, ValueError) else 'Не удалось подключиться к серверу.')
