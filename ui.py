"""Streamlit is the browser UI; Flask owns documents, search, history and GigaChat."""
import os
from pathlib import Path
import time
import urllib.parse

from document_rag.config import audit, load, local_token
audit()
import requests
import streamlit as st

st.set_page_config(page_title='Документы и ответы', page_icon='📚', layout='wide')


def call(method, path, *, raw=False, **kwargs):
    try:
        response = requests.request(method, API + path, headers={'X-RAG-Token': TOKEN},
                                    timeout=(5, 360), allow_redirects=False, **kwargs)
    except requests.RequestException:
        raise RuntimeError('Flask недоступен или истекло время ожидания. Проверьте терминал запуска.') from None
    if not response.ok:
        try:
            message = response.json().get('error', 'Ошибка сервера')
        except ValueError:
            message = f'HTTP {response.status_code}'
        raise RuntimeError(message)
    return response.content if raw else response.json()


def new_chat():
    st.session_state.chat_id = None


def resume_chat(chat_id):
    try:
        chat = call('GET', '/chats/' + chat_id)
        scope = chat['turns'][-1]['scope'] if chat['turns'] else {'documents': [], 'compare': False}
        st.session_state.selected_docs = scope['documents']
        st.session_state.compare = scope['compare']
        st.session_state.chat_id = chat_id
    except RuntimeError as exc:
        st.session_state.notice = str(exc)


def show_answer(answer):
    if answer.get('status') == 'answered':
        st.caption('Цитаты найдены в источниках. Правильность выводов модели автоматически не подтверждена.')
    coverage = answer.get('retrieval_coverage')
    if coverage:
        st.caption(f"В ответ отобраны фрагменты из {coverage['represented_documents']} из {coverage['selected_documents']} выбранных документов. Это не полная проверка набора.")
        if coverage['unrepresented_documents']:
            with st.expander('Документы, не представленные в контексте ответа'):
                st.write(coverage['unrepresented_documents'])
    if answer.get('status') != 'answered':
        st.warning(answer.get('reason') or 'Недостаточно подтверждённых данных.')
    sources = {s['source_id']: s for s in answer.get('sources', [])}
    for claim in answer.get('claims', []):
        st.write(claim['text'])
        for proof in claim['evidence']:
            source = sources[proof['source_id']]
            with st.expander(f"[{proof['source_id']}] {source['document']} · {source['locator']}"):
                st.text(proof['quote'])
    if answer.get('search_question') != answer.get('original_question'):
        st.caption('Поисковый вопрос: ' + answer.get('search_question', ''))
    with st.expander('Найденные фрагменты и ограничения проверки'):
        st.caption(answer.get('verification', ''))
        st.caption('Поиск выбирает фрагменты. Это не полная сверка всех пунктов документов.')
        for doc in answer.get('extraction_warnings', []):
            st.warning(doc['name'] + ': ' + '; '.join(doc['warnings']))
        for source in sources.values():
            st.caption(f"[{source['source_id']}] {source['document']} · {source['locator']}")
            st.text(source['text'])


st.title('Документы и ответы')
st.caption('Загрузите документы, выберите источники и задайте вопрос. Ответы содержат проверяемые цитаты.')
try:
    cfg = load()
    TOKEN = local_token(Path(cfg['storage']['database']).parent)
    API = os.environ.get('RAG_API_URL', 'http://127.0.0.1:8765').rstrip('/')
    parts = urllib.parse.urlsplit(API)
    if parts.scheme != 'http' or parts.hostname not in ('127.0.0.1', 'localhost') or parts.username or parts.password:
        raise ValueError('Интерфейс работает только с локальным Flask API.')
    health = call('GET', '/health')
    documents = call('GET', '/documents')
except (ValueError, RuntimeError) as exc:
    st.error(str(exc))
    st.code('.venv/bin/python manage.py run', language='bash')
    st.stop()

st.session_state.setdefault('chat_id', None)
st.session_state.setdefault('selected_docs', [d['id'] for d in documents])
st.session_state.setdefault('compare', False)
if notice := st.session_state.pop('notice', None):
    st.info(notice)
known = {d['id']: d for d in documents}
st.session_state.selected_docs = [i for i in st.session_state.selected_docs if i in known]
if len(st.session_state.selected_docs) < 2:
    st.session_state.compare = False

with st.sidebar:
    st.header('Документы')
    files = st.file_uploader('PDF, DOCX, текст или изображения', type=['pdf', 'docx', 'txt', 'md', 'png', 'jpg', 'jpeg', 'tif', 'tiff'],
                             accept_multiple_files=True)
    st.caption('До 25 MiB на файл и до 200 страниц PDF. Одинаковое имя заменяет предыдущую версию.')
    if st.button('Загрузить и обработать', disabled=not files):
        added, failures = [], []
        for file in files:
            try:
                with st.spinner('Обрабатывается ' + file.name + '…'):
                    job_id = call('POST', '/documents', files={'file': (file.name, file.getvalue())})['job_id']
                    deadline = time.monotonic() + 19000
                    while time.monotonic() < deadline:
                        job = call('GET', '/jobs/' + job_id)
                        if job['status'] in ('done', 'failed'):
                            break
                        time.sleep(1)
                    else:
                        raise RuntimeError('Обработка ещё выполняется. Обновите страницу позже.')
                    if job['status'] == 'failed':
                        raise RuntimeError(job['error'])
                    added.append(job['result']['id'])
            except RuntimeError as exc:
                failures.append(file.name + ': ' + str(exc))
        st.session_state.selected_docs = list(dict.fromkeys(st.session_state.selected_docs + added))
        if added:
            new_chat()
        st.session_state.notice = f'Обработано файлов: {len(added)} из {len(files)}. Успешные документы выбраны для поиска.'
        st.session_state.upload_failures = failures
        st.rerun()
    for failure in st.session_state.get('upload_failures', []):
        st.error(failure)
    selected = st.multiselect('Источники ответа', options=list(known), format_func=lambda i: known[i]['name'],
                             key='selected_docs', on_change=new_chat)
    compare = st.checkbox('Сравнить выбранные документы', key='compare', on_change=new_chat, disabled=len(selected) < 2)
    if compare:
        st.caption('От 2 до 8 документов; ответ должен содержать цитаты из каждого.')
    st.button('Новый диалог', on_click=new_chat, use_container_width=True)
    st.caption('Данные и история сохраняются в локальной базе между запусками.')

chat_tab, search_tab, library_tab, history_tab, settings_tab = st.tabs(['Чат', 'Проверить поиск', 'Библиотека', 'История', 'Состояние'])
with chat_tab:
    if not documents:
        st.info('Загрузите документы слева или выполните команду manage.py demo для тестового примера.')
    if st.session_state.chat_id:
        try:
            chat = call('GET', '/chats/' + st.session_state.chat_id)
            for turn in chat['turns']:
                with st.chat_message('user'):
                    st.write(turn['question'])
                with st.chat_message('assistant'):
                    show_answer(turn['answer'])
        except RuntimeError as exc:
            st.error(str(exc))
            st.session_state.chat_id = None
    question = st.chat_input('Задайте вопрос по выбранным документам', disabled=not selected or (compare and len(selected) > 8), max_chars=2000)
    if question:
        try:
            with st.spinner('Ищу фрагменты и проверяю ответ…'):
                answer = call('POST', '/ask', json={'question': question, 'document_ids': selected,
                              'compare': compare, 'chat_id': st.session_state.chat_id})
            st.session_state.chat_id = answer['chat_id']
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))
    st.caption('В GigaChat передаются вопрос, найденные фрагменты и при продолжении диалога — до двух предыдущих пар сообщений. '
               'Запросы расходуют квоту вашего аккаунта. Цитата подтверждает наличие текста, но не гарантирует правильность вывода.')

with search_tab:
    st.write('Проверка поиска без обращения к GigaChat.')
    with st.form('search'):
        query = st.text_input('Поисковый вопрос', max_chars=2000)
        search = st.form_submit_button('Найти фрагменты', disabled=not selected)
    if search and query.strip():
        try:
            with st.spinner('Поиск…'):
                hits = call('POST', '/search', json={'question': query, 'document_ids': selected})
            if not hits:
                st.info('Фрагменты не найдены.')
            for hit in hits:
                st.markdown('**' + hit['document'].replace('*', '') + '** · ' + hit['locator'])
                st.text(hit['text'])
                st.caption('Ранг RRF: ' + format(hit['score'], '.4f') + ' (это не вероятность правильности ответа)')
        except RuntimeError as exc:
            st.error(str(exc))

with library_tab:
    for doc in documents:
        with st.expander(doc['name'] + ' · ' + str(round(doc['bytes'] / 1024, 1)) + ' KiB'):
            for warning in doc['warnings']:
                st.warning(warning)
            if st.button('Подготовить оригинал для скачивания', key='prepare_' + doc['id']):
                try:
                    data = call('GET', '/documents/' + doc['id'] + '/original', raw=True)
                    st.download_button('Скачать оригинал', data, file_name=doc['name'], key='download_' + doc['id'])
                except RuntimeError as exc:
                    st.error(str(exc))
            confirm = st.checkbox('Удалить документ и связанные диалоги', key='confirm_' + doc['id'])
            if st.button('Удалить', key='delete_' + doc['id'], disabled=not confirm):
                try:
                    call('DELETE', '/documents/' + doc['id'])
                    new_chat()
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))

with history_tab:
    try:
        chats = call('GET', '/chats')
        if not chats:
            st.info('Здесь появятся сохранённые диалоги.')
        for chat in chats:
            left, middle, right = st.columns([6, 2, 2])
            left.write(chat['title'])
            left.caption(chat['updated'] + ' UTC')
            middle.button('Продолжить', key='resume_' + chat['id'], on_click=resume_chat, args=(chat['id'],))
            if right.button('Удалить диалог', key='remove_chat_' + chat['id']):
                call('DELETE', '/chats/' + chat['id'])
                if st.session_state.chat_id == chat['id']:
                    new_chat()
                st.rerun()
    except RuntimeError as exc:
        st.error(str(exc))

with settings_tab:
    st.write('Flask работает. Модель ответа: **' + health['model'] + '**')
    st.write('Поиск: **' + health['retrieval'] + '**. OCR: **' + ('включён' if health['ocr'] else 'выключен') + '**.')
    st.caption('Настройки изменяются в config.json; затем перезапустите оба сервера. Ключ хранится отдельно в файле secrets/gigachat.key.')
    if st.button('Проверить доступ к GigaChat и список моделей'):
        try:
            with st.spinner('Проверка API…'):
                result = call('GET', '/provider/models')
            st.success('OAuth и список моделей доступны. Генерация ответа пока не проверена.')
            st.write(result['models'])
            if not result['selected_available']:
                st.error('Выбранная модель отсутствует. Укажите один из доступных ID в gateway.model и перезапустите приложение.')
        except RuntimeError as exc:
            st.error(str(exc))
    st.code('.venv/bin/python manage.py doctor\n.venv/bin/python manage.py selftest', language='bash')
