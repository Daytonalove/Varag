"""Explicit local administration actions; authentication stays on the API."""
import json
import streamlit as st


def render(call,project,docs):
    names={d['id']:d['name'] for d in docs}
    st.subheader('Реестр документов')
    if names:
        document=st.selectbox('Документ в реестре',list(names),format_func=names.get,key='registry_document')
        if st.button('Открыть карточку реестра'):
            st.session_state['registry']=dict(document=document,project=project,**call('GET','/documents/'+document+'/registry').json())
        registry=st.session_state.get('registry')
        if registry and registry['document']==document and registry['project']==project:
            st.caption('Автоматические предположения требуют подтверждения. Даты вводятся в формате YYYY-MM-DD.')
            st.json(registry['proposal'])
            if st.button('Принять предложенные метаданные'):
                payload=dict(registry['proposal'],verification_status='confirmed')
                call('PATCH','/documents/'+document+'/metadata',json=payload)
                st.session_state.pop('registry',None)
                st.rerun()
            with st.form('metadata_form_'+document):
                from .registry import TEXT_FIELDS,DATE_FIELDS
                payload={}
                values=registry['metadata']
                labels={'title':'Название','document_type':'Тип документа','document_number':'Номер','revision':'Редакция',
                        'approving_body':'Утвердивший орган','department':'Подразделение','language':'Язык','upload_source':'Источник загрузки',
                        'product':'Продукт','status':'Статус','approved_at':'Дата утверждения','effective_from':'Действует с','effective_until':'Действует до'}
                for key in sorted(TEXT_FIELDS|DATE_FIELDS):
                    value=st.text_input(labels[key],value=values.get(key) or '',key='metadata_'+document+'_'+key)
                    payload[key]=value or (None if key in DATE_FIELDS else '')
                tags=st.text_input('Метки',value=', '.join(values.get('tags',[])))
                payload['tags']=[t.strip() for t in tags.split(',') if t.strip()]
                payload['verification_status']=st.selectbox('Подтверждение',['proposed','confirmed','rejected'],index=['proposed','confirmed','rejected'].index(values.get('verification_status','proposed')))
                if st.form_submit_button('Сохранить карточку'):
                    call('PATCH','/documents/'+document+'/metadata',json=payload)
                    st.session_state.pop('registry',None)
                    st.rerun()
            with st.expander('Связать с другим документом'):
                targets=[i for i in names if i!=document]
                if targets and registry['blocks']:
                    from .registry import RELATIONS
                    target=st.selectbox('Документ-основание изменяет документ',targets,format_func=names.get)
                    kind=st.selectbox('Тип связи',sorted(RELATIONS))
                    blocks={b['id']:b for b in registry['blocks']}
                    source=st.selectbox('Пункт-основание',list(blocks),format_func=lambda k:blocks[k]['text_exact'][:160])
                    when=st.text_input('Дата вступления связи в силу')
                    confirmed=st.checkbox('Связь проверена по оригиналу')
                    if st.button('Создать связь'):
                        call('POST','/relations',json=dict(source_document_id=document,target_document_id=target,kind=kind,
                             source_block_id=source,effective_from=when or None,confirmed=confirmed))
                        st.success('Связь сохранена с пунктом-основанием.')
        with st.expander('Доступ к выбранному документу'):
            restricted=st.checkbox('Ограничить доступ списком участников')
            ids=st.text_input('Идентификаторы участников через запятую')
            if st.button('Сохранить доступ'):
                call('PUT','/documents/'+document+'/acl',json={'restricted':restricted,'user_ids':[s.strip() for s in ids.split(',') if s.strip()]})
                st.success('Права сохранены.')
    if st.button('Показать связи документов'):
        st.session_state['relations_'+project]=call('GET','/projects/'+project+'/relations').json()
    for relation in st.session_state.get('relations_'+project,[]):
        if relation['source_document_id'] not in names or relation['target_document_id'] not in names:
            continue
        st.write(names[relation['source_document_id']]+' → '+relation['kind']+' → '+names[relation['target_document_id']])
        if st.button('Отозвать связь',key='revoke_'+relation['id']):
            call('DELETE','/relations/'+relation['id'])
            st.session_state.pop('relations_'+project,None)
            st.rerun()
    with st.expander('Участники проекта'):
        st.caption('SSO отложен. Пользователи аутентифицируются локальными серверными токенами.')
        if st.button('Показать участников'):
            st.dataframe(call('GET','/projects/'+project+'/members').json(),hide_index=True)
        member=st.text_input('Идентификатор пользователя')
        role=st.selectbox('Роль',['viewer','auditor','owner','remove'])
        if st.button('Сохранить участника'):
            from urllib.parse import quote
            call('PUT','/projects/'+project+'/members/'+quote(member,safe=''),json={'role':None if role=='remove' else role})
            st.success('Участник обновлён.')
    with st.expander('Расходы и состояние проекта'):
        if st.button('Обновить показатели'):
            value=call('GET','/projects/'+project+'/metrics').json()
            st.metric('Использовано токенов',value['tokens'])
            st.metric('Вызовы / ошибки',str(value['calls'])+' / '+str(value['failed_calls']))
            st.caption('Стоимость: '+(str(value['estimated_cost_rub'])+' руб.' if value['estimated_cost_rub'] is not None else 'тариф не настроен'))
            st.json(value)
    with st.expander('Срок хранения исключённых документов'):
        days=st.number_input('Хранить после исключения, дней',min_value=1,max_value=36500,value=365)
        if st.button('Рассчитать список очистки'):
            st.session_state['retention_'+project]=call('POST','/projects/'+project+'/retention',json={'days':int(days),'execute':False}).json()
        if plan:=st.session_state.get('retention_'+project):
            st.json(plan)
            confirmed=st.checkbox('Удалить содержимое и связанные ответы из указанного списка без восстановления')
            if st.button('Выполнить очистку',disabled=not confirmed):
                st.json(call('POST','/projects/'+project+'/retention',json={'days':int(days),'execute':True}).json())
                st.session_state.pop('retention_'+project,None)
