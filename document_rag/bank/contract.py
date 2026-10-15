"""OpenAPI 3.1 inventory plus typed core contracts; generated from actual routes."""
import re


def specification(app):
    paths = {}
    for rule in app.url_map.iter_rules():
        if not rule.rule.startswith('/api/v1'):
            continue
        path = re.sub(r'<(?:int:)?([^>]+)>', r'{\1}', rule.rule)
        parameters = [{'name': name, 'in': 'path', 'required': True, 'schema': {'type': 'integer' if name == 'number' else 'string'}} for name in rule.arguments]
        paths.setdefault(path, {}).update({method.lower(): {'operationId': rule.endpoint + '_' + method.lower(), 'parameters': parameters,
            'responses': {'200': {'description': 'Successful response'}, '400': {'description': 'Invalid request'},
                          '401': {'description': 'Missing local token'}, '404': {'description': 'Missing or inaccessible resource'}}}
                      for method in sorted(rule.methods - {'HEAD', 'OPTIONS'})})
    question = paths['/api/v1/projects/{project}/questions']['post']
    question['requestBody'] = {'required': True, 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/QuestionRequest'}}}}
    question['responses']['202'] = {'description': 'Persisted question job', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/QuestionJob'}}}}
    question['responses']['201'] = {'description': 'Synchronous answer when async=false'}
    upload = paths['/api/v1/projects/{project}/documents']['post']
    upload['requestBody'] = {'required': True, 'content': {'multipart/form-data': {'schema': {'type': 'object', 'properties': {
        'files': {'type': 'array', 'items': {'type': 'string', 'format': 'binary'}}, 'file': {'type': 'string', 'format': 'binary'},
        'document_id': {'type': 'string'}, 'external_allowed': {'type': 'string', 'enum': ['true', 'false']}}}}}}
    upload['responses']['202'] = {'description': 'Per-file accepted/error results'}
    search = paths['/api/v1/projects/{project}/search']['post']
    search['description'] = 'Local hybrid retrieval and cross-encoder reranking. No LLM calls. ACL checked before and after inference.'
    search['requestBody'] = {'required': True, 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/SearchRequest'}}}}
    search['responses']['200']['content'] = {'application/json': {'schema': {'$ref': '#/components/schemas/SearchResult'}}}
    return {'openapi': '3.1.0', 'info': {'title': 'Bank Audit Local Search', 'version': '0.3.0'}, 'paths': paths,
        'security': [{'LocalToken': []}], 'components': {'securitySchemes': {'LocalToken': {'type': 'apiKey', 'in': 'header', 'name': 'X-RAG-Token'}},
        'schemas': {
            'SearchRequest': {'type': 'object', 'required': ['query'], 'additionalProperties': False, 'properties': {
                'query': {'type': 'string', 'minLength': 1, 'maxLength': 2000},
                'top_k': {'type': 'integer', 'minimum': 1, 'maximum': 50, 'default': 10},
                'as_of': {'type':['string','null'],'format':'date'},
                'filters': {'type':'object','description':'Exact registry metadata filters; tags require all specified values.'},
                'document_ids': {'type': ['array', 'null'], 'minItems': 1, 'maxItems': 200, 'items': {'type': 'string'}}}},
            'SearchResult': {'type': 'object', 'required': ['sources', 'coverage_manifest', 'index_version', 'normative'], 'properties': {
                'sources': {'type': 'array', 'items': {'type': 'object', 'required': ['block_id', 'document_id', 'text_exact', 'page_number'], 'properties': {
                    'block_id': {'type': 'string'}, 'document_id': {'type': 'string'}, 'document_version_id': {'type': 'string'},
                    'text_exact': {'type': 'string'}, 'page_number': {'type': 'string'},
                    'rerank_score': {'type': 'number', 'description': 'Relevance logit, not probability'},
                    'selection_reason': {'type': 'string'}}}},
                'coverage_manifest': {'type': 'object'}, 'index_version': {'type': 'integer'}, 'normative': {'type': 'boolean'}}},
            'QuestionRequest': {'type': 'object', 'required': ['question'], 'properties': {
                'question': {'type': 'string', 'minLength': 1, 'maxLength': 2000}, 'use_llm': {'type': 'boolean', 'default': False},
                'async': {'type': 'boolean', 'default': True}, 'conversation_id': {'type': ['string', 'null']},
                'as_of': {'type': ['string', 'null'], 'format': 'date'}}},
            'QuestionJob': {'type': 'object', 'required': ['id', 'status', 'conversation_id'], 'properties': {
                'id': {'type': 'string'}, 'conversation_id': {'type': 'string'},
                'status': {'type': 'string', 'enum': ['queued', 'running', 'done', 'failed']}, 'error': {'type': ['string', 'null']},
                'answer': {'type': 'object'}}}}}}
