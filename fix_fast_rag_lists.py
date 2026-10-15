#!/usr/bin/env python3
"""Fix incomplete enumerations in an installed Fast RAG v2 (offline patch).

Usage (project root):
  python3.12 fix_fast_rag_lists.py --check
  python3.12 fix_fast_rag_lists.py
  python3.12 fix_fast_rag_lists.py --restore

Changes only document_rag/bank/fast_generation.py. No GigaChat requests.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys

MARKER_OLD = 'FAST_RAG_VERSION = 2'
MARKER_NEW = 'FAST_RAG_VERSION = 3'
BACKUP_SUFFIX = '.before_list_fix'

INSTRUCTION = (
    'Если пользователь просит перечислить определённое число пунктов, '
    'найди и перечисли ВСЕ пункты из источников в исходном порядке. '
    'Каждый пункт перечисления оформи отдельным объектом claims с кратким пояснением. '
    'Не останавливайся на шести. Если источники не содержат полного списка, '
    'перечисли подтверждённые пункты и объясни неполноту в limitations. '
    'Не придумывай отсутствующие пункты.\n'
)

# Check an explicit list request locally, without a second LLM call.
COUNT_HELPER = '''def _expected_enumeration_count(question):
    """Conservative check for explicit 'перечисли все восемь ...' requests."""
    if not re.search(r'\\b(?:перечисли|назови|укажи|перечислить)\\b', question, re.I):
        return None
    words = {
        'один': 1, 'два': 2, 'две': 2, 'три': 3, 'четыре': 4,
        'пять': 5, 'шесть': 6, 'семь': 7, 'восемь': 8,
        'девять': 9, 'десять': 10, 'одиннадцать': 11,
        'двенадцать': 12, 'тринадцать': 13, 'четырнадцать': 14,
        'пятнадцать': 15, 'шестнадцать': 16, 'семнадцать': 17,
        'восемнадцать': 18, 'девятнадцать': 19, 'двадцать': 20,
    }
    match = re.search(r'\\bвсе\\s+(\\d{1,2}|[а-яё]+)\\b', question, re.I)
    if not match:
        return None
    token = match.group(1).lower()
    count = int(token) if token.isdigit() else words.get(token)
    return count if count is not None and 1 <= count <= 24 else None


'''


def replace_one(s: str, a: str, b: str, label: str) -> str:
    n = s.count(a)
    if n != 1:
        raise ValueError(f'Неожиданный код ({label}): найдено {n} совпадений вместо 1. Файлы не изменены.')
    return s.replace(a, b, 1)


def transform(original: str) -> str:
    s = original
    s = replace_one(s, MARKER_OLD, MARKER_NEW, 'версия Fast RAG')
    s = replace_one(
        s,
        'Отвечай именно на вопрос. Для «чем отличается X от Y»',
        'Отвечай именно на вопрос. ' + INSTRUCTION + 'Для «чем отличается X от Y»',
        'полнота перечислений',
    )
    s = replace_one(s, 'claims — массив из 0–6 объектов', 'claims — массив из 0–24 объектов', 'инструкция claims')
    s = replace_one(s, "'claims': {'type': 'array', 'maxItems': 6,", "'claims': {'type': 'array', 'maxItems': 24,", 'JSON maxItems')
    s = replace_one(s, 'len(claims) > 6 or not isinstance(limitations', 'len(claims) > 24 or not isinstance(limitations', 'проверка количества')
    s = replace_one(s, "'FAST_RAG_SOURCE_CHARS', '1400'", "'FAST_RAG_SOURCE_CHARS', '2400'", 'размер источника')
    s = replace_one(s, "'FAST_RAG_OUTPUT_TOKENS', '1600'", "'FAST_RAG_OUTPUT_TOKENS', '2500'", 'выходные токены')
    s = replace_one(s, 'def analyze_fast(provider, question, sources,', COUNT_HELPER + 'def analyze_fast(provider, question, sources,', 'проверка перечня')
    s = replace_one(
        s,
        "claims, limitations = _validate_response(choice['message']['content'], chosen)\n        meta = {",
        "claims, limitations = _validate_response(choice['message']['content'], chosen)\n"
        "        expected = _expected_enumeration_count(reference_question)\n"
        "        if expected is not None and len(claims) < expected:\n"
        "            limitations.append(\n"
        "                f'В ответе перечислено только {len(claims)} из {expected} запрошенных пунктов; '\n"
        "                'список может быть неполным.'\n"
        "            )\n"
        "        meta = {",
        'предупреждение о неполноте',
    )
    ast.parse(s, filename='fast_generation.py')
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description='Fix enumerations in Fast RAG v2')
    ap.add_argument('--check', action='store_true', help='Проверить без изменения файлов')
    ap.add_argument('--restore', action='store_true', help='Восстановить Fast RAG v2')
    ap.add_argument('project_root', nargs='?', type=Path, default=Path('.'))
    args = ap.parse_args()
    target = args.project_root.resolve() / 'document_rag/bank/fast_generation.py'
    backup = target.with_name(target.name + BACKUP_SUFFIX)
    if not target.is_file():
        print(f'ОШИБКА: не найден {target}. Запусти из корня проекта.', file=sys.stderr)
        return 1
    original = target.read_text(encoding='utf-8')
    if args.restore:
        if not backup.is_file():
            print(f'ОШИБКА: резервная копия не найдена: {backup.name}', file=sys.stderr)
            return 1
        saved = backup.read_text(encoding='utf-8')
        if MARKER_OLD not in saved:
            print('ОШИБКА: резервная копия не относится к Fast RAG v2', file=sys.stderr)
            return 1
        ast.parse(saved)
        target.write_text(saved, encoding='utf-8')
        print('Восстановлен Fast RAG v2. Перезапусти приложение.')
        return 0
    if MARKER_NEW in original:
        print('Исправление перечней уже установлено. Повторно ничего делать не нужно.')
        return 0
    try:
        updated = transform(original)
    except (SyntaxError, ValueError) as exc:
        print('ОШИБКА:', exc, file=sys.stderr)
        return 1
    if args.check:
        print('Проверка пройдена. Совместимо с Fast RAG v2, синтаксис корректен; файлы не изменены.')
        return 0
    if backup.exists():
        print('ОШИБКА: резервная копия уже существует. Не буду её перезаписывать.', file=sys.stderr)
        return 1
    backup.write_bytes(target.read_bytes())
    temporary = target.with_name(target.name + '.tmp_list_fix')
    try:
        temporary.write_text(updated, encoding='utf-8')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print('Установлено исправление перечислений Fast RAG. Один запрос GigaChat сохранён.')
    print('Разрешено до 24 пунктов, контекст до 2400 символов на фрагмент, выход до 2500 токенов по умолчанию.')
    print('Резервная копия:', backup.name)
    print('Перезапусти: .venv/bin/python manage.py bank')
    return 0


if __name__ == '__main__':
    sys.exit(main())
