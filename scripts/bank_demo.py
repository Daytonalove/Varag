"""Seed only synthetic files; never enables external transmission."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from document_rag.bank.store import Store
from document_rag.config import ROOT


def main():
    store = Store(os.environ.get('BANK_DATABASE', str(ROOT / 'data/bank/audit.sqlite3')))
    user, name = 'local-auditor', 'Демо · Кредитный комитет'
    existing = next((p for p in store.projects(user) if p['name'] == name), None)
    project = existing or store.create_project(name, user)
    for path in sorted((ROOT / 'fixtures/synthetic_documents').glob('*.txt')):
        store.upload(project['id'], user, path.name, path.read_bytes())
    print('Создан/обновлён синтетический проект. Файлы в очереди; обработает запущенный manage.py bank.')
    print('Внешняя передача документов остаётся выключенной.')


if __name__ == '__main__':
    main()
