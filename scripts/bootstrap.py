"""Install only into the project venv. Python 3.12 recommended, macOS/Linux CPU."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / '.venv'
PYTHON = VENV / 'bin/python'


def run(args):
    print('\n▶ ' + ' '.join(map(str, args)), flush=True)
    with subprocess.Popen(list(map(str, args)), cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True) as process:
        with (ROOT / 'bootstrap.log').open('a') as log:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
        if process.wait():
            raise RuntimeError('Установка остановлена. Вывод сохранён в bootstrap.log.')


def main():
    if platform.system() not in ('Darwin', 'Linux'):
        raise RuntimeError('Этот выпуск предназначен для macOS/Linux.')
    if sys.version_info[:2] not in ((3, 11), (3, 12), (3, 13)):
        raise RuntimeError('Запустите установку через Python 3.12 (поддерживаются 3.11–3.13).')
    if sys.prefix != sys.base_prefix and Path(sys.prefix).resolve() != VENV.resolve():
        print('Создаётся отдельное окружение проекта; текущее окружение не изменяется.')
    run([sys.executable, '-m', 'venv', '--without-pip', VENV])
    if subprocess.run([PYTHON, '-m', 'pip', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        if subprocess.run([sys.executable, '-m', 'pip', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            run([sys.executable, '-m', 'pip', '--python', PYTHON, 'install', 'pip'])
        else:
            run([PYTHON, '-m', 'ensurepip', '--upgrade'])
    # Resolve and audit the complete dependency plan BEFORE installation.
    torch_args = ['torch==2.14.1']
    if platform.system() == 'Linux':
        torch_args += ['--index-url', 'https://download.pytorch.org/whl/cpu']
    plans = [(torch_args, 'torch-plan.json'),
             (['-r', ROOT / 'requirements.txt', '-c', ROOT / 'constraints.txt'], 'requirements-plan.json')]
    for args, report in plans:
        report_path = ROOT / report
        run([PYTHON, '-m', 'pip', 'install', '--dry-run', '--report', report_path, *args])
        entries = json.loads(report_path.read_text())['install']
        forbidden = [x['metadata']['name'] for x in entries if 'openai' in x['metadata']['name'].lower()
                     or x['metadata']['name'].lower().startswith('nvidia-') or x['metadata']['name'].lower() == 'tiktoken']
        if forbidden:
            raise RuntimeError('Запрещённые зависимости: ' + ', '.join(forbidden))
        run([PYTHON, '-m', 'pip', 'install', *args])
        report_path.unlink(missing_ok=True)
    run([PYTHON, '-m', 'pip', 'check'])
    run([PYTHON, '-c', 'from document_rag.config import audit; audit()'])
    config = ROOT / 'config.json'
    if not config.exists():
        shutil.copyfile(ROOT / 'config.example.json', config)
    print('\nГотово. Далее: .venv/bin/python manage.py download-model')
    print('Настройка API: .venv/bin/python manage.py configure')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError) as exc:
        print('\nОШИБКА: ' + str(exc), file=sys.stderr)
        sys.exit(1)
