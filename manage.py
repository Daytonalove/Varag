"""Local project lifecycle. No API key is put into subprocess arguments."""
import argparse
import errno
from contextlib import closing
import getpass
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request

from document_rag.config import ROOT, audit, load, local_token


def bank_settings():
    path = ROOT / 'config.json'
    return json.loads(path.read_text()).get('bank', {}) if path.exists() else {}


def bank_database():
    value = os.environ.get('BANK_DATABASE') or bank_settings().get('database') or 'data/bank/audit.sqlite3'
    return str((ROOT / Path(value).expanduser()).resolve())


def forward_bank_workspace():
    """A local checkout can delegate bank commands to one canonical workspace."""
    if len(sys.argv) < 2 or sys.argv[1] not in ('bank', 'bank-backup', 'bank-restore'):
        return
    target = bank_settings().get('workspace')
    if not target:
        return
    target = (ROOT / Path(target).expanduser()).resolve()
    if target == ROOT.resolve():
        return
    config = target / 'config.json'
    destination = json.loads(config.read_text()).get('bank', {}) if config.exists() else {}
    if destination.get('workspace') and (target / destination['workspace']).resolve() != target:
        raise ValueError('Цепочка перенаправлений рабочих папок запрещена.')
    python = target / '.venv/bin/python'
    entry = target / 'manage.py'
    if not python.is_file() or not entry.is_file():
        raise ValueError('Основная рабочая папка Варяга недоступна: ' + str(target))
    print('Основной проект Варяга: ' + str(target), flush=True)
    os.chdir(target)
    os.execv(str(python), [str(python), str(entry), *sys.argv[1:]])


def configure():
    path = Path(os.environ.get('RAG_CONFIG', ROOT / 'config.json')).resolve()
    if not path.exists():
        shutil.copyfile(ROOT / 'config.example.json', path)
    cfg = json.loads(path.read_text())
    key = getpass.getpass('Authorization key (ввод скрыт; Enter — оставить текущий): ').strip()
    if key:
        if any(c.isspace() for c in key):
            raise ValueError('Нужен Authorization key без пробелов и префикса Basic.')
        keypath = (path.parent / Path(cfg['gateway']['key_file']).expanduser()).resolve()
        keypath.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(keypath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w') as output:
            output.write(key)
    model = input('ID модели [GigaChat-2; Enter — оставить текущую]: ').strip()
    if model:
        cfg['gateway']['model'] = model
    ca = input('Путь к PEM с доверенными CA (Enter — оставить текущий): ').strip()
    if ca:
        capath = (path.parent / Path(ca).expanduser()).resolve()
        ssl.create_default_context().load_verify_locations(cafile=capath)
        cfg['gateway']['ca_file'] = str(capath)
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
    print('Настройки сохранены. После изменения config.json перезапустите приложение.')


def download_model(cfg):
    from huggingface_hub import snapshot_download
    r = cfg['retrieval']
    snapshot_download(repo_id=r['model_id'], revision=r['model_revision'], local_dir=r['model_directory'],
                      allow_patterns=['*.json', '*.txt', '*.model', '*.safetensors', '1_Pooling/*'],
                      ignore_patterns=['onnx/*', 'openvino/*'])
    from document_rag.engine import Encoder
    Encoder(r['model_directory']).encode(['Проверка модели'])
    print('Модель поиска сохранена и проверена. Дальнейший поиск работает локально.')


def doctor(cfg, live=False):
    import importlib.metadata
    audit()
    print('Python:', sys.version.split()[0])
    print('База:', cfg['storage']['database'])
    print('Поиск:', cfg['retrieval']['mode'], '| OCR:', cfg['ocr']['enabled'])
    for package in ('Flask', 'streamlit', 'sentence-transformers', 'torch', 'pypdf'):
        print(package + ':', importlib.metadata.version(package))
    import torch
    if torch.version.cuda or torch.version.hip:
        raise ValueError('Нужна CPU-сборка torch.')
    if cfg['ocr']['enabled']:
        for program in ('tesseract', 'pdftoppm'):
            if not shutil.which(program):
                raise ValueError('Не найден ' + program + '. Установите OCR согласно README.')
        result = subprocess.run(['tesseract', '--list-langs'], capture_output=True, text=True, check=True)
        installed = set((result.stdout + result.stderr).split())
        missing = set(cfg['ocr']['languages'].split('+')) - installed
        if missing:
            raise ValueError('Не установлены языки OCR: ' + ', '.join(missing))
        print('OCR и языки: OK')
    if cfg['retrieval']['mode'] == 'hybrid':
        from document_rag.engine import Encoder
        Encoder(cfg['retrieval']['model_directory']).encode(['Проверка поиска'])
        print('Локальная модель: OK')
    if cfg['gateway'].get('ca_file'):
        ssl.create_default_context().load_verify_locations(cafile=cfg['gateway']['ca_file'])
        print('PEM CA: читается; TLS проверяется при соединении')
    if live:
        from document_rag.provider import GigaChat
        models = GigaChat(cfg['gateway']).models()
        print('Доступные модели:', ', '.join(models))
        if cfg['gateway']['model'] not in models:
            raise ValueError('Выбранная модель отсутствует. Исправьте gateway.model в config.json.')
        print('OAuth и /models: OK. Генерация ответа этой командой не проверяется.')
    print('Проверка завершена.')


def run_servers(cfg, api_port, ui_port, bank=False):
    if api_port == ui_port or not all(1024 <= p <= 65535 for p in (api_port, ui_port)):
        raise ValueError('Нужны два разных порта в диапазоне 1024–65535.')
    for port in (api_port, ui_port):
        with socket.socket() as sock:
            try:
                sock.bind(('127.0.0.1', port))
            except OSError as exc:
                if exc.errno == errno.EADDRINUSE:
                    raise ValueError(f'Порт {port} занят. Остановите старый сервер или выберите другой порт.') from None
                raise ValueError(f'Нельзя открыть локальный порт {port}: {type(exc).__name__} (errno={exc.errno}). Проверьте ограничения среды.') from None
    token = local_token(Path(cfg['storage']['database']).parent)
    env = dict(os.environ, RAG_API_PORT=str(api_port), RAG_API_URL=f'http://127.0.0.1:{api_port}',
               PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
    if bank:
        env.update(BANK_API_PORT=str(api_port), BANK_API_URL=f'http://127.0.0.1:{api_port}', LOCAL_DEMO_MODE='true',
                   BANK_DATABASE=cfg['storage']['database'])
    children = []
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        children.append(subprocess.Popen([sys.executable, '-m', 'document_rag.bank.api' if bank else 'document_rag.api'], cwd=ROOT, env=env))
        for _ in range(100):
            if children[0].poll() is not None:
                raise ValueError('Flask не запустился. Проверьте сообщения выше.')
            try:
                req = urllib.request.Request(env['RAG_API_URL'] + ('/api/v1/health' if bank else '/health'), headers={'X-RAG-Token': token})
                with urllib.request.urlopen(req, timeout=1) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                time.sleep(.2)
        else:
            raise ValueError('Flask не ответил за 20 секунд.')
        children.append(subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', str(ROOT / ('bank_ui.py' if bank else 'ui.py')),
                         '--server.address=127.0.0.1', f'--server.port={ui_port}', '--server.headless=true'], cwd=ROOT, env=env))
        print(f'\nОткройте интерфейс: http://127.0.0.1:{ui_port}\nОстановка: Ctrl+C или Stop Debugging в VS Code.\n', flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(.5)
        raise ValueError('Один из серверов остановился. Оба процесса будут завершены.')
    except KeyboardInterrupt:
        print('\nОстанавливаю Flask и Streamlit…')
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main():
    forward_bank_workspace()
    parser = argparse.ArgumentParser(description='Локальный RAG: Flask + Streamlit + GigaChat')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('configure')
    sub.add_parser('download-model')
    check = sub.add_parser('doctor'); check.add_argument('--live', action='store_true')
    run = sub.add_parser('run'); run.add_argument('--api-port', type=int, default=8765); run.add_argument('--ui-port', type=int, default=8501)
    bank = sub.add_parser('bank', help='Локальный банковский контур (отдельная база, demo-auth)')
    bank.add_argument('--api-port', type=int, default=8766)
    bank.add_argument('--ui-port', type=int, default=8502)
    bank_backup = sub.add_parser('bank-backup', help='Согласованная резервная копия SQLite банковского контура')
    bank_backup.add_argument('path')
    bank_backup.add_argument('--key-file',help='32-byte key: write authenticated AES-256-GCM encrypted backup')
    restore = sub.add_parser('bank-restore',help='Restore encrypted backup to a new SQLite file')
    restore.add_argument('source');restore.add_argument('target');restore.add_argument('--key-file',required=True)
    sub.add_parser('demo')
    backup = sub.add_parser('backup'); backup.add_argument('path')
    sub.add_parser('selftest')
    args = parser.parse_args()
    audit()
    if args.command == 'configure':
        configure(); return
    if args.command == 'bank':
        database = bank_database()
        run_servers({'storage': {'database': database}}, args.api_port, args.ui_port, bank=True)
        return
    if args.command == 'bank-backup':
        from document_rag.bank.store import Store
        store=Store(bank_database())
        if args.key_file:
            from document_rag.bank.operations import encrypted_backup
            encrypted_backup(store,args.path,args.key_file)
        else:
            store.backup(args.path)
        print('Резервная копия банковского контура создана.')
        return
    if args.command == 'bank-restore':
        from document_rag.bank.operations import restore_encrypted_backup
        restore_encrypted_backup(args.source,args.target,args.key_file)
        print('Резервная копия проверена и восстановлена в новый файл.')
        return
    cfg = load()
    if args.command == 'download-model':
        download_model(cfg)
    elif args.command == 'doctor':
        doctor(cfg, args.live)
    elif args.command == 'run':
        run_servers(cfg, args.api_port, args.ui_port)
    elif args.command == 'selftest':
        from scripts.selftest import run
        run(cfg)
    else:
        from document_rag.engine import Engine
        engine = Engine(cfg)
        if args.command == 'demo':
            for file in sorted((ROOT / 'sample-documents').glob('*.txt')):
                result = engine.ingest(file)
                print(result['name'], '— фрагментов:', result['chunks'])
        elif args.command == 'backup':
            engine.backup(args.path)
            print('Копия базы создана:', Path(args.path).resolve())


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        from document_rag.provider import ProviderError
        message = str(exc) if isinstance(exc, (ValueError, ProviderError, RuntimeError)) else type(exc).__name__
        print('ОШИБКА: ' + message, file=sys.stderr)
        sys.exit(1)
