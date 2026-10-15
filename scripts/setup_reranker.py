"""Explicit download of a pinned public model; never sends document content."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requests
from document_rag.config import ROOT
from document_rag.bank.reranking import MODEL_REPO, MODEL_REVISION, MODEL_ID, MODEL_FILES, file_digest


def main():
    directory = ROOT / 'models/mmarco-mMiniLMv2-L12-H384-v1'
    directory.mkdir(parents=True, exist_ok=True)
    response = requests.get(f'https://huggingface.co/api/models/{MODEL_REPO}/revision/{MODEL_REVISION}',
                            params={'blobs': 'true'}, timeout=60)
    response.raise_for_status()
    metadata = response.json()
    if metadata['sha'] != MODEL_REVISION:
        raise RuntimeError('Model revision mismatch')
    remote = {f['rfilename']: f for f in metadata['siblings']}
    hashes = {}
    for name in MODEL_FILES:
        target = directory / name
        expected = remote[name].get('lfs', {}).get('sha256')
        if target.exists() and expected and file_digest(target) == expected:
            print(name + ': verified', flush=True)
        else:
            print('Downloading ' + name, flush=True)
            with requests.get(f'https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}', stream=True, timeout=(30, 120)) as download:
                download.raise_for_status()
                partial = target.with_suffix(target.suffix + '.part')
                with partial.open('wb') as output:
                    for chunk in download.iter_content(1024 * 1024):
                        output.write(chunk)
            if expected and file_digest(partial) != expected:
                raise RuntimeError('Checksum mismatch: ' + name)
            partial.replace(target)
        hashes[name] = file_digest(target)
    (directory / 'bank-model.json').write_text(json.dumps({'model_id': MODEL_ID, 'license': 'Apache-2.0', 'sha256': hashes}, indent=2) + '\n')
    print('Installed: ' + MODEL_ID, flush=True)


if __name__ == '__main__':
    main()
