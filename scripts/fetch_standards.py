"""Fetch public test copies and verify their exact SHA256 before adding to corpus."""
import hashlib
import json
from pathlib import Path
import urllib.request
ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'evaluation/manifest.json').read_text())
    digests={d['file']:d['sha256'] for d in manifest['documents']}
    for name,url in manifest['standard_sources'].items():
        destination=ROOT/'evaluation/corpus'/name
        if destination.exists():
            data=destination.read_bytes()
        else:
            with urllib.request.urlopen(url,timeout=90) as response:data=response.read(20_000_001)
        if len(data)>20_000_000 or not data.startswith(b'%PDF') or hashlib.sha256(data).hexdigest()!=digests[name]:
            raise ValueError('Копия изменилась или не является ожидаемым PDF: '+name+'. Сверьте источник и редакцию; файл не записан.')
        if not destination.exists():destination.write_bytes(data)
        print(name,'SHA256 OK')
    print('В evaluation/corpus теперь 50 документов. Источники копий указаны в evaluation/manifest.json.')

if __name__=='__main__':main()
