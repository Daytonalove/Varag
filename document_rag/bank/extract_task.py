"""Resource-bounded extraction child. Input/output remain in a private temporary directory."""
import json
from pathlib import Path
import resource
import sys

from .ingest import extract_pages


def main():
    source, destination = Path(sys.argv[1]), Path(sys.argv[2])
    resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
    resource.setrlimit(resource.RLIMIT_FSIZE, (150 * 1024 * 1024, 150 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    if sys.platform.startswith('linux'):
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    def progress(stage):
        destination.with_suffix('.stage').write_text(stage)
    try:
        pages = extract_pages(source, on_stage=progress)
        destination.write_text(json.dumps(pages, ensure_ascii=False))
    except Exception:
        raise SystemExit(2) from None


if __name__ == '__main__':
    main()
