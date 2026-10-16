"""Offline, pinned multilingual cross-encoder with bounded token windows.

Scores are relevance logits, NOT probabilities or factual confidence. Each
result refers to the unchanged canonical block and an exact character span.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
from typing import Protocol

MODEL_REPO = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
MODEL_REVISION = '1427fd652930e4ba29e8149678df786c240d8825'
MODEL_ID = MODEL_REPO + '@' + MODEL_REVISION
MODEL_FILES = ('config.json', 'model.safetensors', 'sentencepiece.bpe.model',
               'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json')


@dataclass(frozen=True)
class RankScore:
    score: float
    start: int
    end: int
    windows: int = 1
    total_windows: int = 1
    query_truncated: bool = False


class RerankerProvider(Protocol):
    model_id: str

    def score(self, question: str, texts: list[str]) -> list[RankScore]: ...


def file_digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def verify_model(directory):
    directory = Path(directory)
    try:
        manifest = json.loads((directory / 'bank-model.json').read_text())
        if manifest['model_id'] != MODEL_ID:
            raise ValueError('Другая версия модели.')
        for name in MODEL_FILES:
            if file_digest(directory / name) != manifest['sha256'][name]:
                raise ValueError('Контрольная сумма модели не совпала.')
    except (OSError, KeyError, ValueError) as exc:
        raise RuntimeError('Реранкер не установлен или повреждён. Выполните .venv/bin/python scripts/setup_reranker.py.') from exc


class LocalCrossEncoder:
    model_id = MODEL_ID

    def __init__(self, directory, *, batch_size=8, max_length=384, max_windows=12):
        verify_model(directory)
        from sentence_transformers import CrossEncoder
        import torch
        # Avoid severe oversubscription when search and indexing run together.
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        self.model = CrossEncoder(str(directory), device='cpu', local_files_only=True,
                                  trust_remote_code=False, max_length=max_length,
                                  model_kwargs={'use_safetensors': True})
        self.lock = threading.Lock()
        self.batch_size, self.max_length, self.max_windows = batch_size, max_length, max_windows

    def score(self, question, texts):
        if not texts:
            return []
        import torch
        with self.lock:
            tokenizer = self.model.tokenizer
            query_ids = tokenizer(question, add_special_tokens=False, verbose=False)['input_ids']
            qids = query_ids[:96]
            query = tokenizer.decode(qids, skip_special_tokens=True)
            width = self.max_length - len(qids) - tokenizer.num_special_tokens_to_add(pair=True) - 4
            pairs, owners, windows = [], [], []
            for index, text in enumerate(texts):
                offsets = tokenizer(text, add_special_tokens=False, truncation=False,
                                    return_offsets_mapping=True, verbose=False)['offset_mapping']
                starts = list(range(0, max(1, len(offsets) - width + 1), max(1, width - 64)))
                tail = max(0, len(offsets) - width)
                if starts[-1] != tail:
                    starts.append(tail)
                count = len(starts)
                if count > self.max_windows:
                    # Sample the entire block, including its end; report incomplete scoring.
                    starts = [starts[round(i * (count - 1) / (self.max_windows - 1))] for i in range(self.max_windows)]
                windows.append((len(starts), count))
                for start in starts:
                    begin = offsets[start][0] if offsets else 0
                    end = offsets[min(start + width, len(offsets)) - 1][1] if offsets else len(text)
                    pairs.append((query, text[begin:end]))
                    owners.append((index, begin, end))
            values = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False,
                                        activation_fn=torch.nn.Identity(), convert_to_numpy=True).reshape(-1).tolist()
            if len(values) != len(pairs) or not all(math.isfinite(float(v)) for v in values):
                raise ValueError('Невалидный ответ локального реранкера.')
            best = [None] * len(texts)
            for value, (index, start, end) in zip(values, owners):
                if best[index] is None or value > best[index].score:
                    best[index] = RankScore(float(value), start, end, *windows[index], len(query_ids) > 96)
            return best
