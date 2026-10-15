"""Local-only embedding adapter; no network downloads during ingestion or search."""
from typing import Protocol

MODEL_ID = 'intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3'


class EmbeddingProvider(Protocol):
    def encode(self, texts: list[str], query: bool = False) -> list[list[float]]: ...


def local_encoder(directory):
    from document_rag.engine import Encoder
    return Encoder(str(directory))
