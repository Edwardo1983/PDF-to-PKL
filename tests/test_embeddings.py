import os
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

import pdf_converter_working as converter_module
from pdf_converter_working import PDFEmbeddingsConverter


class FakeModel:
    def encode(self, sentences: List[str], normalize_embeddings: bool = True):
        return np.ones((len(sentences), 3), dtype=np.float32)


class FakeCollection:
    def __init__(self, name: str, documents: List[str], ids: List[str]):
        self.name = name
        self._documents = documents
        self._ids = ids
        self._metadatas = [{"page_from": index + 1} for index, _ in enumerate(documents)]
        self._distances = [0.1 for _ in documents]
        self.add_called = False
        self.last_n_results = None

    def count(self):
        return len(self._documents)

    def query(self, query_embeddings, n_results: int):
        self.last_n_results = n_results
        limit = min(n_results, len(self._documents))
        return {
            "documents": [self._documents[:limit]],
            "metadatas": [self._metadatas[:limit]],
            "distances": [self._distances[:limit]],
            "ids": [self._ids[:limit]],
        }

    def add(self, *args, **kwargs):  # pragma: no cover - should never be called
        self.add_called = True


class FakeClient:
    def __init__(self, collections):
        self._collections = {collection.name: collection for collection in collections}

    def list_collections(self):
        return list(self._collections.values())

    def get_collection(self, name):
        return self._collections[name]

    def delete_collection(self, name):
        self._collections.pop(name, None)


@pytest.fixture
def converter():
    instance = PDFEmbeddingsConverter.__new__(PDFEmbeddingsConverter)
    instance.model = FakeModel()
    documents = ["Primul document", "Al doilea document", "Al treilea document", "Primul document"]
    ids = ["doc-1", "doc-2", "doc-3", "doc-1"]
    collection = FakeCollection("educatie", documents, ids)
    instance.client = FakeClient([collection])
    return instance


def test_model_probe_runs():
    instance = PDFEmbeddingsConverter.__new__(PDFEmbeddingsConverter)
    instance.model = FakeModel()
    assert instance.test_model() is True


def test_query_no_add_and_no_duplicates(converter):
    result = converter.search_similar("învățare", top_k=5)

    # asigură-te că add nu a fost apelat în timpul căutării
    collection = converter.client.get_collection("educatie")
    assert collection.add_called is False

    returned_ids = result["ids"][0]
    assert len(returned_ids) == len(set(returned_ids))


def test_topk_clamped_to_collection_size(converter, caplog):
    caplog.set_level("INFO")
    converter.search_similar("învățare", top_k=10)
    collection = converter.client.get_collection("educatie")

    assert collection.last_n_results == collection.count()
    assert any("Clamped top_k" in record.message for record in caplog.records)


def test_telemetry_off_when_env_set(monkeypatch):
    instance = PDFEmbeddingsConverter.__new__(PDFEmbeddingsConverter)
    os.environ["CHROMA_TELEMETRY"] = "0"
    captured = {}

    def fake_client(path, settings):
        captured["path"] = path
        captured["settings"] = settings
        return SimpleNamespace(path=path, settings=settings)

    monkeypatch.setattr(converter_module.chromadb, "PersistentClient", fake_client)

    client = instance._create_chroma_client("/tmp/embeddings")
    assert client.path == "/tmp/embeddings"
    assert getattr(captured["settings"], "anonymized_telemetry", None) is False
    monkeypatch.delenv("CHROMA_TELEMETRY", raising=False)
