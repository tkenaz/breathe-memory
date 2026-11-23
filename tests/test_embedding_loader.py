"""
load_embedding_model: short texts must embed correctly on every device.

The unit tests fake sentence-transformers and the MPS probe, so they run anywhere.
The last test uses the real model on a real MPS device and is opt-in:

    BREATHE_TEST_MPS_MODEL=intfloat/multilingual-e5-large pytest tests/test_embedding_loader.py
"""
import os
import sys
import types

import pytest

import memory_nexus.postgres as pg


class FakeSentenceTransformer:
    """Records how it was constructed; optionally rejects eager attention."""

    reject_eager_with = None
    calls: list = []

    def __init__(self, name, **kwargs):
        type(self).calls.append((name, kwargs))
        wants_eager = kwargs.get("model_kwargs", {}).get("attn_implementation") == "eager"
        if wants_eager and type(self).reject_eager_with is not None:
            raise type(self).reject_eager_with("eager attention is not supported")
        self.name, self.kwargs = name, kwargs


@pytest.fixture
def fake_st(monkeypatch):
    FakeSentenceTransformer.calls = []
    FakeSentenceTransformer.reject_eager_with = None
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return FakeSentenceTransformer


def test_plain_load_when_mps_is_absent(fake_st, monkeypatch):
    monkeypatch.setattr(pg, "_mps_available", lambda: False)
    model = pg.load_embedding_model("some/model")
    assert fake_st.calls == [("some/model", {})]
    assert model.kwargs == {}


def test_eager_attention_is_requested_on_mps(fake_st, monkeypatch):
    monkeypatch.setattr(pg, "_mps_available", lambda: True)
    model = pg.load_embedding_model("some/model")
    assert fake_st.calls == [("some/model", {"model_kwargs": {"attn_implementation": "eager"}})]
    assert "device" not in model.kwargs  # stays on MPS


@pytest.mark.parametrize("error", [TypeError, ValueError])
def test_cpu_fallback_when_the_model_rejects_eager_attention(fake_st, monkeypatch, error):
    monkeypatch.setattr(pg, "_mps_available", lambda: True)
    fake_st.reject_eager_with = error
    model = pg.load_embedding_model("some/model")
    assert [kwargs for _, kwargs in fake_st.calls] == [
        {"model_kwargs": {"attn_implementation": "eager"}},
        {"device": "cpu"},
    ]
    assert model.kwargs == {"device": "cpu"}


def test_other_load_errors_are_not_swallowed(fake_st, monkeypatch):
    monkeypatch.setattr(pg, "_mps_available", lambda: True)
    fake_st.reject_eager_with = OSError  # e.g. model not found
    with pytest.raises(OSError):
        pg.load_embedding_model("some/model")
    assert len(fake_st.calls) == 1


def test_store_uses_the_loader(fake_st, monkeypatch):
    monkeypatch.setattr(pg, "_mps_available", lambda: True)
    store = pg.PostgresMemoryStore(dsn="postgresql://unused/unused", model_name="some/model")
    model = store._get_model()
    assert model.kwargs == {"model_kwargs": {"attn_implementation": "eager"}}
    assert store._get_model() is model  # loaded once


def test_mps_probe_is_false_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    assert pg._mps_available() is False


REAL_MODEL = os.environ.get("BREATHE_TEST_MPS_MODEL")


@pytest.mark.skipif(not REAL_MODEL, reason="set BREATHE_TEST_MPS_MODEL to run against a real model")
def test_short_texts_match_the_cpu_reference_on_mps():
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS device")
    from sentence_transformers import SentenceTransformer

    reference = SentenceTransformer(REAL_MODEL, device="cpu")
    model = pg.load_embedding_model(REAL_MODEL)
    assert str(model.device).startswith("mps")
    for text in ["lisbon", "audit log", "наши коты", "a longer sentence about moving a database to new servers"]:
        a = reference.encode(text, normalize_embeddings=True)
        b = model.encode(text, normalize_embeddings=True)  # one text, as a query is encoded
        assert float(np.dot(a, b)) > 0.999, text
