from unittest.mock import AsyncMock

import pytest

from dynavec.async_client import AsyncDynavec
from dynavec.models import SearchResult, UpsertResult

try:
    from fastapi.testclient import TestClient

    from dynavec.server import create_app
    HAS_SERVER = True
except ImportError:
    HAS_SERVER = False

pytestmark = pytest.mark.skipif(not HAS_SERVER, reason="FastAPI not installed")


@pytest.fixture
def mock_client():
    client = AsyncMock(spec=AsyncDynavec)
    
    # Mock search
    client.search.return_value = [
        SearchResult(id="doc1", score=0.9, text="hello world", metadata={"a": 1})
    ]
    
    # Mock upsert
    client.upsert.return_value = UpsertResult(count=1, ids=["doc1"])
    
    # Add delete mock explicitly since AsyncMock(spec=...) won't include dynamic methods
    client.delete = AsyncMock()
    
    return client


@pytest.fixture
def client(mock_client):
    app = create_app(mock_client)
    return TestClient(app)


def test_search(client, mock_client):
    resp = client.post("/search", json={"query": "test", "top_k": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["id"] == "doc1"
    
    mock_client.search.assert_awaited_once_with(
        query="test",
        top_k=2,
        namespace="default",
        filter=None,
    )


def test_upsert(client, mock_client):
    resp = client.post("/upsert", json={
        "documents": [{"id": "doc1", "text": "hello"}],
        "namespace": "custom"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["ids"] == ["doc1"]
    
    mock_client.upsert.assert_awaited_once()


def test_delete(client, mock_client):
    resp = client.post("/delete", json={"ids": ["doc1"]})
    assert resp.status_code == 200
    mock_client.delete.assert_awaited_once_with(["doc1"], namespace="default")
