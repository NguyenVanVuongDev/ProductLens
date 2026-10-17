import io
import os
import sys
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.encoder import EnsembleCLIPEncoder
from app.index_service import FAISSIndexService
from app.exceptions import InvalidQueryError, InvalidImageError, IndexTrapError, VectorIDMappingError
from app.main import create_app, INDEX_PATH, CATALOG_PATH

EXPECTED_DIM = 1024


@pytest.fixture(scope="module")
def shared_encoder():
    return EnsembleCLIPEncoder()


@pytest.fixture(scope="module")
def shared_index_service():
    return FAISSIndexService(INDEX_PATH, CATALOG_PATH)


@pytest.fixture(scope="module")
def client(shared_encoder, shared_index_service):
    app = create_app(encoder=shared_encoder, index_service=shared_index_service)
    return TestClient(app)


def make_image_bytes(fmt="PNG", size=(224, 224), color=(120, 160, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def test_encoder_text_shape_and_norm(shared_encoder):
    emb = shared_encoder.encode_text("a black leather jacket")
    assert emb.shape == (1, EXPECTED_DIM)
    assert emb.dtype == np.float32
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)


def test_encoder_text_rejects_empty(shared_encoder):
    with pytest.raises(InvalidQueryError):
        shared_encoder.encode_text("   ")


def test_encoder_image_shape_and_norm(shared_encoder):
    raw = make_image_bytes("JPEG")
    emb = shared_encoder.encode_image_bytes(raw, content_type="image/jpeg")
    assert emb.shape == (1, EXPECTED_DIM)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)


def test_encoder_image_rejects_corrupted(shared_encoder):
    with pytest.raises(InvalidImageError):
        shared_encoder.encode_image_bytes(b"not_an_image", content_type="image/png")


def test_encoder_image_rejects_too_small(shared_encoder):
    tiny = make_image_bytes("PNG", size=(8, 8))
    with pytest.raises(InvalidImageError):
        shared_encoder.encode_image_bytes(tiny, content_type="image/png")


def test_mapping_consistency(shared_index_service):
    m0 = shared_index_service.mapper.get_by_faiss_id(0)
    assert m0["faiss_id"] == 0
    assert shared_index_service.mapper.get_by_product_id(m0["product_id"]) == m0


def test_mapping_out_of_bounds(shared_index_service):
    with pytest.raises(VectorIDMappingError):
        shared_index_service.mapper.get_by_faiss_id(-1)
    with pytest.raises(VectorIDMappingError):
        shared_index_service.mapper.get_by_faiss_id(999999)


def test_index_dimension_mismatch(shared_index_service):
    wrong = np.ones((1, 512), dtype=np.float32)
    with pytest.raises(IndexTrapError):
        shared_index_service.search(wrong, top_k=5)


def test_index_nan_vector(shared_index_service):
    nan = np.full((1, EXPECTED_DIM), np.nan, dtype=np.float32)
    with pytest.raises(IndexTrapError):
        shared_index_service.search(nan, top_k=5)


def test_rerank_boosts_matching_category(shared_index_service):
    vec = np.ascontiguousarray(
        shared_index_service.index.reconstruct(0).reshape(1, -1), dtype=np.float32
    )
    p0 = shared_index_service.mapper.get_by_faiss_id(0)

    hits_no, rr_no = shared_index_service.search(vec, top_k=10, use_rerank=False)
    hits_yes, rr_yes = shared_index_service.search(
        vec, top_k=10,
        query_meta={
            "category": p0["category"],
            "main_attribute": p0["main_attribute"],
            "other_attributes": p0["other_attributes"],
        },
        use_rerank=True,
    )
    assert rr_no is False
    assert rr_yes is True
    assert hits_no[0]["product"]["faiss_id"] == 0
    assert hits_yes[0]["product"]["faiss_id"] == 0


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "healthy"
    assert d["index_loaded"] is True
    assert d["embedding_dim"] == EXPECTED_DIM
    assert d["total_indexed_vectors"] == d["total_catalog_products"]
    assert len(d["models"]) == 2


def test_search_text_no_meta(client):
    r = client.post("/search/text", json={"query": "women floral summer dress", "top_k": 5})
    assert r.status_code == 200
    b = r.json()
    assert b["query_type"] == "text"
    assert b["total_results"] == 5
    assert b["rerank_applied"] is False


def test_search_text_with_meta(client):
    r = client.post("/search/text", json={
        "query": "women floral summer dress",
        "top_k": 5,
        "category": "dress",
        "main_attribute": "floral",
        "other_attributes": "summer, print",
    })
    assert r.status_code == 200
    assert r.json()["rerank_applied"] is True


def test_search_text_empty_rejected(client):
    r = client.post("/search/text", json={"query": "   ", "top_k": 5})
    assert r.status_code == 422


def test_search_text_invalid_top_k(client):
    assert client.post("/search/text", json={"query": "abc", "top_k": 0}).status_code == 422
    assert client.post("/search/text", json={"query": "abc", "top_k": 999}).status_code == 422


def test_search_image_success(client):
    img = make_image_bytes("PNG")
    r = client.post("/search/image?top_k=4", files={"file": ("sample.png", img, "image/png")})
    assert r.status_code == 200
    b = r.json()
    assert b["query_type"] == "image"
    assert len(b["results"]) == 4


def test_search_image_with_meta(client):
    img = make_image_bytes("JPEG")
    r = client.post(
        "/search/image?top_k=5&category=jacket&main_attribute=leather",
        files={"file": ("sample.jpg", img, "image/jpeg")}
    )
    assert r.status_code == 200
    assert r.json()["rerank_applied"] is True


def test_search_image_invalid_upload(client):
    r = client.post("/search/image?top_k=5", files={"file": ("mal.txt", b"hello", "text/plain")})
    assert r.status_code == 400
    assert r.json()["error_type"] == "InvalidImageError"


def test_get_product_ok_and_404(client, shared_index_service):
    pid = shared_index_service.mapper.get_by_faiss_id(0)["product_id"]
    assert client.get(f"/products/{pid}").status_code == 200
    assert client.get("/products/nope_999").status_code == 404


def test_endpoint_index_unavailable(shared_encoder, shared_index_service):
    app = create_app(encoder=shared_encoder, index_service=shared_index_service)
    app.state.index_service = None
    broken = TestClient(app)
    r = broken.post("/search/text", json={"query": "red shirt", "top_k": 5})
    assert r.status_code == 503
    assert r.json()["error_type"] == "IndexTrapError"
