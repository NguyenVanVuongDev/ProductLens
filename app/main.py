import os
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Request
from fastapi.responses import JSONResponse
from app.schemas import TextSearchRequest, SearchResponse, ProductMetadata, HealthResponse
from app.encoder import EnsembleCLIPEncoder
from app.index_service import FAISSIndexService
from app.exceptions import InvalidQueryError, InvalidImageError, IndexTrapError, VectorIDMappingError

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
INDEX_PATH   = os.getenv("FAISS_INDEX_PATH",   os.path.join(DEFAULT_DATA_DIR, "lookbench_gallery_hnsw_m16.faiss"))
CATALOG_PATH = os.getenv("PRODUCT_CATALOG_PATH", os.path.join(DEFAULT_DATA_DIR, "product_catalog.parquet"))


def create_app(encoder: EnsembleCLIPEncoder = None, index_service: FAISSIndexService = None) -> FastAPI:
    app = FastAPI(
        title="LookBench Fashion Retrieval API (MURE v5)",
        version="5.0.0",
        description="Multimodal visual search với FashionCLIP ⊕ ViT-B/32 + MURE v5 Dual-EKMA Reranker (>90% Benchmark)"
    )

    app.state.encoder = encoder or EnsembleCLIPEncoder()
    app.state.index_service = index_service or FAISSIndexService(INDEX_PATH, CATALOG_PATH)

    @app.exception_handler(InvalidQueryError)
    async def _h1(_: Request, exc: InvalidQueryError):
        return JSONResponse(status_code=400, content={"detail": str(exc), "error_type": "InvalidQueryError"})

    @app.exception_handler(InvalidImageError)
    async def _h2(_: Request, exc: InvalidImageError):
        return JSONResponse(status_code=400, content={"detail": str(exc), "error_type": "InvalidImageError"})

    @app.exception_handler(IndexTrapError)
    async def _h3(_: Request, exc: IndexTrapError):
        return JSONResponse(status_code=503, content={"detail": str(exc), "error_type": "IndexTrapError"})

    @app.exception_handler(VectorIDMappingError)
    async def _h4(_: Request, exc: VectorIDMappingError):
        return JSONResponse(status_code=500, content={"detail": str(exc), "error_type": "VectorIDMappingError"})

    @app.get("/health", response_model=HealthResponse)
    async def health():
        idx: FAISSIndexService = app.state.index_service
        enc: EnsembleCLIPEncoder = app.state.encoder
        if idx is None or not idx.is_ready:
            raise IndexTrapError("FAISS index không khả dụng.")
        return HealthResponse(
            status="healthy",
            models=[m["tag"] for m in enc.models],
            index_loaded=idx.is_ready,
            total_indexed_vectors=int(idx.index.ntotal),
            total_catalog_products=len(idx.mapper),
            embedding_dim=enc.embedding_dim,
        )

    @app.post("/search/text", response_model=SearchResponse)
    async def search_text(payload: TextSearchRequest):
        enc: EnsembleCLIPEncoder = app.state.encoder
        idx: FAISSIndexService = app.state.index_service
        if idx is None or not idx.is_ready:
            raise IndexTrapError("FAISS index chưa sẵn sàng.")

        q_vec = enc.encode_text(payload.query)
        query_meta = {
            "category": payload.category or "",
            "main_attribute": payload.main_attribute or "",
            "other_attributes": payload.other_attributes or "",
        }
        hits, rr = idx.search(q_vec, top_k=payload.top_k, query_meta=query_meta, use_rerank=True)
        return SearchResponse(
            query_type="text", top_k=payload.top_k,
            total_results=len(hits), rerank_applied=rr, results=hits,
        )

    @app.post("/search/image", response_model=SearchResponse)
    async def search_image(
        file: UploadFile = File(..., description="JPEG/PNG/WEBP"),
        top_k: int = Query(default=10, ge=1, le=50),
        category: str = Query(default=""),
        main_attribute: str = Query(default=""),
        other_attributes: str = Query(default=""),
    ):
        enc: EnsembleCLIPEncoder = app.state.encoder
        idx: FAISSIndexService = app.state.index_service
        if idx is None or not idx.is_ready:
            raise IndexTrapError("FAISS index chưa sẵn sàng.")

        raw = await file.read()
        q_vec = enc.encode_image_bytes(raw, content_type=file.content_type)
        query_meta = {
            "category": category, "main_attribute": main_attribute,
            "other_attributes": other_attributes,
        }
        hits, rr = idx.search(q_vec, top_k=top_k, query_meta=query_meta, use_rerank=True)
        return SearchResponse(
            query_type="image", top_k=top_k,
            total_results=len(hits), rerank_applied=rr, results=hits,
        )

    @app.get("/products/{product_id}", response_model=ProductMetadata)
    async def get_product(product_id: str):
        idx: FAISSIndexService = app.state.index_service
        if idx is None or not idx.is_ready:
            raise IndexTrapError("Product catalog chưa sẵn sàng.")
        try:
            return ProductMetadata(**idx.mapper.get_by_product_id(product_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


app = create_app()
