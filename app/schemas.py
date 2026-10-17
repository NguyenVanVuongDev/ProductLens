from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class QueryMeta(BaseModel):
    """Metadata tùy chọn của query để kích hoạt MURE v5 advanced rerank."""
    category: Optional[str] = ""
    main_attribute: Optional[str] = ""
    other_attributes: Optional[str] = ""


class TextSearchRequest(BaseModel):
    query: str = Field(..., description="Mô tả văn bản cần tìm")
    top_k: int = Field(default=10, ge=1, le=50)
    category: Optional[str] = ""
    main_attribute: Optional[str] = ""
    other_attributes: Optional[str] = ""

    @field_validator("query")
    @classmethod
    def _validate_query(cls, v: str) -> str:
        cleaned = (v or "").strip()
        if not cleaned:
            raise ValueError("Truy vấn không được để trống.")
        if len(cleaned) > 500:
            raise ValueError("Truy vấn vượt quá 500 ký tự.")
        return cleaned


class ProductMetadata(BaseModel):
    product_id: str
    faiss_id: int
    item_ID: int
    subset: str
    category: str
    main_attribute: str
    other_attributes: Optional[str] = ""


class SearchResultItem(BaseModel):
    rank: int
    score: float
    product: ProductMetadata


class SearchResponse(BaseModel):
    query_type: str
    top_k: int
    total_results: int
    rerank_applied: bool
    results: List[SearchResultItem]


class HealthResponse(BaseModel):
    status: str
    models: List[str]
    index_loaded: bool
    total_indexed_vectors: int
    total_catalog_products: int
    embedding_dim: int
