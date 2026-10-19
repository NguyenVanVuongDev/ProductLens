import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import faiss
from app.exceptions import IndexTrapError, VectorIDMappingError


class VectorIDMapper:
    """Ánh xạ hai chiều FAISS ID ↔ Product Metadata."""

    def __init__(self, catalog_df: pd.DataFrame):
        if "faiss_id" not in catalog_df.columns or "product_id" not in catalog_df.columns:
            raise ValueError("Catalog phải có cột 'faiss_id' và 'product_id'.")
        if catalog_df["faiss_id"].isna().any() or catalog_df["product_id"].isna().any():
            raise ValueError("Catalog không được chứa FAISS ID hoặc product ID rỗng.")
        faiss_ids = pd.to_numeric(catalog_df["faiss_id"], errors="coerce")
        if faiss_ids.isna().any() or faiss_ids.duplicated().any():
            raise ValueError("Catalog chứa FAISS ID không hợp lệ hoặc trùng lặp.")
        if not np.array_equal(faiss_ids.to_numpy(), faiss_ids.astype(np.int64).to_numpy()):
            raise ValueError("FAISS ID trong catalog phải là số nguyên.")
        if set(faiss_ids.astype(np.int64)) != set(range(len(catalog_df))):
            raise ValueError("FAISS ID trong catalog phải liên tục từ 0 đến N-1.")
        if catalog_df["product_id"].astype(str).duplicated().any():
            raise ValueError("Catalog chứa product_id trùng lặp.")
        self._f2m: Dict[int, dict] = {}
        self._p2m: Dict[str, dict] = {}
        self._gallery_positions: Dict[str, int] = {}
        gallery_occurrences: Dict[tuple, int] = {}
        for r in catalog_df.to_dict(orient="records"):
            fid = int(r["faiss_id"])
            pid = str(r["product_id"])
            subset = str(r.get("subset", ""))
            item_id = int(r.get("item_ID", -1))
            gallery_key = (subset, item_id)
            self._gallery_positions[pid] = gallery_occurrences.get(gallery_key, 0)
            gallery_occurrences[gallery_key] = self._gallery_positions[pid] + 1
            item = {
                "product_id": pid,
                "faiss_id": fid,
                "item_ID": item_id,
                "subset": subset,
                "category": str(r.get("category", "")),
                "main_attribute": str(r.get("main_attribute", "")),
                "other_attributes": str(r.get("other_attributes", "")),
                "title": "" if pd.isna(r.get("title", "")) else str(r.get("title", "")),
                "image_path": "" if pd.isna(r.get("image_path", "")) else str(r.get("image_path", "")),
            }
            self._f2m[fid] = item
            self._p2m[pid] = item

    def __len__(self):
        return len(self._f2m)

    def get_by_faiss_id(self, faiss_id: int) -> dict:
        fid = int(faiss_id)
        if fid < 0 or fid not in self._f2m:
            raise VectorIDMappingError(f"FAISS ID {fid} không tồn tại.")
        return self._f2m[fid]

    def get_by_product_id(self, product_id: str) -> dict:
        if product_id not in self._p2m:
            raise KeyError(f"Không tìm thấy product_id='{product_id}'.")
        return self._p2m[product_id]

    def get_gallery_position(self, product_id: str) -> int:
        if product_id not in self._gallery_positions:
            raise KeyError(f"Không tìm thấy product_id='{product_id}'.")
        return self._gallery_positions[product_id]


class FAISSIndexService:
    """Dịch vụ tìm kiếm FAISS tích hợp MURE v5 Reranker."""

    def __init__(self, index_path: str, catalog_path: str, rerank_cfg: Optional[dict] = None,
                 gallery_root: Optional[str] = None, image_root: Optional[str] = None):
        if not os.path.exists(index_path):
            raise IndexTrapError(f"Không tìm thấy index: {index_path}")
        if not os.path.exists(catalog_path):
            raise IndexTrapError(f"Không tìm thấy catalog: {catalog_path}")

        try:
            self.index = faiss.read_index(index_path)
        except Exception as exc:
            raise IndexTrapError(f"Không load được FAISS index: {exc}") from exc

        catalog_df = pd.read_parquet(catalog_path)
        catalog_df["faiss_id"] = pd.to_numeric(catalog_df["faiss_id"], errors="raise").astype(np.int64)
        catalog_df = catalog_df.sort_values("faiss_id").reset_index(drop=True)
        self.mapper = VectorIDMapper(catalog_df)
        self.gallery_root = Path(
            gallery_root or os.path.join(os.path.dirname(catalog_path), "v20251201")
        ).resolve()
        self.image_root = Path(image_root or os.path.dirname(catalog_path)).resolve()

        if self.index.ntotal == 0:
            raise IndexTrapError("FAISS index rỗng.")
        if self.index.ntotal != len(self.mapper):
            raise IndexTrapError(
                f"Lệch đồng bộ Index-Metadata: index.ntotal={self.index.ntotal} != len(catalog)={len(self.mapper)}"
            )
        if self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise IndexTrapError("Index phải dùng inner product trên vector L2-normalized (cosine similarity).")
        if self.index.d != 1024:
            raise IndexTrapError(f"Dimension index không hợp lệ: cần 1024, nhận {self.index.d}.")

        self._cats = catalog_df["category"].fillna("").astype(str).str.lower().values
        self._mains = catalog_df["main_attribute"].fillna("").astype(str).str.lower().values
        self._others = catalog_df["other_attributes"].fillna("").astype(str).str.lower().values
        self._other_sets = [
            set(t.strip() for t in str(x).split(",") if t.strip() and t.strip() not in ("none", "nan"))
            for x in self._others
        ]
        self._sigs = np.array([
            f"{c}|{m}|{'|'.join(sorted(oset))}"
            for c, m, oset in zip(self._cats, self._mains, self._other_sets)
        ])

        self.rerank_cfg = rerank_cfg or {
            "n_retrieve": 1000,
            "w_cat": 0.08,
            "w_main": 0.10,
            "w_other": 0.08,
            "w_other_per_token": 0.02,
            "w_sig": 0.10,
        }

    @property
    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    @lru_cache(maxsize=128)
    def get_product_image(self, product_id: str) -> bytes:
        metadata = self.mapper.get_by_product_id(product_id)
        image_path = metadata.get("image_path", "")
        if image_path:
            candidate = Path(image_path)
            if not candidate.is_absolute():
                candidate = self.image_root / candidate
            candidate = candidate.resolve()
            if self.image_root not in candidate.parents or not candidate.is_file():
                raise FileNotFoundError(f"Không tìm thấy ảnh cục bộ cho product_id='{product_id}'.")
            return candidate.read_bytes()

        gallery_path = (self.gallery_root / metadata["subset"] / "gallery.parquet").resolve()
        if self.gallery_root not in gallery_path.parents or not gallery_path.is_file():
            raise FileNotFoundError(f"Gallery không tồn tại cho product_id='{product_id}'.")

        rows = pd.read_parquet(
            gallery_path,
            columns=["item_ID", "image"],
            filters=[("item_ID", "==", str(metadata["item_ID"]))],
        )
        position = self.mapper.get_gallery_position(product_id)
        if position >= len(rows):
            raise KeyError(f"Không tìm thấy ảnh cho product_id='{product_id}'.")

        image = rows.iloc[position]["image"]
        image_bytes = image.get("bytes") if isinstance(image, dict) else None
        if image_bytes:
            return bytes(image_bytes)

        image_path = image.get("path") if isinstance(image, dict) else None
        if image_path:
            path = Path(image_path)
            if not path.is_absolute():
                path = gallery_path.parent / path
            if path.is_file():
                return path.read_bytes()
        raise KeyError(f"Ảnh không có dữ liệu cho product_id='{product_id}'.")

    def _rerank(self, cand_idx: np.ndarray, cand_scores: np.ndarray, query_meta: dict) -> tuple:
        if query_meta is None:
            return cand_idx, cand_scores

        add = np.zeros(len(cand_idx), dtype=np.float32)
        cfg = self.rerank_cfg

        cq = (query_meta.get("category") or "").strip().lower()
        mq = (query_meta.get("main_attribute") or "").strip().lower()
        oq = (query_meta.get("other_attributes") or "").strip().lower()
        q_tokens = set(t.strip() for t in oq.split(",") if t.strip() and t.strip() not in ("none", "nan"))
        q_sig = f"{cq}|{mq}|{'|'.join(sorted(q_tokens))}"

        if cq and cq not in ("none", "nan"):
            add += cfg["w_cat"] * (self._cats[cand_idx] == cq).astype(np.float32)

        if mq and mq not in ("none", "nan"):
            add += cfg["w_main"] * (self._mains[cand_idx] == mq).astype(np.float32)

        if cq and mq and q_tokens:
            add += cfg.get("w_sig", 0.10) * (self._sigs[cand_idx] == q_sig).astype(np.float32)

        if q_tokens:
            q_len = len(q_tokens)
            for j, gi in enumerate(cand_idx):
                if gi < 0:
                    continue
                g_tokens = self._other_sets[gi]
                if g_tokens:
                    ov = len(q_tokens & g_tokens)
                    if ov > 0:
                        dice = (2.0 * ov) / (q_len + len(g_tokens))
                        add[j] += cfg["w_other"] * dice + cfg["w_other_per_token"] * ov

        adjusted = cand_scores + add
        order = np.argsort(-adjusted)
        return cand_idx[order], adjusted[order]

    def search(self, query_vec: np.ndarray, top_k: int = 10,
               query_meta: Optional[dict] = None, use_rerank: bool = True):
        if not self.is_ready:
            raise IndexTrapError("FAISS index chưa sẵn sàng.")
        if not isinstance(top_k, int) or top_k < 1 or top_k > 50:
            raise ValueError(f"top_k={top_k} không hợp lệ (1..50).")
        if query_vec is None:
            raise IndexTrapError("Query vector phải có shape (1, dim).")
        query_vec = np.asarray(query_vec)
        if query_vec.ndim != 2 or query_vec.shape[0] != 1:
            raise IndexTrapError("Query vector phải có shape (1, dim).")
        if query_vec.shape[1] != self.index.d:
            raise IndexTrapError(f"Lệch dim: query {query_vec.shape[1]} vs index {self.index.d}.")
        if not np.all(np.isfinite(query_vec)):
            raise IndexTrapError("Query vector chứa NaN/Inf.")
        query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)
        if float(np.linalg.norm(query_vec)) <= 1e-12:
            raise IndexTrapError("Query vector có norm bằng 0.")
        faiss.normalize_L2(query_vec)

        n_ret = self.rerank_cfg["n_retrieve"] if use_rerank else top_k
        n_ret = min(n_ret, self.index.ntotal)

        try:
            scores, indices = self.index.search(
                query_vec, n_ret
            )
        except Exception as exc:
            raise IndexTrapError(f"Lỗi FAISS: {exc}") from exc

        cand_idx, cand_scores = indices[0], scores[0]
        rerank_applied = False

        if use_rerank and query_meta is not None:
            has_meta = any(
                (query_meta.get(k) or "").strip()
                for k in ("category", "main_attribute", "other_attributes")
            )
            if has_meta:
                cand_idx, cand_scores = self._rerank(cand_idx, cand_scores, query_meta)
                rerank_applied = True

        results = []
        rank = 1
        for fid, sc in zip(cand_idx[:top_k], cand_scores[:top_k]):
            if int(fid) == -1:
                continue
            product_meta = self.mapper.get_by_faiss_id(int(fid))
            results.append({
                "rank": rank,
                "score": round(float(sc), 6),
                "product": product_meta,
            })
            rank += 1
        return results, rerank_applied
