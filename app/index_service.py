import os
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
        self._f2m: Dict[int, dict] = {}
        self._p2m: Dict[str, dict] = {}
        for r in catalog_df.to_dict(orient="records"):
            fid = int(r["faiss_id"])
            pid = str(r["product_id"])
            item = {
                "product_id": pid,
                "faiss_id": fid,
                "item_ID": int(r.get("item_ID", -1)),
                "subset": str(r.get("subset", "")),
                "category": str(r.get("category", "")),
                "main_attribute": str(r.get("main_attribute", "")),
                "other_attributes": str(r.get("other_attributes", "")),
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


class FAISSIndexService:
    """Dịch vụ tìm kiếm FAISS tích hợp MURE v5 Reranker."""

    def __init__(self, index_path: str, catalog_path: str, rerank_cfg: Optional[dict] = None):
        if not os.path.exists(index_path):
            raise IndexTrapError(f"Không tìm thấy index: {index_path}")
        if not os.path.exists(catalog_path):
            raise IndexTrapError(f"Không tìm thấy catalog: {catalog_path}")

        try:
            self.index = faiss.read_index(index_path)
        except Exception as exc:
            raise IndexTrapError(f"Không load được FAISS index: {exc}") from exc

        catalog_df = pd.read_parquet(catalog_path)
        self.mapper = VectorIDMapper(catalog_df)

        if self.index.ntotal == 0:
            raise IndexTrapError("FAISS index rỗng.")

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
        if query_vec is None or query_vec.ndim != 2 or query_vec.shape[0] != 1:
            raise IndexTrapError("Query vector phải có shape (1, dim).")
        if query_vec.shape[1] != self.index.d:
            raise IndexTrapError(f"Lệch dim: query {query_vec.shape[1]} vs index {self.index.d}.")
        if not np.all(np.isfinite(query_vec)):
            raise IndexTrapError("Query vector chứa NaN/Inf.")

        n_ret = self.rerank_cfg["n_retrieve"] if use_rerank else top_k
        n_ret = min(n_ret, self.index.ntotal)

        try:
            scores, indices = self.index.search(
                np.ascontiguousarray(query_vec, dtype=np.float32), n_ret
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
