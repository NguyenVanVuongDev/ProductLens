import io
import logging
import os
import pickle
import warnings
import numpy as np
import torch
import open_clip
from PIL import Image, UnidentifiedImageError
from app.exceptions import InvalidQueryError, InvalidImageError


ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
ALLOWED_FORMAT = {"JPEG", "PNG", "WEBP", "MPO"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRANSFORM_DIR = os.getenv(
    "TRANSFORM_DIR", os.path.join(PROJECT_ROOT, "artifacts", "embeddings")
)
logger = logging.getLogger(__name__)


class EnsembleCLIPEncoder:
    """
    Ensemble 2 model (FashionCLIP 0.7 ⊕ ViT-B/32 0.3) tạo vector chuẩn L2 1024 chiều.
    """

    MODEL_SPECS = [
        ("hf-hub:Marqo/marqo-fashionCLIP", None, 0.7, "FashionCLIP"),
        ("ViT-B-32", "laion2b_s34b_b79k", 0.3, "ViT-B-32-laion"),
    ]

    def __init__(self, device: str = None, transform_dir: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.transform_dir = transform_dir or DEFAULT_TRANSFORM_DIR
        self.models = []

        for name, pretrained, weight, tag in self.MODEL_SPECS:
            try:
                if pretrained is None:
                    model, _, preprocess = open_clip.create_model_and_transforms(name, device=self.device)
                else:
                    model, _, preprocess = open_clip.create_model_and_transforms(
                        name, pretrained=pretrained, device=self.device
                    )
                tokenizer = open_clip.get_tokenizer(name)
                model.eval()
                self.models.append({
                    "model": model,
                    "preprocess": preprocess,
                    "tokenizer": tokenizer,
                    "weight": weight,
                    "tag": tag,
                })
                print(f"   ✅ Loaded {tag} (w={weight})")
            except Exception as exc:
                raise RuntimeError(f"Không load được model bắt buộc {tag}: {exc}") from exc

        if len(self.models) != len(self.MODEL_SPECS):
            raise RuntimeError("Không load đủ các model ensemble bắt buộc.")
        self.embedding_dim = 512 * len(self.MODEL_SPECS)
        self._load_transform_weights()

    @staticmethod
    def _l2(v, eps=1e-12):
        v = np.asarray(v, dtype=np.float32)
        if v.ndim != 2 or not np.all(np.isfinite(v)):
            raise RuntimeError("Embedding phải là ma trận hữu hạn 2D.")
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        if np.any(norms <= eps):
            raise RuntimeError("Model trả về embedding zero-vector.")
        return (v / norms).astype(np.float32)

    def _load_transform_weights(self):
        bundle_path = os.path.join(self.transform_dir, "transform_pipeline.pkl")
        means_path = os.path.join(self.transform_dir, "domain_centering_means.npz")
        w_qi_path = os.path.join(self.transform_dir, "dual_ekma_W_QI.npy")
        w_qt_path = os.path.join(self.transform_dir, "dual_ekma_W_QT.npy")

        if os.path.isfile(bundle_path):
            with open(bundle_path, "rb") as bundle_file:
                bundle = pickle.load(bundle_file)
            if int(bundle.get("embedding_dim", -1)) != self.embedding_dim:
                raise RuntimeError(
                    f"Transform dimension {bundle.get('embedding_dim')} không khớp encoder {self.embedding_dim}."
                )
            self.gamma_center = float(bundle["gamma_center"])
            self.mean_img = np.asarray(bundle["global_q_vis_mean"], dtype=np.float32).reshape(1, -1)
            self.mean_txt = np.asarray(bundle["global_q_txt_mean"], dtype=np.float32).reshape(1, -1)
            self.W_QI = np.asarray(bundle["W_QI"], dtype=np.float32)
            self.W_QT = np.asarray(bundle["W_QT"], dtype=np.float32)
        elif all(os.path.isfile(path) for path in (means_path, w_qi_path, w_qt_path)):
            means = np.load(means_path)
            self.gamma_center = float(means["gamma_center"][0]) if "gamma_center" in means else 0.45
            self.mean_img = np.asarray(means["global_q_vis_mean"], dtype=np.float32).reshape(1, -1)
            self.mean_txt = np.asarray(means["global_q_txt_mean"], dtype=np.float32).reshape(1, -1)
            self.W_QI = np.load(w_qi_path).astype(np.float32)
            self.W_QT = np.load(w_qt_path).astype(np.float32)
        else:
            raise RuntimeError(f"Thiếu transform artifacts trong {self.transform_dir}.")

        expected_matrix = (self.embedding_dim, self.embedding_dim)
        if (
            self.mean_img.shape != (1, self.embedding_dim)
            or self.mean_txt.shape != (1, self.embedding_dim)
            or self.W_QI.shape != expected_matrix
            or self.W_QT.shape != expected_matrix
        ):
            raise RuntimeError("Shape của transform artifacts không khớp embedding dimension.")
        if not all(np.all(np.isfinite(value)) for value in (self.mean_img, self.mean_txt, self.W_QI, self.W_QT)):
            raise RuntimeError("Transform artifacts chứa NaN/Inf.")
        if not np.isfinite(self.gamma_center):
            raise RuntimeError("gamma_center không hữu hạn.")
        self.transform_loaded = True

    def apply_query_transform(self, raw_vec: np.ndarray, flow: str = "image") -> np.ndarray:
        if flow not in {"image", "text"}:
            raise ValueError("flow phải là 'image' hoặc 'text'.")
        vec = self._l2(raw_vec)
        if vec.shape[1] != self.embedding_dim:
            raise RuntimeError(f"Embedding dimension {vec.shape[1]} != {self.embedding_dim}.")
        mean_vec = self.mean_img if flow == "image" else self.mean_txt
        transform = self.W_QI if flow == "image" else self.W_QT
        vec = self._l2(vec - self.gamma_center * mean_vec)
        return self._l2(vec @ transform)

    def _combine(self, parts):
        if len(parts) != len(self.models):
            raise RuntimeError("Số embedding không khớp số model ensemble.")
        weighted = [self._l2(p) * m["weight"] for p, m in zip(parts, self.models)]
        combined = self._l2(np.hstack(weighted))
        if combined.shape != (1, self.embedding_dim):
            raise RuntimeError(f"Embedding shape không hợp lệ: {combined.shape}.")
        return combined

    def encode_text(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            raise InvalidQueryError("Câu truy vấn văn bản trống.")
        cleaned = text.strip()
        parts = []
        for m in self.models:
            with torch.inference_mode():
                tok = m["tokenizer"]([cleaned]).to(self.device)
                emb = m["model"].encode_text(tok).float().cpu().numpy()
            parts.append(emb)
        return self.apply_query_transform(self._combine(parts), flow="text")

    def validate_and_load_image(self, raw_bytes: bytes, content_type: str = None) -> Image.Image:
        if not raw_bytes:
            raise InvalidImageError("File ảnh rỗng (0 bytes).")
        if len(raw_bytes) > MAX_IMAGE_BYTES:
            raise InvalidImageError(f"Ảnh vượt giới hạn {MAX_IMAGE_BYTES // (1024 * 1024)}MB.")
        if content_type and content_type.lower() not in ALLOWED_MIME:
            raise InvalidImageError(f"Content-Type không hỗ trợ: {content_type}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                img = Image.open(io.BytesIO(raw_bytes))
                w, h = img.size
                if w < 16 or h < 16:
                    raise InvalidImageError(f"Ảnh quá nhỏ ({w}x{h}). Cần ≥16x16.")
                if w * h > MAX_IMAGE_PIXELS:
                    raise InvalidImageError("Ảnh vượt giới hạn 40 megapixels.")
                img.verify()
                img = Image.open(io.BytesIO(raw_bytes))
            if img.format and img.format.upper() not in ALLOWED_FORMAT:
                raise InvalidImageError(f"Format {img.format} không hỗ trợ.")
            return img.convert("RGB")
        except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise InvalidImageError(f"Dữ liệu ảnh hỏng: {exc}") from exc

    def encode_image_bytes(self, raw_bytes: bytes, content_type: str = None) -> np.ndarray:
        img = self.validate_and_load_image(raw_bytes, content_type=content_type)
        parts = []
        for m in self.models:
            tensor = m["preprocess"](img).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                emb = m["model"].encode_image(tensor).float().cpu().numpy()
            parts.append(emb)
        return self.apply_query_transform(self._combine(parts), flow="image")
