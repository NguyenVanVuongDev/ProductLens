import io
import numpy as np
import torch
import open_clip
from PIL import Image, UnidentifiedImageError
from app.exceptions import InvalidQueryError, InvalidImageError


ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
ALLOWED_FORMAT = {"JPEG", "PNG", "WEBP", "MPO"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024


class EnsembleCLIPEncoder:
    """
    Ensemble 2 model (FashionCLIP 0.7 ⊕ ViT-B/32 0.3) tạo vector chuẩn L2 1024 chiều.
    """

    MODEL_SPECS = [
        ("hf-hub:Marqo/marqo-fashionCLIP", None, 0.7, "FashionCLIP"),
        ("ViT-B-32", "laion2b_s34b_b79k", 0.3, "ViT-B-32-laion"),
    ]

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
            except Exception as e:
                print(f"   ⚠️ Bỏ qua {tag}: {e}")

        if not self.models:
            raise RuntimeError("Không load được model nào!")
        self.embedding_dim = 512 * len(self.models)

    @staticmethod
    def _l2(v, eps=1e-12):
        return (v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), eps)).astype(np.float32)

    def _combine(self, parts):
        weighted = [self._l2(p) * m["weight"] for p, m in zip(parts, self.models)]
        return self._l2(np.hstack(weighted))

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
        return self._combine(parts)

    def validate_and_load_image(self, raw_bytes: bytes, content_type: str = None) -> Image.Image:
        if not raw_bytes:
            raise InvalidImageError("File ảnh rỗng (0 bytes).")
        if len(raw_bytes) > MAX_IMAGE_BYTES:
            raise InvalidImageError(f"Ảnh vượt giới hạn {MAX_IMAGE_BYTES // (1024 * 1024)}MB.")
        if content_type and content_type.lower() not in ALLOWED_MIME:
            raise InvalidImageError(f"Content-Type không hỗ trợ: {content_type}")
        try:
            img = Image.open(io.BytesIO(raw_bytes))
            img.verify()
            img = Image.open(io.BytesIO(raw_bytes))
            if img.format and img.format.upper() not in ALLOWED_FORMAT:
                raise InvalidImageError(f"Format {img.format} không hỗ trợ.")
            w, h = img.size
            if w < 16 or h < 16:
                raise InvalidImageError(f"Ảnh quá nhỏ ({w}x{h}). Cần ≥16x16.")
            return img.convert("RGB")
        except (UnidentifiedImageError, OSError, SyntaxError) as exc:
            raise InvalidImageError(f"Dữ liệu ảnh hỏng: {exc}") from exc

    def encode_image_bytes(self, raw_bytes: bytes, content_type: str = None) -> np.ndarray:
        img = self.validate_and_load_image(raw_bytes, content_type=content_type)
        parts = []
        for m in self.models:
            tensor = m["preprocess"](img).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                emb = m["model"].encode_image(tensor).float().cpu().numpy()
            parts.append(emb)
        return self._combine(parts)
