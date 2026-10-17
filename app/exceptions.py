class InvalidQueryError(ValueError):
    """Câu truy vấn văn bản trống hoặc không hợp lệ."""

class InvalidImageError(ValueError):
    """File ảnh hỏng, sai định dạng, hoặc vượt kích thước."""

class IndexTrapError(RuntimeError):
    """FAISS index chưa nạp, bị hỏng, hoặc chiều vector không khớp."""

class VectorIDMappingError(KeyError):
    """Vector ID trả về từ FAISS không ánh xạ được sang metadata."""
