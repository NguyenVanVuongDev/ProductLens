# Multimodal Visual Retrieval

Backend tìm kiếm ảnh thời trang bằng text-to-image và image-to-image. API hiện dùng ensemble FashionCLIP (0.7) + ViT-B/32-laion (0.3), embedding 1024 chiều và FAISS HNSW; đây không phải baseline CLIP ViT-B/32 thuần. ABO là domain đồ gia dụng, còn LookBench là benchmark thời trang; không gộp hai domain hoặc suy diễn nhãn benchmark từ metadata sản phẩm.

## Cấu trúc runtime

- `app/main.py`: FastAPI `/search/text`, `/search/image`, `/products/{product_id}`, `/products/{product_id}/image` và `/health`.
- `app/demo.py`: Streamlit UI gọi API, hỗ trợ text/image, top-K, score và metadata.
- `artifacts/indexes/lookbench_gallery_hnsw_m16.faiss`: index runtime mặc định.
- `artifacts/embeddings/transform_pipeline.pkl`: transform bundle v6 dùng chung cho text/image query.
- `artifacts/embeddings/domain_centering_means.npz`, `dual_ekma_W_QI.npy`, `dual_ekma_W_QT.npy`: transform artifacts dạng NumPy để kiểm tra/tái dựng.
- `data/lookbench/product_catalog.parquet`: catalog runtime mặc định, ánh xạ FAISS ID sang metadata.
- `notebooks/da1-day-3-day-4.ipynb`: notebook pipeline và đánh giá hiện có.

Index, catalog và transform v6 phải thuộc cùng một lần build trước khi khởi chạy API. API từ chối khởi động nếu transform thiếu/sai dimension hoặc dimension index không khớp; không fallback âm thầm sang embedding raw. Docker chỉ đóng gói dịch vụ, không tải dữ liệu hoặc build index trong quá trình startup. Repo hiện chưa có CLI độc lập để tái tạo index từ dữ liệu thô; quy trình tạo artifact vẫn nằm trong notebook. Dữ liệu `data/` và artifact lớn không được đưa vào image.

## Chạy local

Yêu cầu Python 3.11+, đủ dung lượng model/cache, và artifacts ở đúng đường dẫn mặc định hoặc cấu hình bằng `FAISS_INDEX_PATH`, `PRODUCT_CATALOG_PATH`, `TRANSFORM_DIR`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Khởi chạy API và demo ở hai terminal:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```powershell
streamlit run app/demo.py
```

Mở `http://localhost:8501`; API docs ở `http://localhost:8000/docs`. Lần đầu chạy, OpenCLIP có thể tải model weights. CPU inference được hỗ trợ nhưng có thể chậm; PyTorch tự dùng CUDA khi khả dụng.

## Docker Compose

Trước tiên đặt index và catalog đã dựng sẵn ở runtime paths trên máy host. Build image và start API/UI bằng lệnh sau; bước build index vẫn là thao tác riêng ngoài Compose:

```powershell
docker compose up --build
```

Demo: `http://localhost:8501`; API: `http://localhost:8000/docs`. Compose mount `artifacts/` và `data/` read-only, chờ health check API trước khi start demo. Model weights được tải khi API khởi động và cần kết nối mạng lần đầu. Để chỉ build image mà không start services: `docker compose build`.

## Kiểm tra

```powershell
pytest -q
```

Test backend hiện cần catalog/index runtime và model weights sẵn có. Nếu thiếu, API sẽ báo index/catalog không khả dụng; đây không phải một lần benchmark thành công. Chạy notebook theo protocol LookBench để tái lập metric. Báo cáo chỉ nên được công bố cùng split, query/gallery, nhãn relevance, metric, seed, model/config và phần cứng thực sự dùng; không dùng điểm cấu hình làm bằng chứng nếu chưa chạy lại.

## Giới hạn hiện tại

- Không có script tải ABO/LookBench hoặc build index độc lập trong repo; tuân thủ điều khoản dataset và tự chuẩn bị artifact trước.
- Catalog LookBench hiện tại không có `title` hoặc `image_path`, nên UI dùng `product_id` làm nhãn. Thumbnail LookBench được đọc từ `gallery.parquet` tương ứng với `subset` và `item_ID`. Với ABO hoặc catalog khác, `image_path` có thể là đường dẫn tương đối bên trong `DATA_ROOT`; endpoint chỉ phục vụ file nằm trong thư mục này.
- Docker image build chưa được xác nhận trên máy sạch trong phiên này. Bộ test và benchmark cần data/model weights thực tế.
