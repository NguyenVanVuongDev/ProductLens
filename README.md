# Multimodal Visual Retrieval

Đây là **project cá nhân dạng MVP** cho tìm kiếm thời trang text-to-image và image-to-image trên LookBench. API dùng ensemble FashionCLIP (0.7) + ViT-B/32-laion (0.3), embedding 1024 chiều và FAISS HNSW; đây không phải baseline CLIP ViT-B/32 thuần. ABO là domain đồ gia dụng, còn LookBench là benchmark thời trang; không gộp hai domain hoặc suy diễn nhãn benchmark từ metadata sản phẩm.

## Trạng thái dự án

Backend LookBench, Streamlit demo, artifact index/transform và 25 backend tests hiện có. Tuy nhiên, project **chưa hoàn thành toàn bộ Definition of Done của lộ trình 5 ngày**: `data/abo/` đang trống, chưa có loader/embedding/index build CLI độc lập, benchmark reports có số latency/RAM/build-time/ANN recall nhưng các relevance metrics và query counts chưa thống nhất, và Docker chưa được xác minh trên máy sạch. Không nên mô tả hiện trạng là hệ thống ABO hoàn chỉnh hoặc pipeline benchmark tái lập end-to-end.

Các metric dưới đây là số liệu được ghi trong artifact/config, không phải benchmark vừa chạy lại trong môi trường hiện tại:

| Flow (FlatIP, held-out test trong config) | Recall@1 | Recall@5 | Recall@10 | mAP@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| Image-to-image | 75.50% | 86.60% | 92.77% | 79.87% | 82.83% |
| Text-to-image | 90.09% | 92.52% | 94.32% | 76.87% | 81.87% |

Có hai discrepancy cần giải quyết trước khi công bố một con số chuẩn: config ghi 469 test queries, trong khi CSV metrics cộng các split thành 468; ngoài ra `day3_faiss_benchmark_full.csv` có relevance scores khác `best_config.json`/`lookbench_unified_all_metrics.csv`. Báo cáo FAISS riêng có ANN Recall@10, latency, RAM và build time, nhưng chưa chứng minh cùng split/config với unified v6 nên không ghép các bảng thành một kết quả duy nhất.

Ví dụ từ báo cáo FAISS đã lưu: HNSW M32/ef256 đạt ANN Recall@10 99.98%, p50 33.413 ms, p95 37.169 ms, RAM 57.40 MB và build 3567.99 ms; FlatIP ghi p50 124.278 ms, p95 128.202 ms, RAM 53.82 MB và build 53.18 ms. Đây là số liệu trong artifact, chưa được benchmark lại trong phiên này; cần kèm hardware/runtime details để so sánh có thể tái lập.

## Cấu trúc runtime

- `app/main.py`: FastAPI `/search/text`, `/search/image`, `/products/{product_id}`, `/products/{product_id}/image` và `/health`.
- `app/demo.py`: Streamlit UI gọi API, hỗ trợ text/image, top-K, score và metadata.
- `artifacts/indexes/lookbench_gallery_hnsw_m32_ef256.faiss`: index runtime mặc định (IndexHNSWFlat, M=32, efConstruction=200, efSearch=256).
- `artifacts/embeddings/transform_pipeline.pkl`: transform bundle v6 dùng chung cho text/image query.
- `artifacts/embeddings/domain_centering_means.npz`, `dual_ekma_W_QI.npy`, `dual_ekma_W_QT.npy`: transform artifacts dạng NumPy để kiểm tra/tái dựng.
- `data/lookbench/product_catalog.parquet`: catalog runtime mặc định, ánh xạ FAISS ID sang metadata.
- `notebooks/da1-day-3-day-4.ipynb`: notebook pipeline và đánh giá hiện có.

Index, catalog và transform v6 phải thuộc cùng một lần build trước khi khởi chạy API. API từ chối khởi động nếu transform thiếu/sai dimension hoặc dimension index không khớp; không fallback âm thầm sang embedding raw. Tên index trước đây ghi `m16` dù index thực tế là M32; artifact hiện đã được đổi tên để phản ánh cấu hình bên trong. Docker chỉ đóng gói dịch vụ, không tải dữ liệu hoặc build index trong quá trình startup. Repo hiện chưa có CLI độc lập để tái tạo index từ dữ liệu thô; quy trình tạo artifact vẫn nằm trong notebook. Dữ liệu `data/` và artifact lớn không được đưa vào image.

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

- Không có script tải ABO/LookBench hoặc build index độc lập trong repo; thư mục `data/abo/` hiện trống. Tuân thủ điều khoản dataset và tự chuẩn bị dữ liệu được phép sử dụng.
- Catalog LookBench hiện tại không có `title` hoặc `image_path`, nên UI dùng `product_id` làm nhãn. Thumbnail LookBench được đọc từ `gallery.parquet` tương ứng với `subset` và `item_ID`. Với ABO hoặc catalog khác, `image_path` có thể là đường dẫn tương đối bên trong `DATA_ROOT`; endpoint chỉ phục vụ file nằm trong thư mục này.
- Notebook có outputs và Papermill execution metadata hoàn tất từ 2026-10-17; mình chưa tái chạy/kiểm toán notebook trong lượt này. Cần xác minh môi trường, seed, split, query count (468/469 đang lệch) và đối chiếu các metric giữa reports trước khi công bố một benchmark tái lập. Bộ test và benchmark cần data/model weights thực tế.
- Docker image build chưa được xác nhận trên máy sạch. Xem [docs/PROJECT_INTERVIEW_GUIDE_VI.md](docs/PROJECT_INTERVIEW_GUIDE_VI.md) để có phần giải thích kiến trúc, metric, trade-off và câu hỏi phỏng vấn.
