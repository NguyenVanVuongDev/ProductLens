# Multimodal Visual Retrieval

Đây là project multimodal visual retrieval với thử nghiệm trên **ABO (đồ gia dụng)** và **LookBench (thời trang)**. Pipeline notebook ABO dùng FashionCLIP (0.7) + ViT-B/32-laion (0.3), embedding 1024 chiều, MURE v6 và FAISS; backend local mặc định hiện vẫn cấu hình cho LookBench. Hai domain được xử lý và báo cáo riêng.

## Trạng thái dự án

Notebook ABO đã chạy pipeline trên Kaggle: làm sạch subset 20,000 ảnh, mã hóa gallery/query, đánh giá held-out và xuất backend/index artifacts. Streamlit/FastAPI local và 25 backend tests cũng có sẵn. Các output dưới đây được ghi trong notebook; chưa được tái chạy trên máy local trong lần cập nhật này. ABO artifacts chưa được đưa vào runtime local mặc định, nên cần dùng package được xuất từ notebook hoặc cấu hình riêng trước khi demo ABO.

## ABO Results

Notebook ghi nhận 480 held-out queries trên 8,798 ảnh gallery, với bốn nhóm `cellular_phone_case`, `grocery`, `home`, `shoes` (120 query mỗi nhóm). Trong ablation, MURE v6 cải thiện I2I so với zero-shot ViT-B/32: Recall@1 tăng từ 52.08% lên 78.33%, Recall@10 từ 75.42% lên 93.33%, và mAP@10 từ 50.56% lên 81.56%. T2I Recall@10 tăng từ 82.50% lên 100.00%.

Held-out benchmark tổng hợp trong notebook:

| Flow / Index | Queries | Recall@1 | Recall@5 | Recall@10 | mAP@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| I2I / FlatIP | 480 | 78.34% | 88.96% | 93.54% | 81.53% | 84.11% |
| T2I / FlatIP | 480 | 98.96% | 100.00% | 100.00% | 86.12% | 90.73% |
| I2I / HNSW M32 ef256 | 480 | 78.34% | 88.96% | 93.54% | 81.53% | 84.11% |
| T2I / HNSW M32 ef256 | 480 | 98.96% | 100.00% | 100.00% | 86.12% | 90.73% |

HNSW M32/ef256 đạt ANN Recall@10 100% so với FlatIP trên benchmark này. Với 13,798 vectors (gallery cộng distractors), notebook ghi p50 32.021 ms I2I / 32.969 ms T2I, so với 117.162 / 118.303 ms của FlatIP; RAM 57.48 MB và build time 3477.83 ms. Thử thêm 5,000 distractors, notebook ghi nhận 0 distractor trong top-10 và ΔRecall@10 = 0.00%.

Các con số là output notebook đã lưu (PyTorch 2.11.0+cu128, FAISS 1.16.0, CUDA); chúng mô tả lần chạy Kaggle đó. Phần cứng GPU cụ thể không được ghi trong output. Dataset và ABO runtime artifacts chưa nằm trong checkout local này.

## Cấu trúc runtime

- `app/main.py`: FastAPI `/search/text`, `/search/image`, `/products/{product_id}`, `/products/{product_id}/image` và `/health`.
- `app/demo.py`: Streamlit UI gọi API, hỗ trợ text/image, top-K, score và metadata.
- `artifacts/indexes/lookbench_gallery_hnsw_m32_ef256.faiss`: index runtime mặc định (IndexHNSWFlat, M=32, efConstruction=200, efSearch=256).
- `artifacts/embeddings/transform_pipeline.pkl`: transform bundle v6 dùng chung cho text/image query.
- `artifacts/embeddings/domain_centering_means.npz`, `dual_ekma_W_QI.npy`, `dual_ekma_W_QT.npy`: transform artifacts dạng NumPy để kiểm tra/tái dựng.
- `data/lookbench/product_catalog.parquet`: catalog runtime mặc định, ánh xạ FAISS ID sang metadata.
- `notebooks/ABO DA1.ipynb`: chuẩn bị dữ liệu, embedding, ablation và benchmark ABO.
- `notebooks/da1-day-3-day-4.ipynb`: pipeline LookBench lịch sử; không dùng artifacts của notebook này thay cho ABO.

Index, catalog và transform phải cùng một lần build. API hiện tại mặc định dùng artifacts LookBench; notebook ABO xuất một bộ ABO riêng ở môi trường Kaggle. Docker chỉ đóng gói dịch vụ, không tải dữ liệu hoặc build index trong startup. Dữ liệu và artifact lớn không được đưa vào image.

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

Backend suite chạy với model weights và artifacts tương ứng; lần chạy local gần nhất đạt **25/25 tests**. Notebook ABO ghi nhận **22/22 tests** trong môi trường Kaggle. Khi tái lập benchmark, nên giữ lại split, query/gallery, relevance labels, seed, model/config và thông tin phần cứng cùng kết quả.

## Ghi chú triển khai

- ABO pipeline đã chạy trên Kaggle; dữ liệu không được lưu trong repository. Runtime local mặc định vẫn dùng LookBench, còn demo ABO cần bộ index/catalog/transforms được xuất cùng một lần build.
- LookBench catalog hiện không có `title` hoặc `image_path`, vì vậy UI dùng `product_id` làm nhãn và lấy thumbnail từ gallery parquet theo `subset`/`item_ID`.
- Các kết quả ABO trong notebook là output của lần chạy Kaggle đã lưu. Có thể tái chạy trong cùng cấu hình để kiểm tra khả năng tái lập và đóng gói artifacts cho môi trường local.
- Docker Compose đã có cấu hình API/UI; clean build trên một máy mới là bước xác nhận triển khai tiếp theo.
