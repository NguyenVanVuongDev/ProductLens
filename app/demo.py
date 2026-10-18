import os
import time
from urllib.parse import quote

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Visual Retrieval Demo", page_icon="🔎", layout="wide")
st.title("Multimodal Visual Retrieval")
st.caption("LookBench retrieval API · FashionCLIP + ViT-B/32 ensemble")


def show_results(payload: dict, elapsed: float) -> None:
    results = payload.get("results", [])
    st.caption(f"{len(results)} kết quả · {elapsed * 1000:.0f} ms · rerank: {'on' if payload.get('rerank_applied') else 'off'}")
    for result in results:
        product = result.get("product", {})
        title = product.get("title") or product.get("product_id", "Sản phẩm")
        product_id = quote(str(product.get("product_id", "")), safe="")
        st.divider()
        st.subheader(f"{result.get('rank', 0)}. {title}")
        image_url = f"{API_URL}/products/{product_id}/image"
        try:
            image_response = requests.get(image_url, timeout=15)
            image_response.raise_for_status()
            st.image(image_response.content, caption=title, width=280)
        except requests.RequestException:
            st.caption("Không có ảnh gallery")
        st.write(f"**Category:** {product.get('category') or 'N/A'}")
        st.write(f"**Similarity:** {result.get('score', 0):.4f}")
        st.caption(
            f"Product ID: {product.get('product_id', 'N/A')} · "
            f"{product.get('main_attribute') or 'No main attribute'} · "
            f"{product.get('other_attributes') or 'No additional attributes'}"
        )


with st.sidebar:
    st.subheader("Service")
    st.code(API_URL, language=None)
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.ok:
            status = health.json()
            st.success(f"Ready · {status['total_indexed_vectors']:,} vectors")
            st.caption(f"Models: {', '.join(status['models'])}")
        else:
            st.error(f"API returned {health.status_code}: {health.text}")
    except requests.RequestException as exc:
        st.error(f"Không kết nối được API: {exc}")
    top_k = st.slider("Top K", min_value=1, max_value=50, value=10)


text_tab, image_tab = st.tabs(["Text-to-image", "Image-to-image"])
with text_tab:
    with st.form("text-search"):
        query = st.text_input("Mô tả sản phẩm", value="A modern beige linen sofa for a small living room.")
        submitted = st.form_submit_button("Search", type="primary")
    if submitted:
        if not query.strip():
            st.warning("Nhập mô tả trước khi tìm kiếm.")
        else:
            started = time.perf_counter()
            try:
                response = requests.post(
                    f"{API_URL}/search/text",
                    json={"query": query, "top_k": top_k},
                    timeout=180,
                )
                elapsed = time.perf_counter() - started
                response.raise_for_status()
                show_results(response.json(), elapsed)
            except requests.RequestException as exc:
                st.error(f"Text search thất bại: {exc}")

with image_tab:
    with st.form("image-search"):
        uploaded = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "webp"])
        submitted = st.form_submit_button("Search", type="primary")
    if submitted:
        if uploaded is None:
            st.warning("Chọn ảnh trước khi tìm kiếm.")
        else:
            st.image(uploaded, caption=uploaded.name, width=240)
            started = time.perf_counter()
            try:
                response = requests.post(
                    f"{API_URL}/search/image",
                    params={"top_k": top_k},
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=180,
                )
                elapsed = time.perf_counter() - started
                response.raise_for_status()
                show_results(response.json(), elapsed)
            except requests.RequestException as exc:
                st.error(f"Image search thất bại: {exc}")