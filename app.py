"""Streamlit demo: python -m streamlit run demo/app.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import core

SAMPLES = Path(__file__).parent / "samples"

st.set_page_config(page_title="Phát hiện lỗi bề mặt", layout="wide")


@st.cache_resource(show_spinner="Đang nạp mô hình...")
def get_model(code: str):
    model = core.load_model(core.weights_path(code))
    # The first inference is far slower than the rest; absorb it during loading.
    model.predict(np.zeros((core.IMGSZ, core.IMGSZ, 3), np.uint8), imgsz=core.IMGSZ, verbose=False)
    return model


@st.cache_data(max_entries=32, show_spinner="Đang phân tích ảnh...")
def run_detection(code: str, data: bytes):
    # Cached per (model, image), so moving the threshold slider does not rerun the model.
    return core.detect(get_model(code), data)


with st.sidebar:
    st.header("Mô hình")
    codes = core.available_models()
    code = st.selectbox("Chọn mô hình", codes, index=codes.index(core.DEFAULT_MODEL),
                        format_func=lambda c: f"{c} · {core.MODELS[c]}")
    tau_default = core.tau_star(code)
    st.header("Ảnh đầu vào")
    uploaded = st.file_uploader("Tải ảnh lên", type=["png", "jpg", "jpeg", "bmp"])
    names = sorted(p.name for p in SAMPLES.glob("*.png"))
    wanted = st.query_params.get("sample")  # lets a link open a given sample
    index = names.index(wanted) if wanted in names else 0
    sample = st.selectbox("Hoặc chọn ảnh mẫu", names, index=index, disabled=uploaded is not None)
    st.header("Ngưỡng quyết định")
    tau = st.slider("Ngưỡng τ", core.MIN_CONF, 0.90, max(core.MIN_CONF, tau_default), 0.01,
                    key=f"tau_{code}",  # each model starts at its own τ*
                    help=f"Giá trị mặc định τ* = {tau_default} của {code} được chọn trên tập val.")

st.title("Phát hiện lỗi bề mặt sản phẩm")
st.caption(f"Bản demo thử nghiệm · Mô hình {code} – {core.MODELS[code]} (seed {core.SEEDS[code]})"
           " · Dữ liệu huấn luyện: MVTec AD")

data = uploaded.getvalue() if uploaded else (SAMPLES / sample).read_bytes()

try:
    raw = run_detection(code, data)
    dets = core.apply_threshold(raw, tau)
    annotated = core.annotate(data, dets)
except core.InvalidImage as e:
    st.error(str(e))
    st.stop()

defect = bool(dets)
if defect:
    st.error("Kết luận: CÓ LỖI")
else:
    st.success("Kết luận: KHÔNG LỖI")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Điểm của ảnh", f"{core.image_score(raw):.3f}")
m2.metric("Ngưỡng τ", f"{tau:.2f}")
m3.metric("Số vùng lỗi", len(dets))
m4.metric("Thời gian xử lý", f"{raw.decode_ms + raw.infer_ms:.0f} ms")

left, right = st.columns([3, 2])
left.image(annotated, caption=f"Kết quả ({raw.width} × {raw.height} px)", width="stretch")
with right:
    st.subheader("Chi tiết vùng lỗi")
    if dets:
        st.dataframe(pd.DataFrame([{
            "Loại lỗi": d.cls,
            "Độ tin cậy": round(d.conf, 3),
            "Vị trí (x1, y1, x2, y2)": ", ".join(f"{v:.0f}" for v in d.xyxy),
        } for d in dets]), hide_index=True, width="stretch")
    else:
        st.write("Không có vùng nào đạt ngưỡng.")
