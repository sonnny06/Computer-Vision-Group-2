"""
Giao diện chính Streamlit — Medical-VQA-Project (Phụ trách: Sơn).
Kết nối trực tiếp toàn bộ module của nhóm (Thái, Tôn, Trí, Bão, Phúc).
Chạy bằng lệnh:
    python -m streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for _p in (str(PROJECT_ROOT), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import streamlit as st

from components import inject_light_clinical_css, render_disclaimer, render_header, render_result_box
from src.engine.pipeline import build_pipeline
from src.inference.captioning import generate_caption
from src.inference.vqa import answer_question
from src.preprocessing.image_processing import preprocess_image, validate_image
from src.utils.history import SessionHistory
from src.utils.prompt import get_caption_prompt, get_vqa_prompt

DATA_DIR = PROJECT_ROOT / "data" / "evaluation"
IMAGES_DIR = DATA_DIR / "images"
RESULTS_DIR = PROJECT_ROOT / "results"

st.set_page_config(
    page_title="Medical-VQA | Qwen3-VL Multimodal AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_light_clinical_css()


@st.cache_data
def load_dataset():
    meta_df = pd.read_csv(DATA_DIR / "metadata.csv")
    q_df = pd.read_csv(DATA_DIR / "questions_handover_235.csv")
    return meta_df, q_df


@st.cache_resource(show_spinner="Đang khởi tạo pipeline của nhóm...")
def load_team_pipeline(use_mock: bool):
    return build_pipeline(use_mock=use_mock)


meta_df, q_df = load_dataset()

if "history_items" not in st.session_state:
    st.session_state.history_items = []
if "last_caption" not in st.session_state:
    st.session_state.last_caption = None
if "last_vqa" not in st.session_state:
    st.session_state.last_vqa = None
if "vqa_q_text" not in st.session_state:
    st.session_state.vqa_q_text = ""

history = SessionHistory(st.session_state.history_items)

with st.sidebar:
    st.markdown("## ⚙️ Cấu hình hệ thống")
    mode_opt = st.radio(
        "Chế độ Pipeline (Bão + Thái)",
        ["Mock / Demo nhanh (Không cần GPU)", "Qwen3-VL Thực tế (GPU CUDA)"],
        index=0,
    )
    use_mock = mode_opt.startswith("Mock")
    mode_tag = "mock" if use_mock else "qwen3-vl"

    pipeline = None
    pipeline_err = None
    try:
        pipeline = load_team_pipeline(use_mock=use_mock)
    except Exception as exc:  # noqa: BLE001
        pipeline_err = str(exc)
        st.error(f"Không thể nạp Qwen3-VL thực tế: {exc}")
        st.info("Đang tự động dùng chế độ Mock để kiểm thử giao diện.")
        pipeline = load_team_pipeline(use_mock=True)
        mode_tag = "mock-fallback"

    st.divider()
    st.markdown("### 📂 Dữ liệu nhóm đã tích hợp")
    st.write(f"- **Số ảnh hợp lệ:** `{len(meta_df)}` ảnh (7 vùng cơ thể)")
    st.write(f"- **Số câu hỏi VQA:** `{len(q_df)}` câu hỏi tiếng Việt")
    st.write(f"- **Lịch sử phiên:** `{len(history)}` lượt")

render_header(mode_tag, len(meta_df), len(q_df))

tab_main, tab_hist, tab_eval = st.tabs(
    [
        "🩺 Phân tích ảnh (F01 – F04)",
        f"🕘 Lịch sử phiên ({len(history)})",
        "📊 Kết quả đánh giá (F06)",
    ]
)

with tab_main:
    col_l, col_r = st.columns([1.0, 1.15], gap="large")
    active_source = None
    active_name = ""
    selected_row = None

    with col_l:
        st.subheader("1. Chọn hoặc tải ảnh y tế (F01)")
        src_mode = st.segmented_control(
            "Nguồn ảnh",
            ["Bộ ảnh kiểm thử của nhóm (35 ảnh)", "Tải ảnh từ máy (JPG/PNG)"],
            default="Bộ ảnh kiểm thử của nhóm (35 ảnh)",
        )
        if src_mode == "Tải ảnh từ máy (JPG/PNG)":
            uploaded = st.file_uploader("Chọn ảnh JPG, JPEG, PNG", type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                active_source = uploaded
                active_name = uploaded.name
        else:
            body_parts = ["Tất cả"] + sorted(meta_df["body_part"].dropna().unique().tolist())
            bp = st.selectbox("Lọc theo vùng cơ thể", body_parts)
            sub_df = meta_df if bp == "Tất cả" else meta_df[meta_df["body_part"] == bp]
            chosen_fn = st.selectbox(
                "Chọn ảnh y tế",
                sub_df["filename"].tolist(),
                format_func=lambda fn: f"{fn} — {sub_df.loc[sub_df['filename'] == fn, 'finding_category'].values[0]}",
            )
            selected_row = sub_df[sub_df["filename"] == chosen_fn].iloc[0]
            active_source = IMAGES_DIR / chosen_fn
            active_name = chosen_fn

        if active_source is not None:
            ok, msg, info = validate_image(active_source)
            if not ok:
                st.error(f"❌ Lỗi kiểm tra ảnh (Trí): {msg}")
                active_source = None
            else:
                img_rgb = preprocess_image(active_source, target_max_dim=768)
                st.image(img_rgb, caption=f"{active_name} ({img_rgb.size[0]}×{img_rgb.size[1]}px)", use_container_width=True)
                st.success(f"✓ {msg}")
                if selected_row is not None:
                    with st.expander("Thông tin lâm sàng tham chiếu (metadata.csv)"):
                        st.write(f"**Vùng cơ thể:** {selected_row['body_part']}")
                        st.write(f"**Tư thế / Modal:** {selected_row['view_position']} — {selected_row['finding_category']}")
                        st.write(f"**Mô tả chuẩn:** {selected_row['ground_truth_caption_vi']}")

    with col_r:
        st.subheader("2. Image Captioning (F02)")
        if st.button("✨ Generate Caption", type="primary", use_container_width=True, disabled=(active_source is None)):
            with st.spinner("Đang chạy pipeline Captioning..."):
                res = generate_caption(active_source, pipeline=pipeline)
                if res.success:
                    st.session_state.last_caption = {
                        "image": active_name,
                        "answer": res.answer,
                        "time_s": res.meta.get("time_s", 0.0),
                        "mode": mode_tag,
                        "prompt": get_caption_prompt(),
                    }
                    history.add("Captioning", active_name, res.answer or "", res.meta.get("time_s", 0.0), mode_tag)
                    st.session_state.history_items = history.to_dicts()
                else:
                    st.error(f"❌ [{res.error_code}] {res.error}")

        if st.session_state.last_caption:
            c = st.session_state.last_caption
            render_result_box(f"📝 Mô tả — {c['image']}", c["answer"], c["time_s"], c["mode"], kind="caption")
            with st.expander("Xem Prompt Captioning (Tôn)"):
                st.code(c["prompt"], language="text")

        st.divider()
        st.subheader("3. Medical VQA tiếng Việt (F03)")
        img_qs = q_df[q_df["image_name"] == active_name]["question_vi"].tolist()[:4] if active_name else []
        if img_qs:
            st.caption("Câu hỏi gợi ý từ bộ 235 câu bàn giao của nhóm:")
            cols = st.columns(2)
            for idx, q_txt in enumerate(img_qs):
                if cols[idx % 2].button(f"💡 {q_txt}", key=f"qq_{idx}", use_container_width=True):
                    st.session_state.vqa_q_text = q_txt

        q_input = st.text_area("Nhập câu hỏi tiếng Việt:", value=st.session_state.vqa_q_text, height=85)
        if st.button("💬 Ask a Question", type="primary", use_container_width=True, disabled=(active_source is None)):
            with st.spinner("Đang chạy pipeline VQA..."):
                res = answer_question(active_source, q_input, pipeline=pipeline)
                if res.success:
                    st.session_state.vqa_q_text = q_input
                    st.session_state.last_vqa = {
                        "image": active_name,
                        "question": q_input.strip(),
                        "answer": res.answer,
                        "time_s": res.meta.get("time_s", 0.0),
                        "mode": mode_tag,
                        "prompt": get_vqa_prompt(q_input),
                    }
                    history.add("VQA", active_name, res.answer or "", res.meta.get("time_s", 0.0), mode_tag, q_input.strip())
                    st.session_state.history_items = history.to_dicts()
                else:
                    st.error(f"❌ [{res.error_code}] {res.error}")

        if st.session_state.last_vqa:
            v = st.session_state.last_vqa
            render_result_box(
                f"💬 VQA — {v['image']}",
                f"<b>Câu hỏi:</b> {v['question']}<br/><br/><b>Trả lời:</b> {v['answer']}",
                v["time_s"],
                v["mode"],
                kind="vqa",
            )
            with st.expander("Xem Prompt VQA (Tôn)"):
                st.code(v["prompt"], language="text")

        render_disclaimer()

with tab_hist:
    st.subheader("🕘 Lịch sử phiên làm việc (F05)")
    c1, c2, c3, c4 = st.columns([1, 2, 1, 1])
    t_filter = c1.selectbox("Tác vụ", ["Tất cả", "Captioning", "VQA"])
    kw = c2.text_input("Tìm kiếm", placeholder="Nhập từ khóa...")
    with c3:
        st.write("")
        st.download_button("⬇️ CSV", history.to_csv(), "history.csv", "text/csv", use_container_width=True, disabled=(len(history) == 0))
    with c4:
        st.write("")
        if st.button("🗑️ Xóa hết", use_container_width=True, disabled=(len(history) == 0)):
            history.clear()
            st.session_state.history_items = []
            st.rerun()

    for item in history.list_items(t_filter, kw):
        st.markdown(f"**[{item.task_type}] `{item.image_name}`** • 🕒 {item.timestamp} • ⏱️ {item.time_s}s")
        st.write(f"**Hỏi:** {item.question}")
        st.write(f"**Đáp:** {item.answer}")
        st.divider()

with tab_eval:
    st.subheader("📊 Kết quả đánh giá từ thư mục `results/` (Phúc)")
    cap_csv = RESULTS_DIR / "caption_sample.csv"
    vqa_csv = RESULTS_DIR / "vqa_sample.csv"
    if cap_csv.exists():
        st.markdown("#### Captioning (`results/caption_sample.csv`)")
        st.dataframe(pd.read_csv(cap_csv), use_container_width=True, hide_index=True)
    if vqa_csv.exists():
        st.markdown("#### Medical VQA (`results/vqa_sample.csv`)")
        st.dataframe(pd.read_csv(vqa_csv), use_container_width=True, hide_index=True)
