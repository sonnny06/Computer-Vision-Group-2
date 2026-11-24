"""
Các thành phần giao diện tái sử dụng cho ứng dụng Streamlit (Phụ trách: Sơn).
"""

from __future__ import annotations

import streamlit as st


def inject_light_clinical_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
            max-width: 1380px;
        }
        .header-bar {
            background: linear-gradient(135deg, #0f172a 0%, #0e7490 55%, #0284c7 100%);
            padding: 1.35rem 1.75rem;
            border-radius: 14px;
            color: #ffffff;
            margin-bottom: 1.15rem;
        }
        .header-bar h1 {
            margin: 0 0 0.25rem 0;
            font-size: 1.55rem;
            color: #ffffff;
        }
        .header-bar p {
            margin: 0;
            font-size: 0.92rem;
            color: #e0f2fe;
        }
        .result-card {
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-left: 4px solid #0284c7;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin: 0.65rem 0;
            color: #0f172a;
        }
        .result-card.vqa {
            background: #f0fdfa;
            border-color: #99f6e4;
            border-left-color: #0d9488;
        }
        .result-meta {
            font-size: 0.78rem;
            font-weight: 700;
            color: #0369a1;
            margin-bottom: 0.4rem;
            display: flex;
            justify-content: space-between;
        }
        .result-card.vqa .result-meta { color: #0f766e; }
        .disclaimer-box {
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-left: 4px solid #d97706;
            border-radius: 10px;
            padding: 0.75rem 1rem;
            font-size: 0.84rem;
            color: #92400e;
            margin-top: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(mode_label: str, total_images: int, total_questions: int) -> None:
    st.markdown(
        f"""
        <div class="header-bar">
            <h1>🩺 Medical-VQA — Hệ thống AI Đa phương thức Phân tích Ảnh Y tế</h1>
            <p>
                Tích hợp Qwen3-VL • <b>{total_images} ảnh y tế thực tế (7 vùng cơ thể)</b> •
                <b>{total_questions} câu hỏi kiểm thử tiếng Việt</b> • Chế độ: <b>{mode_label}</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_box(title: str, body: str, time_s: float, mode: str, kind: str = "caption") -> None:
    cls = "result-card vqa" if kind == "vqa" else "result-card"
    st.markdown(
        f"""
        <div class="{cls}">
            <div class="result-meta">
                <span>{title}</span>
                <span>⏱️ {time_s}s • {mode}</span>
            </div>
            <div>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    st.markdown(
        """
        <div class="disclaimer-box">
            ⚠️ <b>Lưu ý y khoa:</b> Kết quả do mô hình AI đa phương thức (Qwen3-VL) sinh ra chỉ phục vụ mục đích
            học thuật và hỗ trợ tham khảo trực quan, không thay thế chẩn đoán lâm sàng của bác sĩ chuyên khoa.
        </div>
        """,
        unsafe_allow_html=True,
    )
