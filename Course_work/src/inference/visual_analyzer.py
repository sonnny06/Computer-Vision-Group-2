"""
Module phân tích thị giác máy tính (OpenCV + NumPy) hỗ trợ:
1. So khớp tương đồng hình ảnh (Visual Retrieval) với bộ dữ liệu 35 ảnh chuẩn của nhóm
   để nhận diện vùng cơ thể / phương pháp chụp cho BẤT KỲ ảnh y tế nào người dùng tải lên.
2. Phát hiện và khoanh vùng dị thường / vùng chú ý (Anomaly & ROI Localization)
   dựa trên phân tích bất đối xứng trái-phải và độ lệch cường độ cục bộ (CLAHE + Saliency).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def compute_visual_descriptor(img_rgb: Image.Image) -> np.ndarray:
    """Trích xuất vector đặc trưng nhỏ gọn (spatial grid + histogram + edge) cho ảnh y tế."""
    gray = np.asarray(img_rgb.convert("L"), dtype=np.uint8)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    spatial = small.flatten()  # 1024 dims

    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-6)

    edges = cv2.Canny(cv2.resize(gray, (128, 128)), 50, 140)
    edge_grid = cv2.resize(edges, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).flatten() / 255.0

    feat = np.concatenate([spatial * 1.0, hist * 2.5, edge_grid * 0.8]).astype(np.float32)
    norm = float(np.linalg.norm(feat)) + 1e-8
    return feat / norm


def _quadrant_label(cx: float, cy: float) -> str:
    horiz = "bên phải ảnh" if cx < 0.42 else ("bên trái ảnh" if cx > 0.58 else "vùng trung tâm")
    vert = "phần trên" if cy < 0.36 else ("phần dưới" if cy > 0.64 else "phần giữa")
    return f"{vert}, {horiz}"


def detect_anomaly_regions(img_rgb: Image.Image, max_regions: int = 2) -> dict[str, Any]:
    """
    Phát hiện vùng nghi ngờ bất thường / vùng chú ý (ROI) bằng Thị giác máy tính:
    - Tách vùng giải phẫu (foreground) loại bỏ nền đen viền phim.
    - Tính bản đồ bất đối xứng trái-phải (hiệu quả cho X-quang phổi, CT não, khung chậu).
    - Kết hợp bản đồ lệch cường độ cục bộ (CLAHE + DoG) để bắt vùng mờ/tăng tỷ trọng/đường gãy.
    Trả về tọa độ chuẩn hóa (0..1) để vẽ trực tiếp lên ảnh trên Web UI.
    """
    gray_full = np.asarray(img_rgb.convert("L"), dtype=np.uint8)
    work_size = 256
    gray = cv2.resize(gray_full, (work_size, work_size), interpolation=cv2.INTER_AREA)

    # 1. Tách nền đen viền phim (foreground mask)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, fg_mask = cv2.threshold(blur, 18, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.erode(fg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    if fg_mask.sum() < 255 * 200:
        fg_mask = np.full((work_size, work_size), 255, dtype=np.uint8)

    # 2. Cân bằng tương phản cục bộ (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enh = clahe.apply(gray).astype(np.float32)

    # 3. Bản đồ bất đối xứng trái - phải
    flipped = cv2.flip(enh, 1)
    asym = np.abs(enh - flipped)
    asym = cv2.GaussianBlur(asym, (15, 15), 0)

    # 4. Bản đồ nổi bật cục bộ (độ lệch so với trung bình vùng lân cận)
    local_mean = cv2.GaussianBlur(enh, (45, 45), 0)
    local_dev = np.abs(enh - local_mean)
    local_dev = cv2.GaussianBlur(local_dev, (13, 13), 0)

    # 5. Tổng hợp điểm chú ý trong vùng foreground
    mask_f = (fg_mask > 0).astype(np.float32)
    # Loại bỏ viền sát mép 8% để tránh bắt nhãn chữ X-quang (L/R)
    border = int(work_size * 0.08)
    mask_f[:border, :] = 0
    mask_f[-border:, :] = 0
    mask_f[:, :border] = 0
    mask_f[:, -border:] = 0

    score_map = (0.58 * asym + 0.42 * local_dev) * mask_f
    max_val = float(score_map.max())
    if max_val < 1e-3:
        return {"regions": [], "summary": "Phân bố cường độ đồng đều, chưa ghi nhận vùng dị thường nổi bật."}

    norm_map = (score_map / max_val * 255.0).astype(np.uint8)
    thresh_val = int(np.percentile(norm_map[mask_f > 0], 93)) if np.any(mask_f > 0) else 200
    thresh_val = max(135, min(225, thresh_val))
    _, bin_map = cv2.threshold(norm_map, thresh_val, 255, cv2.THRESH_BINARY)
    bin_map = cv2.morphologyEx(bin_map, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    contours, _ = cv2.findContours(bin_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, Any]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 90:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Tính mức độ nổi bật trung bình trong vùng
        roi_score = float(norm_map[y : y + h, x : x + w].mean()) / 255.0
        cx = (x + w / 2.0) / work_size
        cy = (y + h / 2.0) / work_size
        candidates.append(
            {
                "x": round(x / work_size, 4),
                "y": round(y / work_size, 4),
                "w": round(w / work_size, 4),
                "h": round(h / work_size, 4),
                "cx": round(cx, 4),
                "cy": round(cy, 4),
                "score": round(min(0.96, max(0.52, roi_score)), 2),
                "location": _quadrant_label(cx, cy),
                "area": area,
            }
        )

    candidates.sort(key=lambda r: (r["score"] * np.sqrt(r["area"])), reverse=True)
    top_regions = []
    for idx, reg in enumerate(candidates[:max_regions], start=1):
        reg_clean = {k: v for k, v in reg.items() if k != "area"}
        reg_clean["id"] = idx
        reg_clean["label"] = f"Vùng chú ý #{idx} ({reg['location']})"
        top_regions.append(reg_clean)

    if top_regions:
        loc_text = "; ".join(f"{r['label']} (mức chú ý {int(r['score']*100)}%)" for r in top_regions)
        summary = f"Phát hiện {len(top_regions)} vùng có độ lệch cường độ/bất đối xứng nổi bật tại: {loc_text}."
    else:
        summary = "Phân bố mật độ ảnh tương đối đồng đều, không ghi nhận vùng bất đối xứng rõ rệt."

    return {"regions": top_regions, "summary": summary}


class VisualKnowledgeIndex:
    """Chỉ mục so khớp ảnh nhanh trên toàn bộ tập dữ liệu 35 ảnh của nhóm."""

    def __init__(self, meta_rows: list[dict[str, Any]], images_dir: Path) -> None:
        self.entries: list[dict[str, Any]] = []
        self.vectors: list[np.ndarray] = []
        for row in meta_rows:
            fn = (row.get("filename") or "").strip()
            p = images_dir / fn
            if not p.exists():
                continue
            try:
                with Image.open(p) as im:
                    vec = compute_visual_descriptor(im.convert("RGB"))
                self.entries.append(row)
                self.vectors.append(vec)
            except Exception:
                continue
        self.matrix = np.vstack(self.vectors) if self.vectors else np.zeros((0, 1), dtype=np.float32)

    def find_similar(self, img_rgb: Image.Image, top_k: int = 3) -> list[dict[str, Any]]:
        if len(self.entries) == 0:
            return []
        q_vec = compute_visual_descriptor(img_rgb)
        sims = self.matrix @ q_vec
        idxs = np.argsort(sims)[::-1][:top_k]
        results = []
        for i in idxs:
            row = self.entries[int(i)]
            results.append(
                {
                    "filename": row.get("filename"),
                    "image_id": row.get("image_id"),
                    "body_part": row.get("body_part"),
                    "view_position": row.get("view_position"),
                    "finding_category": row.get("finding_category"),
                    "finding_description": row.get("finding_description"),
                    "ground_truth_caption_vi": row.get("ground_truth_caption_vi"),
                    "similarity": round(float(sims[int(i)]), 3),
                }
            )
        return results
