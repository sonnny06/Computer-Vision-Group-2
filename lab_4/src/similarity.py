"""
similarity.py - TV5: Khoảng cách Hamming và so khớp ảnh (Bài thực hành 4, Wavelet hash)

Quy ước chung với TV6 (ROC):
  - label      : 1 = cặp ảnh TƯƠNG TỰ, 0 = KHÔNG tương tự
  - distance   : khoảng cách Hamming (số bit khác nhau), càng nhỏ càng giống
  - prediction : 1 nếu distance < threshold (mặc định, nghiêm ngặt)
                 hoặc distance <= threshold nếu inclusive=True
  - Điểm số cho ROC nên dùng  -distance  (hoặc 1 - norm_distance) vì điểm cao = giống.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "to_bits", "hamming_distance", "normalized_hamming", "predict_similar",
    "build_distance_table", "sweep_thresholds", "evaluate_thresholds",
]


# ----------------------------------------------------------------- kiểm tra đầu vào
def to_bits(h) -> np.ndarray:
    """Chuẩn hoá hash về mảng numpy uint8 gồm 0/1. Ném lỗi nếu đầu vào không hợp lệ.

    Chấp nhận: chuỗi '0101...', list/tuple/ndarray gồm 0/1 hoặc bool.
    """
    if h is None:
        raise TypeError("Hash là None")
    if isinstance(h, str):
        s = h.strip()
        if not s:
            raise ValueError("Hash rỗng")
        if set(s) - {"0", "1"}:
            raise ValueError("Hash chuỗi chỉ được chứa ký tự '0' và '1'")
        return np.frombuffer(s.encode("ascii"), dtype=np.uint8) - ord("0")
    if isinstance(h, (list, tuple, np.ndarray)):
        a = np.asarray(h)
        if a.ndim != 1:
            raise ValueError("Hash phải là mảng 1 chiều (hãy flatten trước)")
        if a.size == 0:
            raise ValueError("Hash rỗng")
        if a.dtype == bool:
            return a.astype(np.uint8)
        if not np.issubdtype(a.dtype, np.integer) and not np.issubdtype(a.dtype, np.floating):
            raise TypeError(f"Kiểu dữ liệu hash không hợp lệ: {a.dtype}")
        if not np.isin(a, (0, 1)).all():
            raise ValueError("Hash chỉ được chứa giá trị 0 hoặc 1")
        return a.astype(np.uint8)
    raise TypeError(f"Kiểu hash không được hỗ trợ: {type(h).__name__}")


# ----------------------------------------------------------------- khoảng cách
def hamming_distance(h1, h2) -> int:
    """Số bit khác nhau giữa hai mã hash. Ném ValueError nếu độ dài khác nhau."""
    a, b = to_bits(h1), to_bits(h2)
    if a.size != b.size:
        raise ValueError(f"Hash có độ dài khác nhau: {a.size} vs {b.size}")
    return int(np.count_nonzero(a != b))


def normalized_hamming(h1, h2) -> float:
    """Khoảng cách Hamming chuẩn hoá về [0, 1] (chia cho độ dài hash)."""
    return hamming_distance(h1, h2) / to_bits(h1).size


# ----------------------------------------------------------------- phân loại theo ngưỡng
def predict_similar(distance, threshold, inclusive: bool = False):
    """1 (tương tự) nếu distance < threshold (hoặc <= nếu inclusive=True), ngược lại 0.

    Nhận số đơn hoặc mảng/Series; trả về cùng dạng (int / ndarray).
    """
    d = np.asarray(distance)
    pred = (d <= threshold) if inclusive else (d < threshold)
    pred = pred.astype(int)
    return int(pred) if pred.ndim == 0 else pred


# ----------------------------------------------------------------- bảng khoảng cách
def build_distance_table(hashes: dict, pairs: pd.DataFrame) -> pd.DataFrame:
    """Tính khoảng cách cho từng cặp ảnh, không dừng khi gặp cặp lỗi.

    hashes : {image_id: hash}
    pairs  : DataFrame có cột img1, img2, label (label: 1 tương tự / 0 không)
    Trả về : img1, img2, label, hash_len, distance, norm_distance, status
             status = ok | missing_hash | invalid_hash | length_mismatch
    """
    need = {"img1", "img2", "label"}
    if not need.issubset(pairs.columns):
        raise ValueError(f"pairs thiếu cột: {need - set(pairs.columns)}")

    rows = []
    for r in pairs.itertuples(index=False):
        rec = dict(img1=r.img1, img2=r.img2, label=r.label,
                   hash_len=np.nan, distance=np.nan, norm_distance=np.nan, status="ok")
        if r.img1 not in hashes or r.img2 not in hashes:
            rec["status"] = "missing_hash"
        else:
            try:
                a, b = to_bits(hashes[r.img1]), to_bits(hashes[r.img2])
                if a.size != b.size:
                    rec["status"] = "length_mismatch"
                else:
                    d = int(np.count_nonzero(a != b))
                    rec.update(hash_len=a.size, distance=d, norm_distance=d / a.size)
            except (TypeError, ValueError):
                rec["status"] = "invalid_hash"
        rows.append(rec)
    return pd.DataFrame(rows)


def sweep_thresholds(table: pd.DataFrame, thresholds, inclusive: bool = False) -> pd.DataFrame:
    """Thêm cột pred_t<ngưỡng> cho từng ngưỡng (chỉ với các cặp status == ok)."""
    out = table.copy()
    ok = out["status"] == "ok"
    for t in thresholds:
        col = f"pred_t{t:g}"
        out[col] = np.nan
        out.loc[ok, col] = predict_similar(out.loc[ok, "distance"].to_numpy(), t, inclusive)
    return out


def evaluate_thresholds(table: pd.DataFrame, thresholds, inclusive: bool = False) -> pd.DataFrame:
    """Bảng TP/FP/TN/FN + accuracy/sensitivity/specificity theo ngưỡng (để kiểm tra nhanh;
    đánh giá chính thức & ROC do TV6 thực hiện)."""
    ok = table[table["status"] == "ok"]
    y = ok["label"].to_numpy().astype(int)
    res = []
    for t in thresholds:
        p = predict_similar(ok["distance"].to_numpy(), t, inclusive)
        tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
        tn = int(((p == 0) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
        res.append(dict(
            threshold=t, TP=tp, FP=fp, TN=tn, FN=fn,
            accuracy=(tp + tn) / max(len(y), 1),
            sensitivity=tp / (tp + fn) if tp + fn else np.nan,
            specificity=tn / (tn + fp) if tn + fp else np.nan,
        ))
    return pd.DataFrame(res)
