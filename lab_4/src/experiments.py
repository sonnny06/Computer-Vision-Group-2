"""
experiments.py - TV6: thử nghiệm & so sánh cấu hình Wavelet Hash (Bài thực hành 4)

Quy ước (khớp với TV5 và evaluation.py):
  - label 1 = cặp tương tự, 0 = không tương tự
  - distance = khoảng cách Hamming (số bit khác nhau), dự đoán tương tự nếu distance <= threshold
  - Điểm số cho ROC = -distance (được xử lý trong evaluation.roc_auc_from_distances)

Mọi cấu hình đều cho hash 64 bit (8x8) để so sánh công bằng:
  ảnh xám -> resize (8 * 2**level) -> wavedec2(level) -> lấy LL (8x8) -> lượng tử hoá thành bit.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import pandas as pd
import pywt

from evaluation import evaluate_thresholds, roc_auc_from_distances

HASH_SIZE = 8                      # 8 x 8 = 64 bit
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


# ------------------------------------------------------------------ hash tham số hoá
def wavelet_hash(image, wavelet="haar", level=1, quant="median", hash_size=HASH_SIZE):
    """Trả về mảng uint8 (hash_size**2 bit). Mặc định haar/1 mức/median ~ hàm của TV4.

    quant: 'median' (LL > trung vị) hoặc 'mean' (LL > trung bình).
    Lưu ý: dùng INTER_AREA khi thu nhỏ để ổn định hơn với ảnh lớn (TV4 dùng nội suy mặc định).
    """
    if isinstance(image, str):
        image = read_image(image)
    if image is None:
        raise ValueError("Ảnh không hợp lệ hoặc không đọc được")
    image = np.asarray(image)
    if image.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        image = cv2.cvtColor(image, code)
    elif image.ndim != 2:
        raise ValueError("Ảnh phải là 2D (xám) hoặc 3D (màu)")
    if level < 1:
        raise ValueError("level phải >= 1")

    side = hash_size * 2 ** level
    small = cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
    ll = pywt.wavedec2(small, wavelet, level=level, mode="periodization")[0]
    if ll.shape != (hash_size, hash_size):
        raise RuntimeError(f"LL có kích thước {ll.shape}, mong đợi {(hash_size, hash_size)}")

    if quant == "median":
        thr = np.median(ll)
    elif quant == "mean":
        thr = ll.mean()
    else:
        raise ValueError("quant phải là 'median' hoặc 'mean'")
    return (ll > thr).astype(np.uint8).ravel()


def bits_to_hex(bits) -> str:
    bits = np.asarray(bits)
    return format(int("".join(map(str, bits.tolist())), 2), f"0{len(bits) // 4}x")


def hamming(b1, b2) -> int:
    if len(b1) != len(b2):
        raise ValueError(f"Hash khác độ dài: {len(b1)} vs {len(b2)}")
    return int(np.count_nonzero(np.asarray(b1) != np.asarray(b2)))


# ------------------------------------------------------------------ nạp ảnh
def read_image(path):
    """Đọc ảnh qua np.fromfile để hỗ trợ đường dẫn có dấu tiếng Việt trên Windows."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def load_images(folder, max_side=512):
    """Đọc toàn bộ ảnh trong thư mục -> {tên: ảnh BGR}, thu nhỏ để chạy nhanh."""
    out = {}
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(IMAGE_EXTS):
            img = read_image(os.path.join(folder, name))
            if img is None:
                continue
            scale = max_side / max(img.shape[:2])
            if scale < 1:
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            out[os.path.splitext(name)[0]] = img
    return out


def make_demo_images(n=20, size=256, seed=0):
    """Ảnh giả lập (nhiễu tần số thấp + vài hình) khi chưa có dữ liệu thật."""
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        low = rng.integers(0, 256, (6, 6, 3), dtype=np.uint8)
        img = cv2.resize(low, (size, size), interpolation=cv2.INTER_CUBIC)
        for _ in range(3):
            c = tuple(int(v) for v in rng.integers(0, 256, 3))
            p1 = tuple(int(v) for v in rng.integers(0, size, 2))
            if rng.random() < 0.5:
                cv2.circle(img, p1, int(rng.integers(15, 60)), c, -1)
            else:
                p2 = tuple(int(v) for v in rng.integers(0, size, 2))
                cv2.rectangle(img, p1, p2, c, -1)
        out[f"demo_{i:02d}"] = img
    return out


# ------------------------------------------------------------------ biến đổi tạo cặp tương tự
def _noise(img, sigma=15):
    rng = np.random.default_rng(0)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def _rotate(img, deg=5):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def _crop(img, frac=0.9):
    h, w = img.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    y, x = (h - ch) // 2, (w - cw) // 2
    return cv2.resize(img[y:y + ch, x:x + cw], (w, h), interpolation=cv2.INTER_AREA)


def _jpeg(img, q=30):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


TRANSFORMS = {
    "darken":      lambda im: cv2.convertScaleAbs(im, alpha=0.8, beta=0),
    "brighten":    lambda im: cv2.convertScaleAbs(im, alpha=1.2, beta=0),
    "gauss_noise": _noise,
    "blur":        lambda im: cv2.GaussianBlur(im, (7, 7), 0),
    "jpeg_q30":    _jpeg,
    "rotate_5deg": _rotate,
    "crop_90pct":  _crop,
}


def build_dataset(base_images: dict, transforms=TRANSFORMS, neg_per_pos=1.0, seed=0):
    """Tạo (images, pairs).

    Cặp TƯƠNG TỰ (label=1): ảnh gốc vs bản biến đổi của chính nó.
    Cặp KHÁC (label=0)    : ảnh gốc i vs (gốc hoặc bản biến đổi) của ảnh j != i.
    pairs: img1, img2, label, variant (tên biến đổi, 'none' = ảnh gốc).
    """
    if len(base_images) < 2:
        raise ValueError("Cần ít nhất 2 ảnh gốc để tạo cặp khác nhau")
    rng = np.random.default_rng(seed)
    images = dict(base_images)
    for name, img in base_images.items():
        for t, fn in transforms.items():
            images[f"{name}::{t}"] = fn(img)

    names = list(base_images)
    rows = [(n, f"{n}::{t}", 1, t) for n in names for t in transforms]
    cands = [(i, j, t) for i in names for j in names if i != j for t in [None, *transforms]]
    k = min(len(cands), int(round(len(rows) * neg_per_pos)))
    for idx in rng.choice(len(cands), size=k, replace=False):
        i, j, t = cands[idx]
        rows.append((i, j if t is None else f"{j}::{t}", 0, t or "none"))
    return images, pd.DataFrame(rows, columns=["img1", "img2", "label", "variant"])


# ------------------------------------------------------------------ chạy thử nghiệm
def compute_distances(pairs, images, **cfg) -> pd.DataFrame:
    """Thêm cột distance (khoảng cách Hamming) cho mọi cặp với cấu hình hash cfg."""
    cache = {}
    def h(key):
        if key not in cache:
            cache[key] = wavelet_hash(images[key], **cfg)
        return cache[key]
    out = pairs.copy()
    out["distance"] = [hamming(h(a), h(b)) for a, b in zip(pairs.img1, pairs.img2)]
    return out


def best_threshold(labels, distances, n_bits=HASH_SIZE ** 2):
    """Chọn ngưỡng theo Youden J = recall + specificity - 1 (hoà thì chọn ngưỡng nhỏ hơn)."""
    rows = evaluate_thresholds(list(labels), [int(d) for d in distances], list(range(n_bits + 1)))
    tab = pd.DataFrame(rows)
    tab["youden"] = tab["recall"] + tab["specificity"] - 1
    return int(tab.loc[tab["youden"].idxmax(), "threshold"]), tab


def run_config(pairs, images, **cfg):
    """Một cấu hình -> dict gồm bảng khoảng cách, ROC/AUC, ngưỡng tốt nhất và chỉ số tại ngưỡng đó."""
    df = compute_distances(pairs, images, **cfg)
    y, d = df.label.astype(int).tolist(), df.distance.astype(int).tolist()
    roc = roc_auc_from_distances(y, d)
    thr, table = best_threshold(y, d)
    row = table[table.threshold == thr].iloc[0].to_dict()
    return {"df": df, "roc": roc, "threshold": thr, "metrics": row, "threshold_table": table}


def compare_configs(pairs, images, wavelets=("haar", "db2", "sym2", "coif1"),
                    levels=(1, 2, 3), quants=("median", "mean")):
    """Lưới cấu hình -> (bảng tổng hợp, {tên cấu hình: kết quả run_config})."""
    results, rows = {}, []
    for w in wavelets:
        for lv in levels:
            for q in quants:
                name = f"{w}|L{lv}|{q}"
                r = run_config(pairs, images, wavelet=w, level=lv, quant=q)
                results[name] = r
                m = r["metrics"]
                rows.append(dict(config=name, wavelet=w, level=lv, quant=q, auc=r["roc"]["auc"],
                                 threshold=r["threshold"], accuracy=m["accuracy"], recall=m["recall"],
                                 specificity=m["specificity"], precision=m["precision"]))
    summary = (pd.DataFrame(rows).sort_values(["auc", "accuracy"], ascending=False)
               .reset_index(drop=True))
    return summary, results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="So sánh cấu hình Wavelet Hash")
    ap.add_argument("--data", default=None, help="thư mục ảnh gốc (bỏ trống = dữ liệu giả lập)")
    ap.add_argument("--out", default="output")
    a = ap.parse_args()
    base = load_images(a.data) if a.data else make_demo_images()
    imgs, prs = build_dataset(base)
    summ, _ = compare_configs(prs, imgs)
    os.makedirs(a.out, exist_ok=True)
    summ.to_csv(os.path.join(a.out, "experiments_summary.csv"), index=False)
    print(summ.round(3).to_string())
