"""
common.py - TV1: quy ước chung, đường dẫn và hàm tích hợp dùng cho toàn bộ chương trình.

CẤU TRÚC THƯ MỤC (common.py nằm trong lab_4/src):
    lab_4/
      src/   common.py main.py wavelet_hash.py similarity.py evaluation.py experiments.py
      data/  input/ (ảnh gốc)   pairs.csv (tuỳ chọn)   output/ (kết quả)

QUY ƯỚC ĐẦU VÀO / ĐẦU RA GIỮA CÁC MODULE
  wavelet_hash.py (TV4)  wavelet_hash(ảnh_hoặc_đường_dẫn) -> chuỗi hex 16 ký tự (= 64 bit)
  similarity.py   (TV5)  nhận hash dạng chuỗi '0/1' hoặc mảng 0/1 (KHÔNG nhận hex);
                         build_distance_table(hashes, pairs) -> bảng img1,img2,label,distance,status
  evaluation.py   (TV6)  nhãn 0/1 (1 = tương tự), distance là số nguyên >= 0, dự đoán 1 nếu distance <= ngưỡng,
                         ROC dùng điểm số -distance
  experiments.py  (TV6)  build_dataset / compare_configs: so sánh wavelet, mức phân rã, lượng tử hoá
  common.py       (TV1)  chuyển hash về dạng bit (normalize_hash) để nối TV4 -> TV5 -> TV6

  pairs.csv: cột img1, img2, label (đường dẫn ảnh tương đối so với data/input, hoặc tuyệt đối)
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ đường dẫn
SRC_DIR = Path(__file__).resolve().parent
LAB_DIR = SRC_DIR.parent
ROOT_DIR = LAB_DIR.parent
DATA_DIR = LAB_DIR / "data" / "input"
PAIRS_FILE = LAB_DIR / "data" / "pairs.csv"
OUTPUT_DIR = LAB_DIR / "data" / "output"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
N_BITS = 64
PAIR_COLUMNS = ("img1", "img2", "label")


def setup_path() -> None:
    """Cho phép `import similarity` (từ src) và cả `import lab_4.src.xxx` (từ thư mục gốc)."""
    for p in (SRC_DIR, ROOT_DIR):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


setup_path()

# ------------------------------------------------------------------ module
CORE_MODULES = {
    "wavelet_hash": ["wavelet_hash"],
    "similarity": ["hamming_distance", "build_distance_table", "sweep_thresholds"],
    "evaluation": ["classification_metrics", "evaluate_thresholds", "roc_auc_from_distances"],
    "experiments": ["load_images", "make_demo_images", "build_dataset", "compare_configs"],
}
LIBRARIES = ["numpy", "pandas", "cv2", "pywt", "matplotlib"]


def require(name: str, attrs=()):
    """Import module của nhóm, báo lỗi tiếng Việt dễ hiểu nếu thiếu module hoặc thiếu hàm."""
    try:
        mod = importlib.import_module(name)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Không import được module '{name}' ({type(e).__name__}: {e}). "
            f"Kiểm tra file {name}.py có nằm trong {SRC_DIR} chưa; chạy `python main.py --check`."
        ) from e
    missing = [a for a in attrs if not hasattr(mod, a)]
    if missing:
        raise RuntimeError(
            f"Module '{name}' ({getattr(mod, '__file__', '?')}) thiếu hàm: {', '.join(missing)}. "
            "Có thể đang dùng phiên bản khác của file (trùng tên?) - xem HUONG_DAN_CHAY.md."
        )
    return mod


def module_status() -> pd.DataFrame:
    """Bảng kiểm tra tích hợp: thư viện, module nhóm, hàm cần có."""
    rows = []
    for lib in LIBRARIES:
        try:
            m = importlib.import_module(lib)
            rows.append((lib, "thư viện", True, getattr(m, "__version__", "")))
        except Exception as e:  # noqa: BLE001
            rows.append((lib, "thư viện", False, f"{type(e).__name__}: {e}"))
    for name, attrs in CORE_MODULES.items():
        try:
            m = importlib.import_module(name)
        except Exception as e:  # noqa: BLE001
            rows.append((name, "module nhóm", False, f"{type(e).__name__}: {e}"))
            continue
        missing = [a for a in attrs if not hasattr(m, a)]
        rows.append((name, "module nhóm", not missing,
                     f"thiếu hàm: {', '.join(missing)} | {m.__file__}" if missing else str(m.__file__)))
    return pd.DataFrame(rows, columns=["tên", "loại", "ok", "chi tiết"])


def path_status(data_dir=DATA_DIR, pairs_file=PAIRS_FILE) -> pd.DataFrame:
    data_dir, pairs_file = Path(data_dir), Path(pairs_file)
    n_img = (sum(1 for f in data_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
             if data_dir.is_dir() else 0)
    rows = [("thư mục ảnh", str(data_dir), data_dir.is_dir(), f"{n_img} ảnh"),
            ("pairs.csv", str(pairs_file), pairs_file.is_file(),
             "tuỳ chọn - nếu không có sẽ tự tạo cặp từ thư mục ảnh")]
    return pd.DataFrame(rows, columns=["mục", "đường dẫn", "tồn tại", "ghi chú"])


# ------------------------------------------------------------------ đọc dữ liệu
def read_image(path, max_side=None):
    """Đọc ảnh BGR, hỗ trợ đường dẫn có dấu tiếng Việt. Trả None nếu không đọc được."""
    import cv2
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is not None and max_side:
        scale = max_side / max(img.shape[:2])
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def resolve_image_path(key, data_dir=DATA_DIR) -> Path:
    p = Path(str(key))
    return p if p.is_absolute() else Path(data_dir) / p


def load_pairs_csv(path) -> pd.DataFrame:
    """Đọc pairs.csv và kiểm tra: đủ cột img1,img2,label; label chỉ 0/1."""
    df = pd.read_csv(path, dtype={"img1": str, "img2": str})
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in PAIR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"pairs.csv thiếu cột: {missing} (cần {list(PAIR_COLUMNS)})")
    if df[list(PAIR_COLUMNS)].isna().any().any():
        raise ValueError("pairs.csv có ô trống ở cột img1/img2/label")
    df["label"] = pd.to_numeric(df["label"], errors="raise")
    if not df["label"].isin([0, 1]).all():
        raise ValueError("Cột label chỉ được là 0 (khác nhau) hoặc 1 (tương tự)")
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


# ------------------------------------------------------------------ hash
def normalize_hash(h, n_bits: int = N_BITS, fmt: str = "auto") -> np.ndarray:
    """Đưa hash về mảng uint8 0/1 (để đưa vào similarity.py).

    Chuỗi: fmt='hex' | 'bin' | 'auto'. Với 'auto', chuỗi chỉ gồm 0/1 và dài đúng n_bits được coi là nhị phân,
    còn lại là hex (vd TV4 trả 16 ký tự hex = 64 bit). Mảng/list: phải chỉ chứa 0/1.
    """
    if h is None:
        raise TypeError("Hash là None")
    if isinstance(h, str):
        s = h.strip().lower()
        s = s[2:] if s.startswith("0x") else s
        if not s:
            raise ValueError("Hash rỗng")
        is_bin = set(s) <= {"0", "1"}
        if fmt == "bin" or (fmt == "auto" and is_bin and len(s) == n_bits):
            if not is_bin:
                raise ValueError("Hash nhị phân chỉ chứa 0 và 1")
            return np.frombuffer(s.encode("ascii"), dtype=np.uint8) - ord("0")
        if fmt not in ("hex", "auto"):
            raise ValueError("fmt phải là 'hex', 'bin' hoặc 'auto'")
        try:
            value = int(s, 16)
        except ValueError as e:
            raise ValueError(f"Hash hex không hợp lệ: {h!r}") from e
        return np.frombuffer(format(value, f"0{4 * len(s)}b").encode("ascii"), dtype=np.uint8) - ord("0")
    if isinstance(h, (list, tuple, np.ndarray)):
        a = np.asarray(h)
        if a.ndim != 1 or a.size == 0:
            raise ValueError("Hash mảng phải 1 chiều và không rỗng")
        if not np.isin(a, (0, 1)).all():
            raise ValueError("Hash mảng chỉ được chứa 0 hoặc 1")
        return a.astype(np.uint8)
    raise TypeError(f"Kiểu hash không hỗ trợ: {type(h).__name__}")


def bits_to_hex(bits) -> str:
    bits = np.asarray(bits)
    return format(int("".join(map(str, bits.tolist())), 2), f"0{max(len(bits) // 4, 1)}x")


def get_hash_function():
    """Trả (hàm_hash, tên_nguồn). Ưu tiên wavelet_hash.py của TV4, không có thì dùng experiments.py.

    Hàm trả về nhận ảnh (mảng hoặc đường dẫn) và cho ra mảng bit 0/1.
    """
    try:
        fn = require("wavelet_hash", ["wavelet_hash"]).wavelet_hash
        return (lambda img: normalize_hash(fn(img))), "wavelet_hash.py (TV4)"
    except RuntimeError:
        ex = require("experiments", ["wavelet_hash"])
        return (lambda img: normalize_hash(ex.wavelet_hash(img))), "experiments.py (dự phòng: haar/1 mức/median)"


def compute_hashes(keys, loader, hash_fn):
    """Hash từng ảnh một (đọc -> hash -> bỏ ảnh để tiết kiệm RAM). Lỗi từng ảnh được ghi lại, không dừng."""
    hashes, errors = {}, {}
    for k in keys:
        try:
            hashes[k] = hash_fn(loader(k))
        except Exception as e:  # noqa: BLE001
            errors[k] = f"{type(e).__name__}: {e}"
    return hashes, errors


# ------------------------------------------------------------------ ngưỡng & biểu đồ
def choose_threshold(labels, distances, n_bits: int = N_BITS):
    """Ngưỡng tối đa Youden J = recall + specificity - 1 (hoà -> ngưỡng nhỏ hơn). Dùng evaluation.py."""
    ev = require("evaluation", ["evaluate_thresholds"])
    rows = ev.evaluate_thresholds([int(v) for v in labels], [int(v) for v in distances],
                                  list(range(n_bits + 1)))
    tab = pd.DataFrame(rows)
    tab["youden"] = tab["recall"] + tab["specificity"] - 1
    return int(tab.loc[tab["youden"].idxmax(), "threshold"]), tab


def save_roc_plot(roc: dict, path, point=None, title="ROC - Wavelet Hash (điểm số = -khoảng cách)"):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(roc["fpr"], roc["tpr"], drawstyle="steps-post", label=f"AUC = {roc['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Ngẫu nhiên")
    if point:
        ax.scatter([point[0]], [point[1]], c="red", zorder=3, label=f"Ngưỡng {point[2]}")
    ax.set_xlabel("FPR = 1 - Specificity"); ax.set_ylabel("TPR = Recall")
    ax.set_title(title); ax.legend(loc="lower right"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)