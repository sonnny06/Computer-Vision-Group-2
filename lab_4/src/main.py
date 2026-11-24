"""
main.py - TV1: chạy toàn bộ luồng xử lý

    ảnh  ->  wavelet hash (TV4)  ->  khoảng cách Hamming (TV5)  ->  dự đoán theo ngưỡng
         ->  chỉ số + ROC/AUC (TV6)  ->  (tuỳ chọn) so sánh cấu hình (TV6)

Chạy từ thư mục lab_4/src:
    python main.py --check                       # kiểm tra import + đường dẫn
    python main.py --demo                        # chạy thử bằng ảnh giả lập
    python main.py                               # dùng data/pairs.csv (nếu có) hoặc tự tạo cặp từ data/input
    python main.py --compare                     # thêm bước so sánh wavelet / mức phân rã / lượng tử hoá
    python main.py --match a.jpg b.jpg           # so khớp nhanh 2 ảnh
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common


# ------------------------------------------------------------------ dữ liệu
def load_dataset(data_dir, pairs_file, demo=False, seed=0, max_side=None):
    """Trả dict: pairs (img1,img2,label[,variant]), loader(key)->ảnh BGR, images (hoặc None), source."""
    if demo:
        ex = common.require("experiments", ["make_demo_images", "build_dataset"])
        images, pairs = ex.build_dataset(ex.make_demo_images(), seed=seed)
        return dict(pairs=pairs, loader=images.__getitem__, images=images, source="demo (ảnh giả lập)")

    if pairs_file and Path(pairs_file).is_file():
        pairs = common.load_pairs_csv(pairs_file)
        loader = lambda key: common.read_image(common.resolve_image_path(key, data_dir), max_side)  # noqa: E731
        return dict(pairs=pairs, loader=loader, images=None, source=f"pairs.csv: {pairs_file}")

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Không thấy thư mục ảnh: {data_dir}. Dùng --data <thư mục> hoặc --demo.")
    ex = common.require("experiments", ["load_images", "build_dataset"])
    base = ex.load_images(str(data_dir))
    if len(base) < 2:
        raise ValueError(f"Cần >= 2 ảnh trong {data_dir} (đang có {len(base)}). Hoặc dùng --demo.")
    images, pairs = ex.build_dataset(base, seed=seed)
    return dict(pairs=pairs, loader=images.__getitem__, images=images,
                source=f"tự tạo cặp từ {len(base)} ảnh trong {data_dir}")


# ------------------------------------------------------------------ pipeline
def run_pipeline(data_dir=common.DATA_DIR, pairs_file=common.PAIRS_FILE, out_dir=common.OUTPUT_DIR,
                 threshold=None, compare=False, demo=False, max_side=None, seed=0, verbose=True):
    """Chạy toàn bộ luồng, ghi file vào out_dir và trả về dict kết quả."""
    log = print if verbose else (lambda *a, **k: None)
    sim = common.require("similarity", ["build_distance_table", "sweep_thresholds", "predict_similar"])
    ev = common.require("evaluation", ["classification_metrics", "evaluate_thresholds", "roc_auc_from_distances"])
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1. dữ liệu
    ds = load_dataset(data_dir, pairs_file, demo, seed, max_side)
    pairs = ds["pairs"]
    log(f"[1/5] Dữ liệu: {ds['source']} | {len(pairs)} cặp "
        f"({int((pairs.label == 1).sum())} tương tự / {int((pairs.label == 0).sum())} khác nhau)")

    # 2. hash (TV4)
    hash_fn, hash_source = common.get_hash_function()
    keys = sorted(set(pairs.img1) | set(pairs.img2))
    hashes, hash_errors = common.compute_hashes(keys, ds["loader"], hash_fn)
    log(f"[2/5] Hash: {len(hashes)}/{len(keys)} ảnh OK | nguồn hash: {hash_source}")
    if hash_errors:
        log(f"      ⚠ {len(hash_errors)} ảnh lỗi, ví dụ: {next(iter(hash_errors.items()))}")

    # 3. khoảng cách Hamming (TV5)
    table = sim.build_distance_table(hashes, pairs[list(common.PAIR_COLUMNS)])
    if "variant" in pairs.columns:
        table["variant"] = pairs["variant"].values
    ok = table[table.status == "ok"]
    status_counts = table.status.value_counts().to_dict()
    log(f"[3/5] Khoảng cách: {len(ok)}/{len(table)} cặp hợp lệ | trạng thái: {status_counts}")
    if ok.label.nunique() < 2:
        raise ValueError("Sau khi lọc cặp lỗi, dữ liệu chỉ còn một lớp nhãn -> không tính được ROC/AUC.")
    n_bits = int(ok.hash_len.max())

    # 4. ngưỡng + dự đoán (<=, khớp evaluation.py)
    y = ok.label.astype(int).tolist()
    d = ok.distance.astype(int).tolist()
    thr_table = None
    if threshold is None:
        threshold, thr_table = common.choose_threshold(y, d, n_bits)
        log(f"[4/5] Ngưỡng tự chọn (Youden J): {threshold}")
    else:
        log(f"[4/5] Ngưỡng do người dùng đặt: {threshold}")
    sweep = sorted(set(range(0, n_bits // 2 + 1, 2)) | {int(threshold)})
    pred_table = sim.sweep_thresholds(table, sweep, inclusive=True)
    pred_table["prediction"] = np.nan
    mask = pred_table.status == "ok"
    pred_table.loc[mask, "prediction"] = sim.predict_similar(
        pred_table.loc[mask, "distance"].to_numpy(), threshold, inclusive=True)

    # 5. đánh giá (TV6)
    pred = pred_table.loc[mask, "prediction"].astype(int).tolist()
    metrics = ev.classification_metrics(y, pred)
    roc = ev.roc_auc_from_distances(y, d)
    sweep_metrics = pd.DataFrame(ev.evaluate_thresholds(y, d, list(range(n_bits + 1))))
    fpr_pt = metrics["FP"] / (metrics["FP"] + metrics["TN"]) if (metrics["FP"] + metrics["TN"]) else 0.0
    log(f"[5/5] Đánh giá tại ngưỡng {threshold}: "
        f"Acc={metrics['accuracy']:.3f} Recall={metrics['recall']:.3f} "
        f"Spec={metrics['specificity']:.3f} Prec={metrics['precision']:.3f} | AUC={roc['auc']:.3f}")

    # ghi kết quả
    pred_table.to_csv(out_dir / "distance_predictions.csv", index=False)
    sweep_metrics.to_csv(out_dir / "threshold_metrics.csv", index=False)
    pairs.to_csv(out_dir / "pairs_used.csv", index=False)
    with open(out_dir / "hashes.json", "w", encoding="utf-8") as f:
        json.dump({k: "".join(map(str, v.tolist())) for k, v in hashes.items()}, f)
    common.save_roc_plot(roc, out_dir / "roc.png", point=(fpr_pt, metrics["recall"], threshold))

    summary = dict(source=ds["source"], hash_source=hash_source, n_images=len(keys),
                   n_hash_ok=len(hashes), n_pairs=len(table), n_pairs_ok=len(ok),
                   status_counts=status_counts, n_bits=n_bits, threshold=int(threshold),
                   comparison="distance <= threshold", auc=float(roc["auc"]),
                   metrics={k: (None if isinstance(v, float) and np.isnan(v) else v)
                            for k, v in metrics.items()})
    result = dict(summary=summary, table=pred_table, metrics=metrics, roc=roc, hashes=hashes,
                  hash_errors=hash_errors, sweep_metrics=sweep_metrics, threshold=int(threshold),
                  pairs=pairs, images=ds["images"], loader=ds["loader"], comparison=None)

    # 6. (tuỳ chọn) so sánh cấu hình
    if compare:
        result["comparison"] = run_comparison(result, out_dir, log, seed)
        if result["comparison"] is not None:
            summary["best_config"] = result["comparison"].iloc[0]["config"]
            summary["best_config_auc"] = float(result["comparison"].iloc[0]["auc"])

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    log(f"Đã ghi kết quả vào: {out_dir}")
    return result


def run_comparison(result, out_dir, log=print, seed=0, **grid):
    """So sánh wavelet/mức/lượng tử hoá bằng experiments.compare_configs (TV6). Trả DataFrame hoặc None."""
    try:
        ex = common.require("experiments", ["compare_configs"])
    except RuntimeError as e:
        log(f"[so sánh] BỎ QUA: {e}")
        return None
    pairs, images = result["pairs"], result["images"]
    if images is None:   # chế độ pairs.csv: nạp ảnh (thu nhỏ 512 px) chỉ cho bước so sánh
        import cv2
        images = {}
        for k in sorted(set(pairs.img1) | set(pairs.img2)):
            img = result["loader"](k)
            if img is not None:
                s = min(1.0, 512 / max(img.shape[:2]))
                images[k] = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        keep = pairs.img1.isin(images) & pairs.img2.isin(images)
        pairs = pairs[keep].reset_index(drop=True)
    summary, _ = ex.compare_configs(pairs, images, **grid)
    summary.to_csv(Path(out_dir) / "experiments_summary.csv", index=False)
    log(f"[so sánh] {len(summary)} cấu hình | tốt nhất: {summary.iloc[0]['config']} "
        f"(AUC={summary.iloc[0]['auc']:.3f})")
    return summary


# ------------------------------------------------------------------ so khớp nhanh 2 ảnh
def match_two(path1, path2, threshold=None, out_dir=common.OUTPUT_DIR):
    sim = common.require("similarity", ["hamming_distance"])
    hash_fn, source = common.get_hash_function()
    h1, h2 = (hash_fn(common.read_image(p)) for p in (path1, path2))
    dist = sim.hamming_distance(h1, h2)
    if threshold is None:
        sj = Path(out_dir) / "summary.json"
        threshold = json.loads(sj.read_text(encoding="utf-8"))["threshold"] if sj.is_file() else 10
        note = "ngưỡng từ lần chạy trước" if sj.is_file() else "ngưỡng tạm 10 (chưa chạy pipeline)"
    else:
        note = "ngưỡng do người dùng đặt"
    similar = dist <= threshold
    print(f"{common.bits_to_hex(h1)} vs {common.bits_to_hex(h2)} | Hamming = {dist}/{len(h1)} | "
          f"ngưỡng {threshold} ({note}) -> {'TƯƠNG TỰ' if similar else 'KHÁC NHAU'}")
    return dict(distance=dist, threshold=threshold, similar=bool(similar))


# ------------------------------------------------------------------ CLI
def check(data_dir, pairs_file) -> bool:
    st = common.module_status()
    print(st.to_string(index=False))
    print()
    print(common.path_status(data_dir, pairs_file).to_string(index=False))
    core = st[st["tên"].isin(["similarity", "evaluation", "numpy", "pandas", "cv2", "pywt", "matplotlib"])]
    ok = bool(core.ok.all())
    print("\nKẾT QUẢ:", "SẴN SÀNG CHẠY" if ok else "CÒN LỖI - xem cột 'chi tiết' ở trên")
    if ok and not bool(st.set_index("tên").loc["experiments", "ok"]):
        print("Cảnh báo: experiments.py lỗi/thiếu hàm -> chỉ chạy được với pairs.csv, "
              "KHÔNG dùng được --demo, tự tạo cặp và --compare.")
    if not st.set_index("tên").loc["wavelet_hash", "ok"]:
        print("Lưu ý: chưa có wavelet_hash.py của TV4 -> main.py dùng hàm dự phòng trong experiments.py.")
    return ok


def build_parser():
    ap = argparse.ArgumentParser(description="Pipeline so sánh ảnh bằng Wavelet Hash")
    ap.add_argument("--data", default=str(common.DATA_DIR), help="thư mục ảnh gốc")
    ap.add_argument("--pairs", default=str(common.PAIRS_FILE), help="pairs.csv (img1,img2,label)")
    ap.add_argument("--out", default=str(common.OUTPUT_DIR), help="thư mục kết quả")
    ap.add_argument("--threshold", type=int, default=None, help="ngưỡng Hamming (mặc định: tự chọn)")
    ap.add_argument("--compare", action="store_true", help="so sánh các cấu hình wavelet")
    ap.add_argument("--demo", action="store_true", help="dùng ảnh giả lập")
    ap.add_argument("--max-side", type=int, default=None, help="thu nhỏ ảnh khi đọc từ pairs.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="kiểm tra import/đường dẫn rồi thoát")
    ap.add_argument("--match", nargs=2, metavar=("ANH1", "ANH2"), help="so khớp nhanh hai ảnh")
    ap.add_argument("--debug", action="store_true", help="hiện traceback đầy đủ")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.check:
            return 0 if check(args.data, args.pairs) else 1
        if args.match:
            match_two(*args.match, threshold=args.threshold, out_dir=args.out)
            return 0
        run_pipeline(args.data, args.pairs, args.out, args.threshold, args.compare,
                     args.demo, args.max_side, args.seed)
        return 0
    except Exception as e:  # noqa: BLE001
        if args.debug:
            raise
        print(f"\nLỖI ({type(e).__name__}): {e}\n(Chạy thêm --debug để xem chi tiết, hoặc --check để kiểm tra tích hợp.)",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())