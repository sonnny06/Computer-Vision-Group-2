"""Run one caption per image in the frozen handoff data."""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

COURSE = Path(__file__).resolve().parent
RESULTS = COURSE.parent / "results"
IMAGES = COURSE / "data/evaluation/images"
REFERENCES = RESULTS / "captioning_expanded_41e1c65_references.csv"
MANIFEST = RESULTS / "captioning_expanded_41e1c65_freeze.json"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_inputs():
    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if sha256(REFERENCES) != frozen["references_sha256"]:
        raise ValueError("SHA-256 của CSV không khớp biên bản khóa.")
    with REFERENCES.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["status"] == "selected"]
    if len(rows) != frozen["images"] or len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Số ảnh hoặc ID không khớp biên bản khóa.")
    for row in rows:
        candidate = (IMAGES / row["filename"]).resolve()
        if candidate.parent != IMAGES.resolve() or sha256(candidate) != row["sha256"]:
            raise ValueError(f"Ảnh không khớp tham chiếu: {row['image_id']}")
    return rows, frozen


def run(rows, frozen, output):
    from huggingface_hub import try_to_load_from_cache
    import torch
    from src.engine.pipeline import build_pipeline

    if output.exists():
        raise FileExistsError(output)
    if os.environ.get("HF_HUB_OFFLINE") != "1" or not torch.cuda.is_available():
        raise RuntimeError("Lượt tái lập cần HF_HUB_OFFLINE=1 và CUDA khả dụng.")
    config = try_to_load_from_cache(MODEL_ID, "config.json", revision="main")
    if not isinstance(config, str) or Path(config).parent.name != frozen["model_revision"]:
        raise RuntimeError("Cache Qwen chưa trỏ đúng revision trong biên bản khóa.")
    start = time.perf_counter()
    pipeline = build_pipeline(use_mock=False)
    print(f"Khởi tạo {time.perf_counter() - start:.3f} giây | {pipeline.vlm.device} | {pipeline.vlm.dtype}")
    fields = ("image_id", "filename", "image_sha256", "references_sha256", "modality",
              "region_group", "pilot_overlap", "raw_caption", "success", "error_code", "error", "time_s")
    with output.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            start = time.perf_counter()
            result = pipeline.run(str(IMAGES / row["filename"]), task="caption")
            seconds = time.perf_counter() - start
            writer.writerow({
                **{key: row[key] for key in ("image_id", "filename", "modality", "region_group", "pilot_overlap")},
                "image_sha256": row["sha256"], "references_sha256": frozen["references_sha256"],
                "raw_caption": result.answer or "", "success": result.success,
                "error_code": result.error_code or "", "error": result.error or "", "time_s": seconds,
            })
            stream.flush()
            print(f"{row['image_id']} | {result.success} | {seconds:.3f} giây")


def main():
    parser = argparse.ArgumentParser(description="Tái lập Captioning trên 35 ảnh đã khóa")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-inputs", action="store_true", help="Kiểm tra CSV và ảnh bằng SHA-256")
    mode.add_argument("--output", type=Path, help="Ghi vào một CSV mới")
    args = parser.parse_args()
    rows, frozen = verified_inputs()
    print(f"Đã kiểm tra {len(rows)} ảnh và CSV tham chiếu.")
    if args.output is not None:
        run(rows, frozen, args.output)


if __name__ == "__main__":
    main()
