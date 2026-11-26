"""Read image metadata and VQA prompts for the local interfaces."""
import csv
from pathlib import Path


def load_ui_dataset(data_dir: Path):
    with (data_dir / "metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
        captions = list(csv.DictReader(handle))
    with (data_dir / "questions.csv").open(encoding="utf-8-sig", newline="") as handle:
        questions = list(csv.DictReader(handle))

    caption_by_name = {row["filename"]: row for row in captions}
    images = {}
    for question in questions:
        name = question["image_name"]
        if name in images:
            continue
        meta = caption_by_name.get(name, {})
        images[name] = {
            **meta,
            "filename": name,
            "image_id": meta.get("image_id") or Path(name).stem,
            "body_part": question["region_group"] or question["body_part"],
            "view_position": question["modality"],
            "source": question["source"],
            "finding_category": meta.get("finding_category", ""),
            "finding_description": meta.get("finding_description", ""),
            "ground_truth_caption_vi": meta.get("ground_truth_caption_vi", ""),
        }
    return list(images.values()), questions, captions
