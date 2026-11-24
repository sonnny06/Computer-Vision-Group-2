from pathlib import Path

import yaml

# Thư mục gốc dự án (Course_work): file này nằm ở src/utils/ nên lùi lên 2 cấp
ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_CONFIG = ROOT / "configs" / "model_config.yaml"

# Các khóa bắt buộc (code của Thái truy cập thẳng các khóa này)
_REQUIRED = {
    "model": ["backend", "id", "device", "local_files_only"],
    "inference": ["max_new_tokens", "max_image_side", "max_upload_mb",
                  "max_question_chars"],
}


def load_config(path=None) -> dict:
    """Đọc file YAML cấu hình model và kiểm tra các khóa bắt buộc.

    Args:
        path: đường dẫn file YAML. Bỏ trống sẽ dùng configs/model_config.yaml.
    """
    path = Path(path) if path else DEFAULT_MODEL_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    for section, keys in _REQUIRED.items():
        if section not in cfg:
            raise ValueError(f"Thiếu mục '{section}' trong {path.name}")
        missing = [k for k in keys if k not in cfg[section]]
        if missing:
            raise ValueError(
                f"Thiếu khóa {missing} trong mục '{section}' của {path.name}"
            )
    cfg["model"].setdefault("load_in_4bit", False)
    return cfg


if __name__ == "__main__":
    # Chạy thử độc lập: python src/utils/config.py
    cfg = load_config()
    print("ROOT :", ROOT)
    print("Model:", cfg["model"]["id"], "| device:", cfg["model"]["device"])
    print("Ảnh tối đa:", cfg["inference"]["max_image_side"], "px")