from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "prompt_config.yaml"


def load_prompt_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Đọc file cấu hình prompt YAML."""
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình prompt: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_caption_prompt(language: str | None = None, config_path: Path = DEFAULT_CONFIG_PATH) -> str:
    """Tạo prompt chuẩn cho tác vụ Image Captioning."""
    cfg = load_prompt_config(config_path)
    lang = language or cfg.get("language", "vi")
    templates = cfg.get("caption", {})
    if lang not in templates:
        raise ValueError(f"Ngôn ngữ không được hỗ trợ cho caption: {lang}")
    return templates[lang].strip()


def get_vqa_prompt(
    question: str,
    language: str | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> str:
    """Tạo prompt chuẩn cho tác vụ Medical VQA kèm kiểm tra đầu vào."""
    cfg = load_prompt_config(config_path)
    max_chars = int(cfg.get("max_question_chars", 1500))

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Vui lòng nhập câu hỏi hợp lệ.")
    clean_question = question.strip()
    if len(clean_question) > max_chars:
        raise ValueError(f"Câu hỏi quá dài (tối đa {max_chars} ký tự).")

    lang = language or cfg.get("language", "vi")
    templates = cfg.get("vqa", {})
    if lang not in templates:
        raise ValueError(f"Ngôn ngữ không được hỗ trợ cho VQA: {lang}")
    return templates[lang].strip().format(question=clean_question)


if __name__ == "__main__":
    # Chạy test nhanh độc lập: python src/utils/prompt.py
    print("=== CAPTION PROMPT ===")
    print(get_caption_prompt())
    print("\n=== VQA PROMPT ===")
    print(get_vqa_prompt("Ảnh này chụp bộ phận nào và có thấy hai lá phổi rõ không?"))