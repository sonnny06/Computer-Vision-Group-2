"""brightness_contrast.py
Point operations: brightness and contrast on images in a data folder.

Run from the project root (the folder containing data/):
    python brightness_contrast.py

Or specify another data folder and image names:
    python brightness_contrast.py --data-dir data --images 02_chelsia.png 03_coffee.png 09_moon.png

Brightness: g = clip(f + beta, 0, 255)
Contrast:   g = clip(alpha * (f - 127.5) + 127.5, 0, 255)
"""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

DEFAULT_IMAGES = ["02_chelsia.png", "03_coffee.png", "09_moon.png"]


def adjust_brightness(image, beta):
    """Add beta to every channel; clip values to the valid 8-bit range."""
    image = np.asarray(image, dtype=np.float32)
    return np.clip(image + beta, 0, 255).astype(np.uint8)


def adjust_contrast(image, alpha):
    """Scale contrast around middle gray (127.5). alpha=1 leaves the image unchanged."""
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    image = np.asarray(image, dtype=np.float32)
    return np.clip((image - 127.5) * alpha + 127.5, 0, 255).astype(np.uint8)


def save_comparison(original, outputs, target):
    """Save original and transformed images side-by-side."""
    items = [("Ảnh gốc", np.asarray(original))] + outputs
    cols = 3
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, (title, arr) in zip(axes, items):
        ax.imshow(arr)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_image(image_path, output_dir):
    original_image = Image.open(image_path).convert("RGB")
    original = np.asarray(original_image)
    stem = image_path.stem
    image_out = output_dir / stem
    image_out.mkdir(parents=True, exist_ok=True)

    settings = [
        ("Tăng sáng +40", "brightness_plus_40.png", adjust_brightness(original, 40)),
        ("Tăng sáng +80", "brightness_plus_80.png", adjust_brightness(original, 80)),
        ("Giảm sáng -40", "brightness_minus_40.png", adjust_brightness(original, -40)),
        ("Giảm sáng -80", "brightness_minus_80.png", adjust_brightness(original, -80)),
        ("Giảm tương phản α=0.5", "contrast_alpha_0.5.png", adjust_contrast(original, 0.5)),
        ("Giữ nguyên α=1.0", "contrast_alpha_1.0.png", adjust_contrast(original, 1.0)),
        ("Tăng tương phản α=1.5", "contrast_alpha_1.5.png", adjust_contrast(original, 1.5)),
        ("Tăng tương phản α=2.0", "contrast_alpha_2.0.png", adjust_contrast(original, 2.0)),
    ]
    outputs = []
    for title, filename, result in settings:
        Image.fromarray(result).save(image_out / filename)
        outputs.append((title, result))
    save_comparison(original, outputs, image_out / "comparison_grid.png")
    print(f"{image_path.name}: saved results to {image_out.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Brightness and contrast point operations")
    parser.add_argument("--data-dir", default="data", help="Folder containing source images (default: data)")
    parser.add_argument("--out", default="brightness_contrast_results", help="Folder for generated results")
    parser.add_argument("--images", nargs="+", default=DEFAULT_IMAGES,
                        help="Image filenames to process; defaults to cat, coffee, and moon images")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.out)
    if not data_dir.is_dir():
        raise SystemExit(f"Không tìm thấy thư mục dữ liệu: {data_dir.resolve()}\n"
                         "Hãy chạy lệnh từ thư mục dự án có thư mục data/ hoặc dùng --data-dir.")

    missing = [name for name in args.images if not (data_dir / name).is_file()]
    if missing:
        available = sorted(p.name for p in data_dir.iterdir() if p.is_file())
        raise SystemExit("Không tìm thấy các ảnh sau trong thư mục data: " + ", ".join(missing) +
                         "\nHãy kiểm tra chính xác tên file. Một số file hiện có: " +
                         (", ".join(available[:30]) if available else "(thư mục rỗng)"))

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.images:
        process_image(data_dir / name, output_dir)
    print(f"\nHoàn tất. Ảnh kết quả nằm trong: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
