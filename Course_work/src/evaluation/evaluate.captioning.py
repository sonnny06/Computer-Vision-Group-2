"""Chạy đánh giá Captioning (forwarder sang evaluate_captioning.py)."""
import sys
from pathlib import Path

# Đảm bảo import được src
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.evaluate_captioning import main

if __name__ == "__main__":
    main()
