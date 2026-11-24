"""Khởi chạy nhanh giao diện Web Medical-VQA (tự mở trình duyệt)."""

import threading
import time
import webbrowser

from web.server import run_server


def _open_browser() -> None:
    time.sleep(0.6)
    webbrowser.open("http://127.0.0.1:8000")


if __name__ == "__main__":
    threading.Thread(target=_open_browser, daemon=True).start()
    run_server(host="127.0.0.1", port=8000)
