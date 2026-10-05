# Issue #10 — Grayscale: bàn giao triển khai

Issue: https://github.com/sonnny06/Computer-Vision-Group-2/issues/10

Ngày kiểm tra: 05/10/2026. Branch: `feature/lab01-grayscale`.

## Kết quả

Đã triển khai chuyển BGR sang grayscale và notebook tiếng Việt có kết quả thực thi.
Ảnh minh họa `(512, 512, 3)` chuyển thành `(512, 512)`, giữ `uint8`;
min/max grayscale quan sát được là `0/255`. Các ảnh khác không nhất thiết chạm hai mức này.

| Thành phần | File |
| --- | --- |
| Hàm chuyển đổi và kiểm tra đầu vào | `lab_1/src/color_space.py` |
| Notebook có bảng thống kê và ảnh so sánh | `lab_1/notebook/03_grayscale.ipynb` |
| PNG grayscale không mất mát | `lab_1/data/output/02_portrait_astronaut_gray.png` |
| Dependencies chạy bài thực hành | `requirements.txt` |
| Quy tắc bỏ qua môi trường/cache/QA tạm | `.gitignore` |
| Profile và quy trình phối hợp team | `docs/subagents/` |

`bgr_to_grayscale(image_bgr)` nhận `numpy.ndarray uint8`, BGR `(H,W,3)`,
không rỗng; trả `(H,W)` cùng dtype, không sửa input. Sai kiểu/dtype phát sinh
`TypeError`; shape sai/ảnh rỗng phát sinh `ValueError`.

## Điều phối team

1. Lead kiểm tra hiện trạng và chốt giao diện trước khi phát triển.
2. Image developer viết source; notebook developer viết notebook theo giao diện.
3. QA chuẩn bị dependencies trong `.venv` rồi nhận độc quyền ghi notebook sau khi
   developer bàn giao; thực thi và lưu outputs.
4. Điều phối kiểm tra source/notebook, evidence và ảnh preview độc lập; Lead review cuối.

Runtime có ba thread con: thread Lead được tái sử dụng cho pha QA rồi trở lại
review; bốn vai trò không chạy đồng thời. Profile là instructions giao việc,
không được cài vào `.codex/agents/` trong lần triển khai này. Model và reasoning
của các thread thực tế kế thừa phiên điều phối.

## Kiểm tra thực tế

| Kiểm tra | Kết quả |
| --- | --- |
| Notebook đúng định dạng nbformat | Đạt |
| Kernel mới, cwd repository root | 5 code cells, 0 lỗi |
| Kernel mới, cwd `lab_1/notebook` | 5 code cells, 0 lỗi |
| Pixel BGR xanh dương/xanh lá/đỏ/đen/trắng | 29/150/76/0/255 |
| Đầu vào sai kiểu, dtype, shape hoặc rỗng | 6 trường hợp báo đúng loại lỗi |
| Input không bị sửa | Đạt với pixel mẫu và ảnh minh họa |
| PNG đọc lại khớp mảng grayscale từng pixel | Đạt |
| Ảnh so sánh lưu trong notebook | Đã xem: đúng màu, tiêu đề rõ, không bị cắt |
| Dependency consistency | `pip check`: không có dependency lỗi |

Môi trường đã kiểm tra: Python 3.11.9; NumPy 2.4.6; opencv-python 5.0.0.93;
Matplotlib 3.11.2; Jupyter 1.1.1; nbformat 5.11.1; nbclient 0.11.0; ipykernel 7.4.0.
Dependencies không pin; các phiên bản trên ghi nhận môi trường thực thi, không phải
cam kết kết quả kiểm tra cho mọi phiên bản tương lai.

Evidence chi tiết và script QA nằm trong `.qa/` ở workspace, được ignore.
Notebook được bàn giao có execution counts 1–5 và outputs; PNG được bàn giao riêng.

## Chạy lại

Từ repository root trên Windows:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m jupyter notebook
```

Mở `lab_1/notebook/03_grayscale.ipynb`, chọn kernel dùng Python của `.venv`,
rồi **Restart Kernel → Run All**. Nếu `.venv` đã tồn tại, dùng lại môi trường đó.
Notebook cũng hỗ trợ cwd `lab_1/notebook`.

## Review GitHub

Review nội bộ không thay thế approval của reviewer GitHub. Issue chỉ được coi là
Done sau khi PR được review và merge. PR phải dùng `Closes #10`, vì số #3 trong
tiêu đề issue không phải số issue GitHub thực tế.
