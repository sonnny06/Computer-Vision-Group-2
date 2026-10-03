# Computer-Vision-Group-2
# Hướng Dẫn Quy Trình Làm Việc Nhóm (Git Workflow Standard)

Tài liệu này chuẩn hóa quy trình làm việc nhóm cho tất cả các bài thực hành/Lab môn Xử lý ảnh. Mỗi thành viên có trách nhiệm tuân thủ các bước dưới đây để tránh xung đột mã nguồn (merge conflict) và duy trì chất lượng bài làm.

---

## 🧭 1. Nguyên Tắc Cốt Lõi

1. **Nhánh `main` là nhánh nguồn ổn định:** Tuyệt đối **không** code trực tiếp hoặc `git push` thẳng lên `main`.
2. **Mỗi chức năng/bài tập là một Branch:** Mọi thao tác viết code đều thực hiện trên branch tính năng (feature branch) được tách từ `main`.
3. **Luôn đồng bộ trước khi làm:** Trước khi bắt đầu code mỗi buổi, luôn pull bản mới nhất từ `main` về.
4. **Gộp code qua Pull Request (PR):** Tất cả các bài làm chỉ được gộp vào `main` sau khi tạo PR và được nhóm review.

---

## 🗂️ 2. Quy Ước Cấu Trúc Thư Mục Chuẩn

Mỗi bài lab mới sẽ được tạo trong một thư mục riêng biệt:

```text
image-processing-labs/
│
├── lab01/                      # Bài thực hành chương/lab hiện tại
│   ├── data/                   # Ảnh đầu vào dùng chung
│   ├── results/                # Ảnh đầu ra sinh ra từ code
│   ├── task1_*.py
│   └── task2_*.py
├── lab02/                      # Bài thực hành tiếp theo
├── requirements.txt            # Thư viện toàn dự án
├── .gitignore                  # Bỏ qua kết quả chạy tạm thời, venv, cache
└── README.md
```

---

## 🌿 3. Quy Ước Đặt Tên Nhánh (Branch Naming)

Đặt tên nhánh theo cấu trúc:  
`lab<số_lab>/<mã_bài_tập>-<tên_thành_viên>` hoặc `feature/lab<số_lab>-<mô_tả_ngắn>`

*Ví dụ:*
- `lab01/task2-nam`
- `lab01/task3-hoa`
- `lab02/edge-detection-an`

---

## 🚀 4. Quy Trình Làm Việc Chi Tiết (Từng Bước)

### Bước 1: Thiết lập ban đầu (Chỉ làm 1 lần đầu tiên)
```bash
# Clone repository về máy
git clone <URL_REPOSITORY_CỦA_NHÓM>
cd <TEN_THU_MUC_DU_AN>

# Khởi tạo môi trường ảo (Khuyên dùng)
python -m venv venv

# Kích hoạt môi trường:
# - Trên Windows:
.\venv\Scripts\activate
# - Trên macOS/Linux:
source venv/bin/activate

# Cài đặt thư viện yêu cầu
pip install -r requirements.txt
```

---

### Bước 2: Bắt đầu làm bài tập mới
Mỗi khi bắt đầu nhận một bài thực hành hoặc bài tập con:

```bash
# 1. Quay về main và cập nhật bản mới nhất
git checkout main
git pull origin main

# 2. Tạo và chuyển sang branch làm việc của bạn
git checkout -b <ten_branch_cua_ban>
# Ví dụ: git checkout -b lab01/task2-nam
```

---

### Bước 3: Lập trình và Lưu vết (Commit)
Trong quá trình code, hãy commit thường xuyên sau mỗi mục tiêu nhỏ hoàn thành (không dồn toàn bộ bài vào 1 commit lớn):

```bash
# Kiểm tra file đã chỉnh sửa
git status

# Thêm file vào danh sách commit
git add lab01/task2_io.py

# Commit với thông điệp rõ ràng, mang tính mô tả
git commit -m "Lab01: Hoàn thành chức năng đọc và hiển thị ảnh"
```

---

### Bước 4: Đẩy nhánh lên Remote
Khi đã hoàn tất hoặc muốn lưu code lên GitHub/GitLab:

```bash
# Đẩy branch lên remote (chỉ cần thêm -u ở lần push đầu tiên của branch)
git push -u origin <ten_branch_cua_ban>
```

---

### Bước 5: Cập nhật thay đổi từ `main` (Tránh xung đột trước khi nộp)
Nếu trong khi bạn đang code, các bạn khác đã gộp bài vào `main`, bạn cần cập nhật nhánh của mình trước:

```bash
# Đảm bảo bạn đang ở branch của mình
git checkout <ten_branch_cua_ban>

# Kéo cập nhật từ main vào branch hiện tại
git pull origin main
```
*Nếu xảy ra Conflict:* Mở file báo lỗi (trong VS Code), chọn phương án xử lý code (Accept Current / Incoming / Both), sau đó `git add .` và `git commit -m "Resolve merge conflict"`.

---

### Bước 6: Tạo Pull Request (PR) & Gộp Code
1. Truy cập vào GitHub/GitLab của nhóm.
2. Bạn sẽ thấy thông báo nhánh của bạn vừa được push lên -> Chọn **Compare & pull request**.
3. Điền thông tin:
   - **Tiêu đề:** `[Lab X] Hoàn thành <Tên bài tập> - <Tên thành viên>`
   - **Nội dung:** Tóm tắt ngắn gọn các chức năng đã test, hình ảnh demo nếu có.
4. Gán ít nhất **1 thành viên khác** vào mục **Reviewers**.
5. Sau khi reviewer xác nhận code chạy đúng, nhấn **Merge pull request** để tích hợp vào `main`.

---

## ⚠️️ 5. Những Điều Cần Tránh (Do's & Don'ts)

| NÊN LÀM ✅ | KHÔNG NÊN LÀM ❌ |
| :--- | :--- |
| Commit file mã nguồn (`.py`, `.ipynb`) | Commit thư mục ảo `venv/`, file `.pyc`, cache |
| Giữ ảnh input nhẹ trong `data/` | Commit hàng loạt ảnh kết quả dung lượng nặng |
| Pull code mới từ `main` mỗi ngày | Code liên tục nhiều ngày trên branch cũ không đồng bộ |
| Viết commit message bằng tiếng Việt/Anh rõ nghĩa | Commit với nội dung: `fix`, `a`, `update`, `123` |
