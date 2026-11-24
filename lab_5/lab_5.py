"""
Lab 5 - Nhận diện KHUÔN MẶT THEO NGƯỜI realtime bằng webcam (MTCNN + FaceNet)

Cấu trúc dữ liệu (mỗi người một thư mục, tạo bằng collect_dataset.py):
    data/reference/<tên>/*.jpg                 ảnh tham chiếu (tùy chọn)
    data/collected/<tên>/<điều kiện>/*.jpg     ảnh thu thập

Cách hoạt động:
  1. MTCNN phát hiện khuôn mặt, FaceNet (InceptionResnetV1, vggface2) tạo embedding 512 chiều.
  2. Mỗi người có một "gallery" = embedding của mọi ảnh của người đó.
     Chụp thêm ảnh hoặc thêm người mới rồi nhấn R (hoặc chạy lại) là gallery tự cập nhật.
     Embedding được cache ở results/gallery_cache.npz nên chỉ ảnh mới mới phải tính lại.
  3. Mỗi khuôn mặt trên webcam được tính điểm với từng người bằng Cosine Similarity
     (điểm = trung bình TOP_K ảnh giống nhất của người đó).
  4. Người có điểm cao nhất được chọn nếu điểm >= THRESHOLD và hơn người thứ hai ít nhất MARGIN,
     ngược lại hiển thị "Unknown".

Phím tắt: Q = thoát | R = nạp lại gallery

Chạy:  python lab_5.py
       python lab_5.py --threshold 0.6 --margin 0.03 --camera 1
       python lab_5.py --no-collected      (chỉ dùng ảnh trong data/reference)
       python lab_5.py --rebuild           (bỏ cache, tính lại toàn bộ embedding)
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# ----------------------------- Cấu hình -----------------------------
THRESHOLD = 0.7            # điểm tối thiểu để nhận ra một người (giữ giống các notebook)
MARGIN = 0.05              # người giống nhất phải hơn người thứ hai ít nhất giá trị này
TOP_K = 5                  # lấy trung bình K ảnh giống nhất của mỗi người
USE_COLLECTED = True       # dùng thêm ảnh trong data/collected làm gallery
MIN_FACE_PROB = 0.90       # bỏ qua vùng MTCNN không đủ tự tin là khuôn mặt
DETECT_EVERY = 2           # phát hiện + nhận diện mỗi N khung hình (tăng FPS trên CPU)
DETECT_WIDTH = 640         # thu nhỏ khung hình về chiều rộng này trước khi dò mặt
IMG_EXT = ('.jpg', '.jpeg', '.png')
CONDITIONS = ('normal', 'low_light', 'bright', 'near', 'far', 'left', 'right', 'expression')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(BASE_DIR, 'data', 'reference')
COLLECTED_DIR = os.path.join(BASE_DIR, 'data', 'collected')
CACHE_PATH = os.path.join(BASE_DIR, 'results', 'gallery_cache.npz')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# MTCNN: phát hiện mặt | FaceNet: trích đặc trưng (embedding 512 chiều)
mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


# ----------------------------- Embedding -----------------------------
def l2_normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def get_embeddings(faces):
    """faces: tensor (N,3,160,160) đã chuẩn hóa -> numpy (N,512) đã L2-normalize."""
    with torch.no_grad():
        emb = facenet(faces.to(device)).cpu().numpy()
    return l2_normalize(emb)


def embed_image_file(path):
    """Embedding của 1 file ảnh. Trả về (vector 512, True nếu MTCNN thấy mặt) hoặc (None, False)."""
    try:
        img = Image.open(path).convert('RGB')
    except Exception as error:
        print(f'  [bỏ qua] không đọc được {path}: {error}')
        return None, False

    boxes, probs = mtcnn.detect(img)
    if boxes is not None and len(boxes) > 0:
        # chọn khuôn mặt lớn nhất trong ảnh
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best = int(np.argmax(areas))
        face = mtcnn.extract(img, boxes[best:best + 1], None)[0]
        return get_embeddings(face.unsqueeze(0))[0], True

    # Ảnh trong data/collected vốn đã là ảnh mặt đã cắt sẵn nên MTCNN đôi khi không dò lại được.
    # Khi đó dùng cả ảnh, resize 160x160 và chuẩn hóa giống MTCNN: (x - 127.5) / 128.
    arr = np.asarray(img.resize((160, 160)), dtype=np.float32)
    face = torch.from_numpy((arr - 127.5) / 128.0).permute(2, 0, 1)
    return get_embeddings(face.unsqueeze(0))[0], False


# ----------------------------- Gallery theo người -----------------------------
def list_gallery_files(use_collected):
    """Trả về danh sách (tên người, đường dẫn ảnh). Tên người = tên thư mục ngay dưới reference/ hoặc collected/."""
    roots = [REFERENCE_DIR] + ([COLLECTED_DIR] if use_collected else [])
    files = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            folder = os.path.join(root, name)
            if not os.path.isdir(folder):
                continue
            if name in CONDITIONS:
                print(f'[cảnh báo] {folder} là thư mục kiểu cũ (chưa chia theo người), bỏ qua. '
                      f'Hãy chuyển vào {root}{os.sep}<tên người>{os.sep}{name}')
                continue
            for dirpath, _, filenames in os.walk(folder):
                for fn in sorted(filenames):
                    if fn.lower().endswith(IMG_EXT):
                        files.append((name, os.path.join(dirpath, fn)))
    return files


def file_signature(path):
    st = os.stat(path)
    return f'{st.st_size}_{int(st.st_mtime)}'


def build_gallery(use_collected=True, rebuild=False):
    """Trả về dict {tên người: ma trận embedding (n,512)}. Chỉ tính lại ảnh mới/đã đổi nhờ cache."""
    files = list_gallery_files(use_collected)
    if not files:
        raise FileNotFoundError(
            f'Không có ảnh nào. Hãy chạy collect_dataset.py để chụp, hoặc đặt ảnh vào '
            f'{REFERENCE_DIR}{os.sep}<tên người>{os.sep} hoặc {COLLECTED_DIR}{os.sep}<tên người>{os.sep}<điều kiện>{os.sep}')

    cache = {}
    if not rebuild and os.path.exists(CACHE_PATH):
        try:
            data = np.load(CACHE_PATH, allow_pickle=False)
            for p, s, e in zip(data['paths'], data['sigs'], data['embs']):
                cache[str(p)] = (str(s), e)
        except Exception as error:
            print('Cache lỗi, tính lại từ đầu:', error)

    by_person, cache_paths, cache_sigs, cache_embs = {}, [], [], []
    new_count = fallback_count = 0
    for person, path in files:
        key = os.path.relpath(path, BASE_DIR)
        sig = file_signature(path)
        if key in cache and cache[key][0] == sig:
            emb = cache[key][1]
        else:
            emb, detected = embed_image_file(path)
            if emb is None:
                continue
            new_count += 1
            fallback_count += 0 if detected else 1
        by_person.setdefault(person, []).append(emb)
        cache_paths.append(key)
        cache_sigs.append(sig)
        cache_embs.append(emb)

    if not by_person:
        raise ValueError('Không tạo được embedding nào từ gallery.')

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez(CACHE_PATH, paths=np.array(cache_paths), sigs=np.array(cache_sigs),
             embs=np.stack(cache_embs).astype(np.float32))

    gallery = {p: np.stack(e).astype(np.float32) for p, e in by_person.items()}
    summary = ', '.join(f'{p}: {len(e)}' for p, e in gallery.items())
    print(f'Gallery ({summary}) - {new_count} ảnh mới tính, {fallback_count} ảnh dùng resize thay vì MTCNN')
    return gallery


# ----------------------------- Tính điểm / nhận diện -----------------------------
def topk_mean(sims, top_k=TOP_K):
    k = min(top_k, len(sims))
    return float(np.mean(np.sort(sims)[-k:]))


def score_all(emb, gallery, top_k=TOP_K):
    """Điểm của một embedding với từng người: {tên: trung bình top_k cosine similarity}."""
    return {name: topk_mean(embs @ emb, top_k) for name, embs in gallery.items()}


def identify(scores, threshold=THRESHOLD, margin=MARGIN):
    """Trả về (tên người hoặc 'Unknown', điểm cao nhất)."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_name, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -1.0
    if best_score >= threshold and (best_score - second_score) >= margin:
        return best_name, best_score
    return 'Unknown', best_score


def print_gallery_report(gallery, threshold, margin):
    """Kiểm tra nhanh (tham khảo): mỗi ảnh trong gallery so với các ảnh còn lại (bỏ chính nó)."""
    names = list(gallery)
    X = np.concatenate([gallery[n] for n in names])
    labels = np.concatenate([[i] * len(gallery[n]) for i, n in enumerate(names)])
    if len(X) < 3:
        return

    sim = X @ X.T
    np.fill_diagonal(sim, np.nan)               # bỏ chính nó
    own_scores, impostor_scores, correct = [], [], 0
    for i in range(len(X)):
        scores = {}
        for j, n in enumerate(names):
            s = sim[i, labels == j]
            s = s[~np.isnan(s)]
            if len(s):
                scores[n] = topk_mean(s)
        if names[labels[i]] not in scores:
            continue
        own = scores[names[labels[i]]]
        others = [v for n, v in scores.items() if n != names[labels[i]]]
        own_scores.append(own)
        if others:
            impostor_scores.append(max(others))
        pred, _ = identify(scores, threshold, margin)
        correct += pred == names[labels[i]]

    own_scores = np.array(own_scores)
    print(f'Kiểm tra gallery (leave-one-out, chỉ để tham khảo): đúng người {correct / len(own_scores) * 100:.1f}% '
          f'| điểm cùng người: TB {own_scores.mean():.2f}, thấp nhất {own_scores.min():.2f}')
    if impostor_scores:
        print(f'  điểm nhầm sang người khác cao nhất: {max(impostor_scores):.2f} '
              f'-> ngưỡng nên nằm giữa {max(impostor_scores):.2f} và {np.percentile(own_scores, 5):.2f} (ngưỡng đang dùng {threshold})')
    else:
        print('  Mới có 1 người trong gallery nên chưa ước lượng được mức nhầm người.')


# ----------------------------- Webcam -----------------------------
def open_camera(index):
    cap = None
    if os.name == 'nt':                      # Windows: DirectShow mở camera nhanh và ổn định hơn
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f'Không mở được webcam (index {index}). Thử --camera 1.')
    return cap


def recognize_frame(frame, gallery, threshold, margin):
    """Trả về danh sách (x1, y1, x2, y2, tên, điểm) theo tọa độ khung hình gốc."""
    h, w = frame.shape[:2]
    scale = min(1.0, DETECT_WIDTH / w)
    small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
    img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

    boxes, probs = mtcnn.detect(img)
    if boxes is None:
        return []

    keep = [i for i, p in enumerate(probs) if p is not None and p >= MIN_FACE_PROB]
    if not keep:
        return []
    boxes = boxes[keep]

    faces = mtcnn.extract(img, boxes, None)          # crop + chuẩn hóa mặt, (N,3,160,160)
    embs = get_embeddings(faces)

    results = []
    for box, emb in zip(boxes, embs):
        x1, y1, x2, y2 = (box / scale).astype(int)
        name, score = identify(score_all(emb, gallery), threshold, margin)
        results.append((x1, y1, x2, y2, name, score))
    return results


def draw_results(frame, results):
    for x1, y1, x2, y2, name, score in results:
        known = name != 'Unknown'
        color = (0, 255, 0) if known else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{name} ({score:.2f})', (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)   # putText không vẽ được chữ có dấu -> dùng tên không dấu


def main():
    parser = argparse.ArgumentParser(description='Nhận diện khuôn mặt theo người (MTCNN + FaceNet)')
    parser.add_argument('--threshold', type=float, default=THRESHOLD)
    parser.add_argument('--margin', type=float, default=MARGIN)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--no-collected', action='store_true', help='chỉ dùng ảnh trong data/reference')
    parser.add_argument('--rebuild', action='store_true', help='bỏ cache, tính lại embedding')
    args = parser.parse_args()

    use_collected = USE_COLLECTED and not args.no_collected
    gallery = build_gallery(use_collected, rebuild=args.rebuild)
    print_gallery_report(gallery, args.threshold, args.margin)
    print('Nhấn Q để thoát, R để nạp lại gallery sau khi chụp thêm ảnh.')

    cap = open_camera(args.camera)
    frame_id, results, fps, last = 0, [], 0.0, time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % DETECT_EVERY == 0:
            results = recognize_frame(frame, gallery, args.threshold, args.margin)
        frame_id += 1

        draw_results(frame, results)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now
        cv2.putText(frame, f'FPS: {fps:.1f} | People: {len(gallery)} | Thr: {args.threshold}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow('FaceNet + MTCNN', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            gallery = build_gallery(use_collected)
            print_gallery_report(gallery, args.threshold, args.margin)
            results = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
