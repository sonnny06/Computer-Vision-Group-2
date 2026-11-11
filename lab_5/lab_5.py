import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

THRESHOLD = 0.7
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# MTCNN: phát hiện mặt | FaceNet: trích đặc trưng (embedding 512 chiều)
mtcnn = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embeddings(faces):
    """faces: tensor (N,3,160,160) -> numpy (N,512)"""
    with torch.no_grad():
        return facenet(faces.to(device)).cpu().numpy()

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# --- Embedding khuôn mặt tham chiếu ---
ref_img = Image.open('image/reference.jpg').convert('RGB')
ref_faces = mtcnn(ref_img)
if ref_faces is None:
    raise ValueError("Không tìm thấy mặt trong reference.jpg")
ref_emb = get_embeddings(ref_faces[:1])[0]

# --- Webcam realtime ---
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    boxes, _ = mtcnn.detect(img)
    if boxes is not None:
        faces = mtcnn.extract(img, boxes, None)   # crop + chuẩn hóa mặt
        embs = get_embeddings(faces)

        for box, emb in zip(boxes, embs):
            sim = cosine_sim(ref_emb, emb)
            matched = sim > THRESHOLD
            label = f"Matched ({sim:.2f})" if matched else f"Unknown ({sim:.2f})"
            color = (0, 255, 0) if matched else (0, 0, 255)

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('FaceNet + MTCNN', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
