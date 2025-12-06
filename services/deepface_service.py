import os

# Workaround cho xung đột protobuf mới với TensorFlow cũ (DeepFace dùng TF 2.10.x).
# Tham khảo gợi ý từ thông báo lỗi protobuf:
#   PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
from deepface import DeepFace
from PIL import Image
from deepface.modules import detection
from torchvision import transforms
from PIL import Image

from services.MLT import MLT
import torch
import cv2

import streamlit as st
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

expression_labels = [
    "Neutral", "Happy", "Sad", "Surprise",
    "Fear", "Disgust", "Anger", "Contempt"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

emotion_model = MLT()
emotion_model.load_state_dict(torch.load(
    "./stage2_epoch_7_loss_1.1606_acc_0.5589.pth",
    map_location=device
))
emotion_model.to(device)
emotion_model.eval()


def predict_emotion(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(face_rgb)
    tens = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        emotion_output, _, _ = emotion_model(tens)

    idx = torch.argmax(emotion_output).item()
    return expression_labels[idx]

def analyze_emotion(image: Image.Image):
    """
    Phân tích cảm xúc bằng PyTorch MLT/OpenFace model + DeepFace detector.
    Trả về dict giống DeepFace.
    """

    # PIL → numpy RGB
    img = np.array(image)

    # Detect face bằng DeepFace
    try:
        faces = detection.extract_faces(
            img_path=img,
            detector_backend="opencv",
            enforce_detection=False,
            align=True
        )
    except:
        return None

    if len(faces) == 0:
        return None

    # Lấy mặt lớn nhất hoặc mặt đầu tiên (Streamlit thường chỉ có 1)
    f = faces[0]
    fa = f["facial_area"]
    x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]

    face_img = img[y:y+h, x:x+w]
    if face_img.size == 0:
        return None

    # ------- Predict emotion bằng PyTorch -------
    emotion = predict_emotion(face_img)

    # ------ TRẢ VỀ FORMAT GIỐNG DEEPFACE -------
    result = {
        "dominant_emotion": emotion,
        "emotion": {
            lbl: (100.0 if lbl == emotion else 0.0)
            for lbl in expression_labels
        },
        "region": fa,
        "face_confidence": f.get("confidence", 1.0)
    }
    #
    return result

# def analyze_emotion(image: Image.Image):
#     """
#     Phân tích cảm xúc từ ảnh (PIL Image) bằng DeepFace.
#     Trả về kết quả phân tích hoặc None nếu có lỗi.
#     """
#     try:
#         # Chuyển PIL Image -> numpy array (RGB)
#         img_rgb = np.array(image)
#
#         # DeepFace chấp nhận numpy array (BGR hoặc RGB tùy backend),
#         # ở đây ta truyền trực tiếp array RGB.
#         result = DeepFace.analyze(
#             img_path=img_rgb,
#             actions=["emotion"],
#             enforce_detection=False,
#         )
#
#         # DeepFace có thể trả về list hoặc dict tùy version
#         if isinstance(result, list) and len(result) > 0:
#             result = result[0]
#         return result
#     except Exception as e:
#         st.error(f"Lỗi khi phân tích cảm xúc: {e}")
#         return None



