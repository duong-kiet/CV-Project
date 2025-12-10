import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from services.MLT import MLT
import torch
import cv2

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
    "./services/stage2_epoch_7_loss_1.1606_acc_0.5589.pth",
    map_location=device
))
emotion_model.to(device)
emotion_model.eval()


def predict_emotion(face_bgr):
    """
    Predict emotion từ ảnh khuôn mặt (BGR format từ OpenCV).
    
    Args:
        face_bgr: numpy array, ảnh khuôn mặt đã crop, format BGR
    
    Returns:
        Tuple (best_emotion, best_prob, emotion_probs)
        - best_emotion: str, cảm xúc mạnh nhất
        - best_prob: float, xác suất của cảm xúc mạnh nhất
        - emotion_probs: dict, xác suất của tất cả cảm xúc
    """
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(face_rgb)
    tens = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        emotion_output, _, _ = emotion_model(tens)

    # Softmax để lấy xác suất
    probs = F.softmax(emotion_output, dim=1)[0].cpu().numpy()

    # Emotion mạnh nhất
    idx = int(np.argmax(probs))
    best_emotion = expression_labels[idx]
    best_prob = float(probs[idx])

    # Map toàn bộ xác suất
    emotion_probs = {
        expression_labels[i]: float(probs[i])
        for i in range(len(expression_labels))
    }

    return best_emotion, best_prob, emotion_probs


def analyze_emotion(image: Image.Image):
    """
    Phân tích cảm xúc từ PIL Image.
    Sử dụng OpenCV Haar Cascade để detect face (thay vì DeepFace).
    
    Args:
        image: PIL Image
    
    Returns:
        dict với dominant_emotion, emotion probabilities, region, face_confidence
    """
    # PIL → numpy RGB → BGR
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # Detect face bằng OpenCV Haar Cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return None
    
    # Lấy face lớn nhất
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    
    face_img = img_bgr[y:y+h, x:x+w]
    if face_img.size == 0:
        return None
    
    # Predict emotion bằng MLT model
    emotion, prob, emotion_probs = predict_emotion(face_img)
    
    # Trả về format giống DeepFace
    result = {
        "dominant_emotion": emotion,
        "emotion": emotion_probs,
        "region": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        "face_confidence": prob
    }
    return result
